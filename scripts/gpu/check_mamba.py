"""Can S-Mamba use mamba_ssm on this machine?

    python scripts/gpu/check_mamba.py

S-Mamba (models/SMamba.py) imports mamba_ssm from the environment, else from the source checkouts next to the
repository (scripts/paths.py: ../mamba and ../causal-conv1d), else it falls back to the pure-PyTorch Mamba block
of layers/MambaPy.py (same model, much slower). This script shows
  1. Python, PyTorch, the CUDA version PyTorch was built for, the C++ ABI and the nvcc on the PATH,
  2. the version of the two source checkouts,
  3. whether the compiled extensions (selective_scan_cuda, causal_conv1d_cuda) and the Mamba block of mamba_ssm
     import, and from where (or the exact error); `transformers` is not needed (S-Mamba imports the block
     without mamba_ssm/__init__.py),
  4. on a GPU: one Mamba block of mamba_ssm (fused and unfused path) against the PyTorch block with the same
     parameters (outputs and gradients, float32, TF32 off), and the time of both at the size of S-Mamba on
     Electricity (batch 16, 325 tokens, d_model 512, d_state 16),
and ends with what S-Mamba will use and, if mamba_ssm is not usable, what to do.
"""
import copy, importlib, os, platform, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
os.chdir(REPO)
import torch  # noqa: E402
import paths  # noqa: E402
from models import SMamba  # noqa: E402
from layers.MambaPy import Mamba as TorchMamba  # noqa: E402

issues = []


def version_of(init_py):
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)', init_py.read_text()) if init_py.exists() else None
    return m.group(1) if m else 'unknown'


# ---- 1. environment
print('== environment')
print(f'python {platform.python_version()}, torch {torch.__version__}, built for CUDA {torch.version.cuda}, '
      f'C++11 ABI {torch._C._GLIBCXX_USE_CXX11_ABI}')
cuda = torch.cuda.is_available()
print(f'GPU: {torch.cuda.get_device_name(0) if cuda else "none (CUDA not available)"}')
nvcc = shutil.which('nvcc')
nvcc_ver = None
if nvcc:
    out = subprocess.run([nvcc, '--version'], capture_output=True, text=True).stdout
    m = re.search(r'release (\d+)\.(\d+)', out)
    nvcc_ver = (int(m.group(1)), int(m.group(2))) if m else None
print(f'nvcc: {nvcc} (CUDA {".".join(map(str, nvcc_ver)) if nvcc_ver else "?"})' if nvcc else 'nvcc: not on the PATH')
if os.environ.get('CUDA_HOME'):
    print(f'CUDA_HOME: {os.environ["CUDA_HOME"]}')
torch_major = int(torch.version.cuda.split('.')[0]) if torch.version.cuda else None
if not torch.version.cuda:
    issues.append('this PyTorch is a CPU-only build: install the CUDA build first, '
                  'pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124')

# ---- 2. source checkouts
print('\n== source checkouts next to the repository (scripts/paths.py)')
versions = {}
for src, pkg in ((paths.MAMBA_SRC, 'mamba_ssm'), (paths.CAUSAL_CONV1D_SRC, 'causal_conv1d')):
    if src.is_dir():
        versions[pkg] = version_of(src / pkg / '__init__.py')
        print(f'{src}: {pkg} {versions[pkg]}')
    else:
        print(f'{src}: not present')

# ---- 3. imports: the compiled extensions first, then the Mamba block of mamba_ssm
print('\n== imports')
sources = [str(d) for d in (paths.CAUSAL_CONV1D_SRC, paths.MAMBA_SRC) if d.is_dir()]
ext_ok = {}
for ext, what in (('selective_scan_cuda', 'selective scan of mamba_ssm (required)'),
                  ('causal_conv1d_cuda', 'causal_conv1d (optional: fused fast path)')):
    try:
        mod = importlib.import_module(ext)
        ext_ok[ext] = True
        print(f'{ext}: OK ({mod.__file__}) - {what}')
    except ImportError as e:
        sys.path[:0] = [d for d in sources if d not in sys.path]       # built in place inside a checkout?
        try:
            mod = importlib.import_module(ext)
            ext_ok[ext] = True
            print(f'{ext}: OK ({mod.__file__}) - {what}')
        except ImportError as e2:
            ext_ok[ext] = False
            print(f'{ext}: NOT importable ({e2}) - {what}')
            if 'undefined symbol' in str(e2):
                issues.append(f'{ext} was built for another PyTorch / CUDA / C++ ABI: reinstall it for torch {torch.__version__}')
            elif ext == 'selective_scan_cuda':
                issues.append('the CUDA extension of mamba_ssm is not built or not installed')
try:
    Mamba, impl, note = SMamba.mamba_block('ssm')
    print(f'Mamba block: OK ({note})')
except ImportError as e:
    Mamba, err = None, str(e)
    print(f'Mamba block of mamba_ssm: NOT importable\n  {err}')
    missing = re.findall(r"no module named '([\w.]+)'", err.lower())
    others = sorted({m.split('.')[0] for m in missing} - {'mamba_ssm', 'selective_scan_cuda', 'causal_conv1d_cuda', 'causal_conv1d'})
    if others:
        issues.append(f'install the missing Python package(s): pip install {" ".join(others)}')
    if 'mamba_ssm' in missing and not paths.MAMBA_SRC.is_dir():
        issues.append('mamba_ssm is neither installed nor present at ' + str(paths.MAMBA_SRC))
    if not issues:
        issues.append('the Mamba block of mamba_ssm does not import (error above)')
try:
    import transformers
    print(f'transformers {transformers.__version__}: installed (not needed by S-Mamba)')
except ImportError:
    print('transformers: not installed - not needed: S-Mamba imports the Mamba block without mamba_ssm/__init__.py')

# ---- 4. GPU: equality and speed
if Mamba is not None and cuda:
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    print('\n== one Mamba block: mamba_ssm vs PyTorch block, same parameters (float32, TF32 off)')
    torch.manual_seed(0)
    ref = TorchMamba(64, d_state=16, d_conv=2, expand=1).to(dev)
    x = torch.randn(4, 37, 64, device=dev)

    def grads(model):
        model.zero_grad()
        y = model(x)
        (y ** 2).mean().backward()
        return y.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    yr, gr = grads(ref)
    G = max(g.abs().max().item() for g in gr.values())
    for fast in (True, False):
        path = 'fused' if fast else 'unfused'
        try:
            blk = Mamba(64, d_state=16, d_conv=2, expand=1, use_fast_path=fast).to(dev)
            blk.load_state_dict(ref.state_dict())
            ys, gs = grads(blk)
        except Exception as e:       # e.g. a causal_conv1d whose interface does not fit this mamba_ssm
            print(f'FAIL  {path} path: {type(e).__name__}: {e}')
            issues.append(f'the {path} path of mamba_ssm fails' + (
                ' (causal_conv1d does not fit this mamba_ssm: pip uninstall -y causal_conv1d, or install the pair below)'
                if fast else ''))
            continue
        e_out = ((ys - yr).abs().max() / yr.abs().max()).item()
        e_grad = max((gs[n] - gr[n]).abs().max().item() / max(gr[n].abs().max().item(), 1e-3 * G) for n in gr)
        ok = e_out < 1e-4 and e_grad < 1e-4
        print(f"{'PASS' if ok else 'FAIL'}  {path} path: output {e_out:.1e}, gradients {e_grad:.1e}")
        if not ok:
            issues.append(f'mamba_ssm ({path} path) does not match the PyTorch block')

if Mamba is not None and cuda and not issues:
    print('\n== time of one block, forward + backward (batch 16, 325 tokens, d_model 512, d_state 16)')
    xb = torch.randn(16, 325, 512, device=dev)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    times = {}
    for name, blk in (('mamba_ssm', Mamba(512, d_state=16, d_conv=2, expand=1)), ('PyTorch block', TorchMamba(512, d_state=16, d_conv=2, expand=1))):
        blk = blk.to(dev)
        for i in range(13):
            if i == 3:
                torch.cuda.synchronize(); t = time.time()
            blk(xb).pow(2).mean().backward()
        torch.cuda.synchronize()
        times[name] = (time.time() - t) / 10 * 1e3
        print(f'{name:14s} {times[name]:8.1f} ms')
    print(f'S-Mamba on Electricity has 6 such blocks per step (3 layers, both directions): the PyTorch block is '
          f'{times["PyTorch block"] / times["mamba_ssm"]:.0f}x slower.')

# ---- verdict
print('\n== verdict')
TV = '.'.join(torch.__version__.split('+')[0].split('.')[:2])                  # e.g. 2.5
PY = f'cp{sys.version_info.major}{sys.version_info.minor}'
ABI = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
# a pair released together, with prebuilt wheels for torch 2.5 + CUDA 12 (checked 2026-11 for Python 3.10-3.12 and
# the C++ ABI of the pip build of torch, cxx11abiFALSE); for another PyTorch the versions of the checkouts are proposed
PAIR = {'mamba_ssm': '2.2.4', 'causal_conv1d': '1.5.0.post8'} if TV == '2.5' else versions
REPOS = {'mamba_ssm': 'state-spaces/mamba', 'causal_conv1d': 'Dao-AILab/causal-conv1d'}


def wheel(pkg):
    """the prebuilt wheel that the setup.py of the package downloads for this environment (same naming scheme)"""
    v = PAIR[pkg]
    return (f'https://github.com/{REPOS[pkg]}/releases/download/v{v}/'
            f'{pkg}-{v}+cu{torch_major}torch{TV}cxx11abi{ABI}-{PY}-{PY}-linux_x86_64.whl')


if Mamba is not None and cuda and not issues:
    print('S-Mamba will use mamba_ssm (--mamba_impl auto). Nothing to do.')
elif Mamba is not None and not cuda and not issues:
    print('mamba_ssm imports, but there is no GPU here: S-Mamba runs on the CPU use the PyTorch block.')
else:
    if Mamba is not None:
        print('S-Mamba would use mamba_ssm (--mamba_impl auto), but:')
    else:
        print('S-Mamba will use the pure-PyTorch block: correct, but slow on Electricity and Traffic.')
    for i in dict.fromkeys(issues):
        print(f'- {i}')
    if Mamba is not None:
        print('Until this is resolved, run S-Mamba with --mamba_impl torch.')
    if torch_major and (not ext_ok.get('selective_scan_cuda') or any('causal_conv1d' in i for i in issues)) \
            and 'mamba_ssm' in PAIR:
        print('  Install the prebuilt wheels (no compiler needed; download them elsewhere and copy them here if this\n'
              '  machine cannot reach github.com, then pip install the files):')
        print(f'    pip install --force-reinstall --no-deps {wheel("mamba_ssm")}')
        if 'causal_conv1d' in PAIR:
            print(f'    pip install --force-reinstall --no-deps {wheel("causal_conv1d")}    # optional: fused path')
        print('  They go into the environment, so the checkouts next to the repository are no longer used. --no-deps\n'
              '  keeps pip from installing transformers, which mamba_ssm lists but S-Mamba does not need.')
        if TV != '2.5':
            print('  If a file does not exist, the release page of that version lists its wheels.')
        if nvcc_ver and nvcc_ver[0] != torch_major:
            print(f'  Building from the checkouts instead needs an nvcc of CUDA {torch_major}.x: PyTorch built for CUDA\n'
                  f'  {torch.version.cuda} refuses the nvcc of CUDA {nvcc_ver[0]}.{nvcc_ver[1]} on the PATH. Install a CUDA '
                  f'{torch.version.cuda} toolkit into this\n'
                  f'  environment and build with it (MAMBA_FORCE_BUILD skips the wheel download):\n'
                  f'    conda install -c nvidia/label/cuda-{torch.version.cuda}.1 cuda-toolkit\n'
                  f'    pip install ninja packaging\n'
                  f'    CUDA_HOME=$CONDA_PREFIX MAMBA_FORCE_BUILD=TRUE MAX_JOBS=8 \\\n'
                  f'        pip install --no-build-isolation --no-deps {paths.MAMBA_SRC}\n'
                  f'  (if a CUDA header is not found: export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include)')
    print('Then run this script again.')
sys.exit(0 if (Mamba is not None and not issues) else 1)
