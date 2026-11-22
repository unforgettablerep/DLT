"""Can S-Mamba use mamba_ssm on this machine?

    python scripts/utils/check_mamba.py

Reference GPU environment (README.md, Installation): host CUDA 13.0, PyTorch 2.10.0+cu130, and mamba_ssm 2.3.2.post1 and
causal_conv1d 1.7.0 from their prebuilt wheels for CUDA 13 + torch 2.10.

S-Mamba (src/model/S_Mamba.py) imports mamba_ssm from the environment, else from the source checkouts next to the
repository (src/utils/paths.py: ../mamba and ../causal-conv1d), else it falls back to the pure-PyTorch Mamba block
of src/layers/MambaPy.py (same model, much slower). This script shows
  1. Python, PyTorch, the CUDA version and C++ ABI PyTorch was built with, the NVIDIA driver and the nvcc on the PATH,
  2. the version of the two source checkouts,
  3. whether the compiled extensions (selective_scan_cuda, causal_conv1d_cuda) and the Mamba block of mamba_ssm
     import, and from where (or the exact error); `transformers` is not needed (S-Mamba imports the block
     without mamba_ssm/__init__.py),
  4. on a GPU: one Mamba block of mamba_ssm (fused and unfused path) against the PyTorch block with the same
     parameters (outputs and gradients, float32, TF32 off), and the time of both at the size of S-Mamba on
     Electricity (batch 16, 325 tokens, d_model 512, d_state 16),
and ends with what S-Mamba will use and, if mamba_ssm is not usable, what to install.
"""
import copy, importlib, os, platform, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
import torch  # noqa: E402
from src.utils import paths  # noqa: E402
from src.layers.Mamba_EncDec import mamba_block  # noqa: E402
from src.layers.MambaPy import Mamba as TorchMamba  # noqa: E402

issues = []
REF_TORCH, REF_INDEX = '2.10.0', 'https://download.pytorch.org/whl/cu130'     # reference environment (README.md, Installation)


def version_of(init_py):
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)', init_py.read_text()) if init_py.exists() else None
    return m.group(1) if m else 'unknown'


# ---- 1. environment
print('== environment')
print(f'python {platform.python_version()}, torch {torch.__version__}, built for CUDA {torch.version.cuda}, '
      f'C++11 ABI {torch._C._GLIBCXX_USE_CXX11_ABI}')
cuda = torch.cuda.is_available()
print(f'GPU: {torch.cuda.get_device_name(0) if cuda else "none (CUDA not available)"}')
smi = shutil.which('nvidia-smi')
if smi:
    out = subprocess.run([smi], capture_output=True, text=True, timeout=60).stdout
    drv, top = (re.search(rf'{k}:\s*([\d.]+)', out) for k in ('Driver Version', 'CUDA Version'))
    print(f'NVIDIA driver {drv.group(1) if drv else "?"}, supports CUDA up to {top.group(1) if top else "?"}')
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
    issues.append(f'this PyTorch is a CPU-only build: install the CUDA build first, '
                  f'pip install torch=={REF_TORCH} --index-url {REF_INDEX}')
elif not cuda and smi:
    issues.append(f'PyTorch is built for CUDA {torch.version.cuda} but sees no GPU: the NVIDIA driver must support '
                  f'CUDA {torch.version.cuda} (nvidia-smi shows the highest CUDA version it supports)')

# ---- 2. source checkouts
print('\n== source checkouts next to the repository (src/utils/paths.py)')
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
                issues.append('the CUDA kernel of mamba_ssm (selective_scan_cuda) is not built or not installed')
try:
    Mamba, impl, note = mamba_block('ssm')
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
TV = '.'.join(torch.__version__.split('+')[0].split('.')[:2])                  # e.g. 2.10
PY = f'cp{sys.version_info.major}{sys.version_info.minor}'
ABI = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
# mamba_ssm / causal_conv1d pairs released together, with prebuilt wheels (checked 2026-11):
#   CUDA 13 + torch 2.9 or 2.10: Python 3.10-3.13, cxx11abiTRUE (the C++ ABI of the pip builds of torch >= 2.7)
#   CUDA 12 + torch 2.5:         Python 3.10-3.12, cxx11abiFALSE (the C++ ABI of the pip build of torch 2.5)
WHEELS = {(13, '2.10'): ('2.3.2.post1', '1.7.0'), (13, '2.9'): ('2.3.2.post1', '1.7.0'),
          (12, '2.5'): ('2.2.4', '1.5.0.post8')}
known = WHEELS.get((torch_major, TV))
PAIR = dict(zip(('mamba_ssm', 'causal_conv1d'), known)) if known else versions
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
    if torch_major and (not ext_ok.get('selective_scan_cuda') or any('causal_conv1d' in i for i in issues)):
        if not known:
            print(f'  No wheel pair is known for torch {TV} + CUDA {torch_major}. The reference environment is PyTorch\n'
                  f'  {REF_TORCH}+cu130: pip install torch=={REF_TORCH} --index-url {REF_INDEX}')
        if 'mamba_ssm' in PAIR:
            print('  Install the prebuilt wheels (no compiler needed; download them elsewhere and copy them here if this\n'
                  '  machine cannot reach github.com, then pip install the files):')
            print(f'    pip install --force-reinstall --no-deps {wheel("mamba_ssm")}')
            if 'causal_conv1d' in PAIR:
                print(f'    pip install --force-reinstall --no-deps {wheel("causal_conv1d")}    # optional: fused path')
            print('  They go into the environment, so the checkouts next to the repository are no longer used. --no-deps\n'
                  '  keeps pip from installing transformers and the other dependencies of mamba_ssm that S-Mamba does\n'
                  '  not need.' + ('' if known else ' If a file does not exist, the release page of that version lists its wheels.'))
        if nvcc_ver and nvcc_ver[0] == torch_major:
            print(f'  Or build both from the checkouts with the nvcc on the PATH (CUDA {nvcc_ver[0]}.{nvcc_ver[1]}; this compiles for\n'
                  f'  every GPU architecture and takes long):\n'
                  f'    pip install ninja packaging\n'
                  f'    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 pip install --no-build-isolation --no-deps {paths.CAUSAL_CONV1D_SRC}\n'
                  f'    MAMBA_FORCE_BUILD=TRUE MAMBA_KEEP_CUDA_BUILD=TRUE MAX_JOBS=4 \\\n'
                  f'        pip install --no-build-isolation --no-deps {paths.MAMBA_SRC}\n'
                  f'  (since mamba_ssm 2.3 a source install leaves out the selective-scan kernel unless\n'
                  f'  MAMBA_KEEP_CUDA_BUILD=TRUE; FORCE_BUILD skips the wheel download)')
        elif nvcc_ver:
            print(f'  Building from the checkouts needs an nvcc of CUDA {torch_major}.x: PyTorch built for CUDA '
                  f'{torch.version.cuda} refuses\n  the nvcc of CUDA {nvcc_ver[0]}.{nvcc_ver[1]} on the PATH. Use the PyTorch build '
                  f'for the CUDA of the host' + (
                      f':\n    pip install torch=={REF_TORCH} --index-url {REF_INDEX}'
                      if nvcc_ver[0] == 13 else '.'))
    print('Then run this script again.')
sys.exit(0 if (Mamba is not None and not issues) else 1)
