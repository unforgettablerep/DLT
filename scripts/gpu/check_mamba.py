"""Can S-Mamba use mamba_ssm on this machine?

    python scripts/gpu/check_mamba.py

S-Mamba (models/SMamba.py) imports mamba_ssm from the environment, else from the source checkouts next to the
repository (scripts/paths.py: ../mamba and ../causal-conv1d), else it falls back to the pure-PyTorch Mamba block
of layers/MambaPy.py (same model, much slower). This script shows
  1. Python, PyTorch, the CUDA version PyTorch was built for, the C++ ABI and the nvcc on the PATH,
  2. the two source checkouts: version, and whether their CUDA extensions are compiled,
  3. whether mamba_ssm and causal_conv1d import, and from where (or the exact error),
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


def compiled(src, stem):
    return sorted(str(p.relative_to(src)) for p in src.rglob(f'{stem}*.so'))


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
for src, pkg, ext in ((paths.MAMBA_SRC, 'mamba_ssm', 'selective_scan_cuda'),
                      (paths.CAUSAL_CONV1D_SRC, 'causal_conv1d', 'causal_conv1d_cuda')):
    if not src.is_dir():
        print(f'{src}: not present')
        continue
    so = compiled(src, ext)
    print(f'{src}: {pkg} {version_of(src / pkg / "__init__.py")}; compiled extension: '
          + (', '.join(so[:2]) if so else f'none ({ext}*.so not found: not built in place)'))

# ---- 3. imports
print('\n== imports')
try:
    Mamba, impl, note = SMamba.mamba_block('ssm')
    origin = Path(sys.modules['mamba_ssm'].__file__).parent
    where = 'source checkout' if paths.MAMBA_SRC in origin.parents or origin.parent == paths.MAMBA_SRC else 'environment'
    print(f'mamba_ssm {getattr(sys.modules["mamba_ssm"], "__version__", "?")}: imported from the {where} ({note})')
except ImportError as e:
    Mamba, err = None, str(e)
    print(f'mamba_ssm: NOT importable\n  {err}')
    low = err.lower()
    missing = re.findall(r"no module named '([\w.]+)'", low)
    others = [m for m in missing if m.split('.')[0] not in ('mamba_ssm', 'selective_scan_cuda', 'causal_conv1d_cuda')]
    if others:
        issues.append(f'install the missing Python package(s): pip install {" ".join(sorted({m.split(".")[0] for m in others}))}')
    if 'selective_scan_cuda' in low or 'causal_conv1d_cuda' in low or (paths.MAMBA_SRC.is_dir() and not compiled(paths.MAMBA_SRC, 'selective_scan_cuda')):
        issues.append('the CUDA extensions are not built')
    if 'undefined symbol' in low:
        issues.append('the extension was built for another PyTorch / CUDA / ABI: reinstall it for this torch')
    if not paths.MAMBA_SRC.is_dir() and 'mamba_ssm' in missing:
        issues.append('mamba_ssm is neither installed nor present at ' + str(paths.MAMBA_SRC))
try:
    import causal_conv1d  # noqa: F401
    print(f'causal_conv1d: imported from {Path(causal_conv1d.__file__).parent} (fused fast path available)')
except ImportError as e:
    print(f'causal_conv1d: not importable ({e}); mamba_ssm then runs its unfused path (correct, a little slower)')

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
        blk = Mamba(64, d_state=16, d_conv=2, expand=1, use_fast_path=fast).to(dev)
        blk.load_state_dict(ref.state_dict())
        ys, gs = grads(blk)
        e_out = ((ys - yr).abs().max() / yr.abs().max()).item()
        e_grad = max((gs[n] - gr[n]).abs().max().item() / max(gr[n].abs().max().item(), 1e-3 * G) for n in gr)
        ok = e_out < 1e-4 and e_grad < 1e-4
        print(f"{'PASS' if ok else 'FAIL'}  {'fused' if fast else 'unfused'} path: output {e_out:.1e}, gradients {e_grad:.1e}")
        if not ok:
            issues.append(f'mamba_ssm ({"fused" if fast else "unfused"} path) does not match the PyTorch block')

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
if Mamba is not None and cuda and not issues:
    print('S-Mamba will use mamba_ssm (--mamba_impl auto). Nothing to do.')
elif Mamba is not None and not cuda:
    print('mamba_ssm imports, but there is no GPU here: S-Mamba runs on the CPU use the PyTorch block.')
else:
    print('S-Mamba will use the pure-PyTorch block: correct, but slow on Electricity and Traffic.')
    for i in dict.fromkeys(issues):
        print(f'- {i}')
    if 'the CUDA extensions are not built' in issues and torch_major:
        print(f'  Build and install both checkouts into this environment (their setup.py first downloads a prebuilt\n'
              f'  wheel for torch {torch.__version__.split("+")[0]} / CUDA {torch_major} / this Python, and compiles only if none fits):\n'
              f'    pip install --no-build-isolation {paths.CAUSAL_CONV1D_SRC}\n'
              f'    pip install --no-build-isolation {paths.MAMBA_SRC}')
        if nvcc_ver and torch_major and nvcc_ver[0] != torch_major:
            print(f'  Compiling needs an nvcc of CUDA {torch_major}.x: PyTorch built for CUDA {torch.version.cuda} refuses the\n'
                  f'  nvcc of CUDA {nvcc_ver[0]}.{nvcc_ver[1]} on the PATH. Install a CUDA {torch_major} toolkit into the environment\n'
                  f'  (for example conda install -c nvidia cuda-toolkit={torch.version.cuda}) and set CUDA_HOME to it before\n'
                  f'  building, or use the prebuilt wheels.')
    print('Then run this script again.')
sys.exit(0 if (Mamba is not None and not issues) else 1)
