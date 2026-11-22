import importlib


def _load(name):
    try:
        module = importlib.import_module(".{}".format(name), __name__)
        return module.Model
    except Exception as exc:
        error = exc

        def _missing(*args, **kwargs):
            raise ImportError("Model {} is unavailable: {}".format(name, error)) from error

        return _missing


DLT = _load("DLT")
DLTPlug = _load("DLTPlug")
S_Mamba = _load("S_Mamba")
iTransformer = _load("iTransformer")
PatchTST = _load("PatchTST")
RLinear = _load("RLinear")
DLinear = _load("DLinear")
TiDE = _load("TiDE")
