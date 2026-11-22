from src.configs.common import add_common_args
from src.configs.custom_DLT import add_dlt_parser
from src.configs.custom_DLTPlug import add_dltplug_parser
from src.configs.custom_PatchTST import add_patchtst_parser
from src.configs.custom_S_Mamba import add_s_mamba_parser
from src.configs.custom_TiDE import add_tide_parser

model_parser_dict = {
    "__all__": [add_common_args],
    "DLT": [add_dlt_parser],
    "DLTPlug": [add_dlt_parser, add_dltplug_parser],
    "S_Mamba": [add_s_mamba_parser],
    "iTransformer": [],
    "PatchTST": [add_patchtst_parser],
    "RLinear": [],
    "DLinear": [],
    "TiDE": [add_tide_parser],
}

model_aliases = {
    "SMamba": "S_Mamba",
    "S-Mamba": "S_Mamba",
}


def canonical_model(name):
    return model_aliases.get(name, name)
