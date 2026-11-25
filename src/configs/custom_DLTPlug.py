def add_dltplug_parser(parser):
    """Options of the plug-in only; --dlt_lr, --dlt_train and --dlt_select are shared with DLT (custom_DLT.py)."""
    parser.add_argument("--plug_base", type=str, default="DLinear", help="backbone the DLT branch is added to")
    parser.add_argument("--dlt_fusion", type=str, default="output", choices=["output", "refine", "input"], help="how the branch enters the backbone")
