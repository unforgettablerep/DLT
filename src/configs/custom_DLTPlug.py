def add_dltplug_parser(parser):
    parser.add_argument("--plug_base", type=str, default="DLinear", help="backbone the DLT branch is added to")
    parser.add_argument("--dlt_lr", type=float, default=0.0, help="learning rate of the DLT branch (0 = --learning_rate)")
    parser.add_argument("--dlt_fusion", type=str, default="output", choices=["output", "refine", "input"], help="how the branch enters the backbone")
    parser.add_argument("--dlt_train", type=str, default="joint", choices=["joint", "decoupled"], help="joint loss, or the backbone first and the branch on its error")
    parser.add_argument("--dlt_select", type=str, default="none", choices=["none", "variable"], help="keep the correction only where it lowers the validation MSE")
