def add_tide_parser(parser):
    parser.add_argument("--tide_checkpoint", action="store_true", help="recompute each variable's forward pass in the backward pass (same results, far less memory)")
