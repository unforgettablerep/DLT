def add_s_mamba_parser(parser):
    parser.add_argument("--d_state", type=int, default=32, help="state size of the Mamba blocks")
    parser.add_argument("--mamba_impl", type=str, default="auto", choices=["auto", "ssm", "torch"], help="mamba_ssm block, pure-PyTorch block, or mamba_ssm if usable")
