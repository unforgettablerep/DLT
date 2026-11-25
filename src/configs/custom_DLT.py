from src.layers.DLT_layers import GATES, IMPLS, NEIGHBOUR_MODES, VARIANTS


def add_dlt_parser(parser):
    parser.add_argument("--dlt_variant", type=str, default="dltdyn", choices=list(VARIANTS), help="parameterisation of the cross-variable lag kernel")
    parser.add_argument("--dlt_k", type=int, default=8, help="source neighbours per variable")
    parser.add_argument("--dlt_L", type=int, default=48, help="maximum lag of the cross-variable kernels")
    parser.add_argument("--dlt_R", type=int, default=4, help="number of shared lag-basis kernels")
    parser.add_argument("--dlt_dyn_dim", type=int, default=16, help="summary dimension of the input-adaptive coefficients")
    parser.add_argument("--dlt_ccf_len", type=int, default=8000, help="most recent training steps used for neighbour selection")
    parser.add_argument("--dlt_impl", type=str, default="kernel", choices=list(IMPLS), help="computation path of dlt / dltdyn (identical forecasts)")
    parser.add_argument("--dlt_neighbours", type=str, default="ccf", choices=list(NEIGHBOUR_MODES), help="neighbour selection: CCF top-k, k random sources, all other variables")
    parser.add_argument("--dlt_topk", type=int, default=5, help="lags per pair of the MillGNN-style variants topk / mill / milldyn")
    parser.add_argument("--dlt_gate", type=str, default="none", choices=list(GATES), help="combination of the two branches: none, moe (MoVE-style), static (per-variable sigmoid)")
    parser.add_argument("--dlt_gate_init", type=float, default=-3.0, help="initial logit of the static gate")
    parser.add_argument("--dlt_hp_trial", type=int, default=-1, help="bookkeeping: index of a search trial")
    parser.add_argument("--dlt_lr", type=float, default=0.0, help="learning rate of the cross-variable branch (0 = --learning_rate)")
    parser.add_argument("--dlt_train", type=str, default="joint", choices=["joint", "decoupled"],
                        help="joint loss, or the backbone (DLT: the self head) first and the branch on its error")
    parser.add_argument("--dlt_select", type=str, default="none", choices=["none", "variable"],
                        help="keep the cross-variable correction only for the variables whose validation MSE it lowers")
