import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np

def dlt_suffix(args):
    """Experiment-name suffix of DLT runs: keeps checkpoints / results of different configurations apart."""
    suf = '_{}_k{}_L{}_R{}'.format(args.dlt_variant, args.dlt_k, args.dlt_L, args.dlt_R)
    if args.model == 'DLTPlug':
        suf = '_{}'.format(args.plug_base) + suf
        if args.dlt_lr > 0:
            suf += '_dlr{:g}'.format(args.dlt_lr)
        if args.dlt_fusion != 'output':
            suf += '_f{}'.format(args.dlt_fusion)
        if args.dlt_train != 'joint':
            suf += '_t{}'.format(args.dlt_train)
    if args.dlt_neighbours != 'ccf':
        suf += '_nb{}'.format(args.dlt_neighbours)
    if args.dlt_variant in ('topk', 'mill', 'milldyn'):
        suf += '_K{}'.format(args.dlt_topk)
    if args.dlt_gate != 'none':
        suf += '_g{}'.format(args.dlt_gate)
        if args.dlt_gate == 'static':
            suf += '{:g}'.format(args.dlt_gate_init)
    return suf


def get_parser():
    """All command-line options (module level so that other scripts can build the same configs)."""
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--d_state', type=int, default=32, help='S-Mamba: state size of the Mamba blocks')
    parser.add_argument('--mamba_impl', type=str, default='auto', choices=['auto', 'ssm', 'torch'],
                        help='S-Mamba: mamba_ssm block (ssm), pure-PyTorch block (torch), or ssm if installed (auto)')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # DLT (Distributed-Lag Transfer)
    parser.add_argument('--dlt_variant', type=str, default='dltdyn',
                        choices=['ci', 'scalar', 'shift', 'tap3', 'dlt', 'full', 'scalardyn', 'dltdyn',
                                 'topk', 'mill', 'milldyn'],
                        help='DLT: parameterisation of the cross-variable lag kernel')
    parser.add_argument('--dlt_k', type=int, default=8, help='DLT: number of source neighbours per variable')
    parser.add_argument('--dlt_L', type=int, default=48, help='DLT: maximum lag of the cross-variable kernels')
    parser.add_argument('--dlt_R', type=int, default=4, help='DLT: number of shared lag-basis kernels')
    parser.add_argument('--dlt_dyn_dim', type=int, default=16, help='DLT: summary dimension of the dynamic coefficients')
    parser.add_argument('--dlt_ccf_len', type=int, default=8000,
                        help='DLT: number of (most recent) training steps used for neighbour selection')
    parser.add_argument('--dlt_impl', type=str, default='kernel', choices=['kernel', 'basis', 'dense'],
                        help='DLT: computation path for dlt/dltdyn (identical results; kernel = less memory, '
                             'dense = cheapest when k is close to C - 1)')
    parser.add_argument('--dlt_neighbours', type=str, default='ccf', choices=['ccf', 'random', 'all'],
                        help='DLT: neighbour selection (ccf top-k, k random sources, all other variables)')
    parser.add_argument('--pems_split', type=str, default='tslib', choices=['tslib', 'itransformer'],
                        help='PEMS: validation / test windows take their look-back from the preceding split (tslib) or '
                             'start inside their split (itransformer, as the iTransformer / S-Mamba loaders)')
    parser.add_argument('--tide_checkpoint', action='store_true',
                        help='TiDE: recompute each variable\'s forward pass in the backward pass (same results, '
                             'far less memory on data with many variables)')
    parser.add_argument('--drop_last', action='store_true',
                        help='drop the last incomplete training / validation batch (official iTransformer, S-Mamba and '
                             'RLinear loaders)')
    parser.add_argument('--test_batch_size', type=int, default=0,
                        help='batch size of the test (0 = --batch_size); the iTransformer / S-Mamba code tests with 1')
    parser.add_argument('--no_time_marks', action='store_true',
                        help='give the model no time-mark features (x_mark = None), as the iTransformer and S-Mamba '
                             'code does on PEMS and Solar')
    parser.add_argument('--plug_base', type=str, default='DLinear',
                        help='DLTPlug: backbone from models/ that the DLT branch is added to')
    parser.add_argument('--dlt_hp_trial', type=int, default=-1, help='bookkeeping: index of a search trial')
    parser.add_argument('--dlt_lr', type=float, default=0.0,
                        help='DLTPlug: learning rate of the DLT branch (0 = same as --learning_rate)')
    parser.add_argument('--lagts_lambda', type=float, default=0.1, help='LagTS: weight of the lag relation loss')
    parser.add_argument('--lagts_rel_dim', type=int, default=8, help="LagTS: relation embedding size D'")
    parser.add_argument('--vali_inverse', action='store_true', default=False,
                        help='validation loss = MAE on the original scale (TimeMixer protocol for PEMS)')
    parser.add_argument('--dlt_topk', type=int, default=5,
                        help='DLT: lags per pair for the MillGNN-style variants topk / mill / milldyn')
    parser.add_argument('--dlt_gate', type=str, default='none', choices=['none', 'moe', 'static'],
                        help='DLT: combination of the two branches (moe = MoVE-style softmax gate, '
                             'static = per-variable sigmoid gate)')
    parser.add_argument('--dlt_gate_init', type=float, default=-3.0,
                        help='DLT: initial logit of the static gate (sigmoid(-3) = 0.047)')
    parser.add_argument('--dlt_fusion', type=str, default='output', choices=['output', 'refine', 'input'],
                        help='DLTPlug: how the branch enters (output = backbone + W_c z on the look-back window; '
                             'refine = distributed-lag correction over the observed past and the backbone forecast; '
                             'input = neighbour signal added to the backbone input); see models/DLTPlug.py')
    parser.add_argument('--dlt_train', type=str, default='joint', choices=['joint', 'decoupled'],
                        help='DLTPlug: joint = one loss on the sum; decoupled = the backbone trains on its own loss, '
                             'the branch on the backbone\'s error (output / refine)')
    parser.add_argument('--dlt_select', type=str, default='none', choices=['none', 'variable'],
                        help='DLTPlug: variable = keep the correction only for the variables whose validation MSE '
                             'it lowers (decided after training, on validation data)')

    # reproducibility / bookkeeping (all default to the upstream behaviour)
    parser.add_argument('--fix_seed', type=int, default=2021, help='seed of python / numpy / torch')
    parser.add_argument('--no_epoch_test', action='store_true', default=False,
                        help='skip the per-epoch test loss (not used for model selection)')
    parser.add_argument('--no_save_preds', action='store_true', default=False,
                        help='do not save pred.npy / true.npy and the test plots')
    parser.add_argument('--result_jsonl', type=str, default='',
                        help='append one JSON line per run (config, metrics, epochs, time) to this file')
    return parser


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = get_parser()
    args = parser.parse_args()
    if args.fix_seed != fix_seed:
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            # Override setting for specific model to ensure proper checkpoint naming and logging
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'

            if args.model in ('DLT', 'DLTPlug'):  # keep checkpoints / results of different DLT variants apart
                setting += dlt_suffix(args)
            if args.model == 'LagTS':
                setting += '_lam{:g}_rd{}'.format(args.lagts_lambda, args.lagts_rel_dim)
            if args.fix_seed != fix_seed:
                setting += '_seed{}'.format(args.fix_seed)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        # Override setting for specific model to ensure proper checkpoint naming and logging
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'

        if args.model in ('DLT', 'DLTPlug'):  # must match the training-time setting name
            setting += dlt_suffix(args)
        if args.model == 'LagTS':
            setting += '_lam{:g}_rd{}'.format(args.lagts_lambda, args.lagts_rel_dim)
        if args.fix_seed != fix_seed:
            setting += '_seed{}'.format(args.fix_seed)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
