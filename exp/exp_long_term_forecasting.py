from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import json
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


def mape_timemixer(pred, true):
    """MAPE as in TimeMixer's PEMS protocol: ratios > 5 (true values near 0) count as 0. A perfect
    prediction of a zero value (0 / 0, e.g. a dead sensor that RevIN models reproduce exactly) counts as 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.abs((pred - true) / true)
    r = np.nan_to_num(r, nan=0.0, posinf=np.inf)
    return float(np.mean(np.where(r > 5, 0, r)))


MASK_EPS = 1e-3   # |true| <= MASK_EPS counts as a missing value: zeros come back as ~1e-5 after float32 scaling


def mape_masked(pred, true):
    """MAPE over the entries with |true| > MASK_EPS (the masked MAPE of the traffic-forecasting literature;
    PEMS flows are counts >= 1, but a zero is ~1e-5 rather than 0 after the float32 scale / inverse round trip)."""
    m = np.abs(true) > MASK_EPS
    return float(np.mean(np.abs((pred[m] - true[m]) / true[m]))) if m.any() else float('nan')


class StreamMetrics:
    """Running sums of the test metrics, so that the test split need not be held in memory (--no_save_preds;
    the full prediction arrays of Electricity / Traffic at horizon 720 take tens of GB). Same definitions as
    utils/metrics.py, mape_timemixer and mape_masked; sums in float64, so the values agree with the array
    versions up to float32 rounding (~1e-7 relative)."""

    def __init__(self):
        self.n = self.n_masked = self.nonfinite = 0
        self.s = dict.fromkeys(('ae', 'se', 'ape', 'spe', 'ape_clip5', 'ape_masked'), 0.0)

    def update(self, pred, true):
        self.nonfinite += int(pred.size - np.isfinite(pred).sum())   # NaN / inf forecasts (reported in the record)
        d = pred.astype(np.float64) - true.astype(np.float64)
        t = true.astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = d / t                                              # x / 0 -> +-inf, 0 / 0 -> nan
        a = np.abs(r)
        self.n += d.size
        self.s['ae'] += np.abs(d).sum()
        self.s['se'] += np.square(d).sum()
        self.s['ape'] += a.sum()                                   # inf / nan propagate as in np.mean
        self.s['spe'] += np.square(r).sum()
        tm = np.nan_to_num(a, nan=0.0, posinf=np.inf)
        self.s['ape_clip5'] += np.where(tm > 5, 0.0, tm).sum()
        m = np.abs(t) > MASK_EPS
        self.n_masked += int(m.sum())
        self.s['ape_masked'] += a[m].sum()

    def result(self):
        n, s = self.n, self.s
        mae, mse = s['ae'] / n, s['se'] / n
        return (mae, mse, float(np.sqrt(mse)), s['ape'] / n, s['spe'] / n, s['ape_clip5'] / n,
                s['ape_masked'] / self.n_masked if self.n_masked else float('nan'))


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        dlt_lr = getattr(self.args, 'dlt_lr', 0.0)
        if dlt_lr > 0 and hasattr(model, 'dlt'):
            # plug-in: the DLT branch gets its own learning rate; the schedule scales both groups alike
            dlt_ids = {id(p) for p in model.dlt.parameters()}
            groups = [{'params': [p for p in model.parameters() if id(p) not in dlt_ids], 'lr_scale': 1.0},
                      {'params': list(model.dlt.parameters()), 'lr_scale': dlt_lr / self.args.learning_rate}]
            for g in groups:
                g['lr'] = self.args.learning_rate * g['lr_scale']
            return optim.Adam(groups)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if getattr(self.args, 'loss', 'MSE').upper() == 'MAE':    # e.g. the TimeMixer protocol on PEMS
            return nn.L1Loss()
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if getattr(self.args, 'no_time_marks', False):     # iTransformer / S-Mamba protocol on PEMS, Solar
                    batch_x_mark = batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                if getattr(self.args, 'vali_inverse', False) and vali_data.scale:
                    shape = pred.shape
                    p_np = vali_data.inverse_transform(pred.cpu().numpy().reshape(-1, shape[-1])).reshape(shape)
                    t_np = vali_data.inverse_transform(true.cpu().numpy().reshape(-1, shape[-1])).reshape(shape)
                    total_loss.append(np.mean(np.abs(p_np - t_np)))
                    continue
                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # models that need statistics of the training split before training
        # (e.g. DLT: neighbour / lag selection); unwrap DataParallel first
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, 'prepare'):
            model.prepare(train_data)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        no_epoch_test = getattr(self.args, 'no_epoch_test', False)
        aux_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        extra_loss = getattr(aux_model, 'extra_loss', None)      # e.g. LagTS: lambda * lag relation loss
        self.train_info = {'epochs': 0, 'best_epoch': 0, 'best_vali': None, 'train_secs': 0.0, 'epoch_secs': []}
        train_start = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if getattr(self.args, 'no_time_marks', False):     # iTransformer / S-Mamba protocol on PEMS, Solar
                    batch_x_mark = batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                        if extra_loss is not None:
                            loss = loss + extra_loss(batch_x, outputs, batch_y)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    if extra_loss is not None:
                        loss = loss + extra_loss(batch_x, outputs, batch_y)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            self.train_info['epoch_secs'].append(round(time.time() - epoch_time, 2))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = float('nan') if no_epoch_test else self.vali(test_data, test_loader, criterion)
            self.train_info['epochs'] = epoch + 1
            if self.train_info['best_vali'] is None or vali_loss < self.train_info['best_vali']:
                self.train_info['best_vali'] = float(vali_loss)
                self.train_info['best_epoch'] = epoch + 1

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4}".format(
                epoch + 1, train_steps, train_loss, vali_loss,
                'skipped (--no_epoch_test)' if no_epoch_test else '{:.7f}'.format(test_loss)))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.train_info['train_secs'] = round(time.time() - train_start, 2)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # without saved predictions (and without DTW) the metrics are accumulated batch by batch
        stream = StreamMetrics() if getattr(self.args, 'no_save_preds', False) and not self.args.use_dtw else None
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if getattr(self.args, 'no_time_marks', False):     # iTransformer / S-Mamba protocol on PEMS, Solar
                    batch_x_mark = batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                if stream is not None:
                    stream.update(pred, true)
                    continue
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0 and not getattr(self.args, 'no_save_preds', False):
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if stream is None:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)
        else:
            print('test values (streamed metrics):', stream.n)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        if stream is None:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            mape_clip5, mape_mask = mape_timemixer(preds, trues), mape_masked(preds, trues)
        else:
            mae, mse, rmse, mape, mspe, mape_clip5, mape_mask = stream.result()
            self.test_nonfinite = stream.nonfinite
            if stream.nonfinite:
                print(f'WARNING: {stream.nonfinite} of {stream.n} forecast values are NaN or inf '
                      f'(python scripts/gpu/diag_nonfinite.py {self.args.des} finds where)')
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if not getattr(self.args, 'no_save_preds', False):
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        if getattr(self.args, 'result_jsonl', ''):
            a = self.args
            rec = {'setting': setting, 'model': a.model, 'model_id': a.model_id, 'des': a.des, 'data': a.data,
                   'data_path': a.data_path, 'seq_len': a.seq_len, 'pred_len': a.pred_len,
                   'seed': getattr(a, 'fix_seed', 2021), 'lr': a.learning_rate, 'batch_size': a.batch_size,
                   'train_epochs': a.train_epochs, 'patience': a.patience,
                   'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse),
                   'mape': float(mape) if np.isfinite(mape) else None,
                   'mape_clip5': float(mape_clip5), 'mape_masked': float(mape_mask),
                   'inverse': bool(a.inverse), 'loss': getattr(a, 'loss', 'MSE'),
                   'vali_inverse': bool(getattr(a, 'vali_inverse', False)),
                   'n_params': int(sum(p.numel() for p in self.model.parameters())),
                   'torch': torch.__version__, 'time_marks': not getattr(a, 'no_time_marks', False),
                   'test_batch_size': getattr(a, 'test_batch_size', 0) or a.batch_size,
                   'device': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else self.device.type}
            if a.data == 'PEMS':
                rec['pems_split'] = getattr(a, 'pems_split', 'tslib')
            if a.model in ('DLT', 'DLTPlug'):
                rec.update({'variant': a.dlt_variant, 'gate': a.dlt_gate, 'k': a.dlt_k, 'L': a.dlt_L,
                            'R': a.dlt_R, 'topk': a.dlt_topk, 'impl': a.dlt_impl,
                            'neighbours': getattr(a, 'dlt_neighbours', 'ccf'), 'dyn_dim': a.dlt_dyn_dim})
                if getattr(a, 'dlt_hp_trial', -1) >= 0:
                    rec['hp_trial'] = a.dlt_hp_trial
            if a.model == 'DLTPlug':
                rec['plug_base'] = a.plug_base
                rec['dlt_lr'] = getattr(a, 'dlt_lr', 0.0)
            if a.model == 'LagTS':
                rec.update({'lagts_lambda': a.lagts_lambda, 'lagts_rel_dim': a.lagts_rel_dim})
            net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            if a.model == 'SMamba' or (a.model == 'DLTPlug' and a.plug_base == 'SMamba'):   # which Mamba block ran
                rec.update({'d_state': a.d_state, 'mamba_impl': getattr(getattr(net, 'base', net), 'mamba_impl', None)})
            if a.model != 'DLT':
                rec.update({'d_model': a.d_model, 'd_ff': a.d_ff, 'e_layers': a.e_layers, 'n_heads': a.n_heads})
                if a.dlt_gate == 'static':
                    rec['gate_init'] = a.dlt_gate_init
            rec.update(getattr(self, 'train_info', {}))
            if getattr(self, 'test_nonfinite', None) is not None:
                rec['test_nonfinite'] = self.test_nonfinite
            with open(a.result_jsonl, 'a') as fj:
                fj.write(json.dumps(rec) + '\n')

        return
