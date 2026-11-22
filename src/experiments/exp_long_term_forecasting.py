import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.data_provider.data_factory import data_provider
from src.experiments.exp_basic import Exp_Basic
from src.utils.checkpoint import checkpoint_path, load_checkpoint, log_checkpoint, next_seq, cleanup_checkpoint
from src.utils.metrics import StreamMetrics, mape_clip5, mape_masked, metric
from src.utils.result_io import write_test_result
from src.utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.train_info = {}
        self.test_nonfinite = None
        self.dlt_kept = None
        self._last_checkpoint_seq = None

    @property
    def _model_ref(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("model_params: total={}, trainable={}".format(total, trainable))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        model = self._model_ref
        dlt_lr = getattr(self.args, "dlt_lr", 0.0)
        if dlt_lr > 0 and hasattr(model, "dlt"):
            # the DLT branch of a plug-in gets its own learning rate; the schedule scales both groups alike
            dlt_ids = {id(p) for p in model.dlt.parameters()}
            groups = [
                {"params": [p for p in model.parameters() if id(p) not in dlt_ids], "lr_scale": 1.0},
                {"params": list(model.dlt.parameters()), "lr_scale": dlt_lr / self.args.learning_rate},
            ]
            for g in groups:
                g["lr"] = self.args.learning_rate * g["lr_scale"]
            return optim.Adam(groups)
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if getattr(self.args, "loss", "MSE").upper() == "MAE":
            return nn.L1Loss()
        return nn.MSELoss()

    def _prepare_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        if getattr(self.args, "no_time_marks", False):
            batch_x_mark = batch_y_mark = None
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp

    def vali(self, vali_data, vali_loader, criterion, part=None):
        """Validation loss; part="backbone": of the backbone part of DLTPlug (--dlt_train decoupled, stage 1)."""
        total_loss = []
        net = self._model_ref
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if getattr(self.args, "no_time_marks", False):
                    batch_x_mark = batch_y_mark = None
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if part == "backbone":
                    outputs = net.parts(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                elif self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach()
                true = batch_y.detach()
                if getattr(self.args, "vali_inverse", False) and vali_data.scale:
                    shape = pred.shape
                    p_np = vali_data.inverse_transform(pred.cpu().numpy().reshape(-1, shape[-1])).reshape(shape)
                    t_np = vali_data.inverse_transform(true.cpu().numpy().reshape(-1, shape[-1])).reshape(shape)
                    total_loss.append(np.mean(np.abs(p_np - t_np)))
                    continue
                total_loss.append(criterion(pred, true).item())
        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        net = self._model_ref
        if hasattr(net, "prepare"):
            net.prepare(train_data)
        os.makedirs(os.path.dirname(checkpoint_path(self.args, 1)), exist_ok=True)
        seq = next_seq(self.args)
        best_model_path = checkpoint_path(self.args, seq)
        if getattr(self.args, "checkpoint", 0) > 0:
            load_checkpoint(self.args, self.model, self.args.checkpoint, self.device)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        no_epoch_test = getattr(self.args, "no_epoch_test", False)
        extra_loss = getattr(net, "extra_loss", None)
        # decoupled plug-in: stage 1 stops on the backbone part's validation loss, stage 2 trains the branch on the frozen backbone
        two_stage = getattr(net, "use_dlt", False) and getattr(net, "training_mode", "joint") == "decoupled"
        self.train_info = {"epochs": 0, "best_epoch": 0, "best_vali": None, "train_secs": 0.0, "epoch_secs": []}
        train_start = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._prepare_batch(*batch)
                f_dim = -1 if self.args.features == "MS" else 0
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                        if extra_loss is not None:
                            loss = loss + extra_loss(batch_x, outputs, batch_y)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
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
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
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
            self.train_info["epoch_secs"].append(round(time.time() - epoch_time, 2))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, part="backbone" if two_stage else None)
            test_loss = float("nan") if no_epoch_test else self.vali(test_data, test_loader, criterion)
            self.train_info["epochs"] = epoch + 1
            if self.train_info["best_vali"] is None or vali_loss < self.train_info["best_vali"]:
                self.train_info["best_vali"] = float(vali_loss)
                self.train_info["best_epoch"] = epoch + 1
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, "skipped (--no_epoch_test)" if no_epoch_test else "{:.7f}".format(test_loss)
                )
            )
            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        if two_stage:
            self._train_branch(net, best_model_path, train_loader, vali_data, vali_loader, criterion)
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.train_info["train_secs"] = round(time.time() - train_start, 2)
        log_checkpoint(self.args, seq)
        self._last_checkpoint_seq = seq
        return self.model

    def _train_branch(self, net, path, train_loader, vali_data, vali_loader, criterion):
        """Decoupled plug-in, stage 2: the backbone is frozen (eval mode); the branch trains on its error with its own early stopping."""
        for q in net.base.parameters():
            q.requires_grad_(False)
        lr = getattr(self.args, "dlt_lr", 0.0) or self.args.learning_rate
        opt = optim.Adam([{"params": [q for q in net.dlt.parameters()], "lr": lr, "lr_scale": lr / self.args.learning_rate}])
        es = EarlyStopping(patience=self.args.patience, verbose=True)
        start = self.vali(vali_data, vali_loader, criterion)
        es(start, self.model, path)
        self.train_info.update(
            {"stage1_epochs": self.train_info["epochs"], "branch_epochs": 0, "branch_best_epoch": 0, "branch_start_vali": float(start), "branch_best_vali": float(start)}
        )
        for epoch in range(self.args.train_epochs):
            self.model.train()
            net.base.eval()
            losses = []
            for batch in train_loader:
                opt.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._prepare_batch(*batch)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                loss = criterion(outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:])
                loss.backward()
                opt.step()
                losses.append(loss.item())
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            self.train_info["branch_epochs"] = epoch + 1
            if vali_loss < self.train_info["branch_best_vali"]:
                self.train_info["branch_best_vali"] = float(vali_loss)
                self.train_info["branch_best_epoch"] = epoch + 1
            print("Branch epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(epoch + 1, np.average(losses), vali_loss))
            es(vali_loss, self.model, path)
            if es.early_stop:
                print("Early stopping (branch)")
                break
            adjust_learning_rate(opt, epoch + 1, self.args)
        for q in net.base.parameters():
            q.requires_grad_(True)

    def _variable_mask(self, net):
        """--dlt_select variable: keep the correction only for the variables whose validation MSE it lowers: [C] bool."""
        vali_data, vali_loader = self._get_data(flag="val")
        se_b = se_f = None
        net.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._prepare_batch(*batch)
                y_b, c = net.parts(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                true = batch_y[:, -self.args.pred_len:, :]
                b = ((y_b - true) ** 2).sum((0, 1)).double()
                f = ((y_b + c - true) ** 2).sum((0, 1)).double()
                se_b, se_f = (b, f) if se_b is None else (se_b + b, se_f + f)
        return (se_f < se_b).cpu()

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            load_checkpoint(self.args, self.model, self.args.checkpoint, self.device)
        save_tmp = bool(getattr(self.args, "save_tmp", 1))
        preds, trues = [], []
        stream = None if save_tmp else StreamMetrics()
        folder_path = self._get_test_results_path(setting)
        if save_tmp:
            os.makedirs(folder_path, exist_ok=True)
        net = self._model_ref
        plug = getattr(net, "use_dlt", False) and getattr(net, "fusion", "input") != "input"
        self.dlt_kept = None
        if plug and getattr(self.args, "dlt_select", "none") == "variable":
            net.select_mask = None
            net.select_mask = self._variable_mask(net).to(self.device)
            self.dlt_kept = float(net.select_mask.float().mean())
            print("--dlt_select variable: correction kept for {} of {} variables (validation MSE)".format(int(net.select_mask.sum()), net.select_mask.numel()))
        part_streams = (StreamMetrics(), StreamMetrics()) if plug and stream is not None else None
        f_dim = -1 if self.args.features == "MS" else 0

        def to_eval(o, y_shape):
            o = o[:, -self.args.pred_len:, :].detach().cpu().numpy()
            if test_data.scale and self.args.inverse:
                if o.shape[-1] != y_shape[-1]:
                    o = np.tile(o, [1, 1, int(y_shape[-1] / o.shape[-1])])
                o = test_data.inverse_transform(o.reshape(y_shape[0] * y_shape[1], -1)).reshape(y_shape)
            return o[:, :, f_dim:]

        infer_total, infer_iters = 0.0, 0
        warmup = min(5, max(0, len(test_loader) - 1))
        sync = self.device.type == "cuda"
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._prepare_batch(*batch)
                if sync:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                parts = None
                if part_streams is not None:
                    y_b, c = net.parts(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    m = net.select_mask.to(c.dtype) if net.select_mask is not None else 1.0
                    outputs, parts = y_b + m * c, (y_b, y_b + c)
                elif self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if sync:
                    torch.cuda.synchronize()
                if i >= warmup:
                    infer_total += time.perf_counter() - t0
                    infer_iters += 1
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                if parts is not None:
                    y_shape = tuple(batch_y.shape)
                    true_eval = batch_y.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        true_eval = test_data.inverse_transform(true_eval.reshape(y_shape[0] * y_shape[1], -1)).reshape(y_shape)
                    true_eval = true_eval[:, :, f_dim:]
                    for ps, o in zip(part_streams, parts):
                        ps.update(to_eval(o, y_shape), true_eval)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                pred = outputs[:, :, f_dim:]
                true = batch_y[:, :, f_dim:]
                if stream is not None:
                    stream.update(pred, true)
                    continue
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = inp.shape
                        inp = test_data.inverse_transform(inp.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))
        infer_ms = 1e3 * infer_total / infer_iters if infer_iters else None
        if infer_iters:
            print("inference_speed: {:.6f}s/iter".format(infer_total / infer_iters))
        if stream is None:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print("test shape:", preds.shape, trues.shape)
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            m_clip5, m_masked = mape_clip5(preds, trues), mape_masked(preds, trues)
        else:
            print("test values (streamed metrics):", stream.n)
            mae, mse, rmse, mape, mspe, m_clip5, m_masked = stream.result()
            self.test_nonfinite = stream.nonfinite
            if stream.nonfinite:
                print("WARNING: {} of {} forecast values are NaN or inf (python scripts/utils/diag_nonfinite.py {} finds where)".format(stream.nonfinite, stream.n, self.args.des))
        print("mse:{}, mae:{}".format(mse, mae))
        if save_tmp:
            folder = self._get_results_path(setting)
            os.makedirs(folder, exist_ok=True)
            np.save(os.path.join(folder, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
            np.save(os.path.join(folder, "pred.npy"), preds)
            np.save(os.path.join(folder, "true.npy"), trues)
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape) if np.isfinite(mape) else None,
            "mape_clip5": float(m_clip5),
            "mape_masked": float(m_masked),
        }
        if part_streams is not None:
            (mae_b, mse_b, *_), (mae_f, mse_f, *_) = part_streams[0].result(), part_streams[1].result()
            metrics.update({"mse_backbone": float(mse_b), "mae_backbone": float(mae_b), "mse_full": float(mse_f), "mae_full": float(mae_f)})
        if self.dlt_kept is not None:
            metrics["dlt_kept"] = self.dlt_kept
        if self.test_nonfinite is not None:
            metrics["test_nonfinite"] = self.test_nonfinite
        info = dict(self.train_info)
        info.update(
            {
                "n_params": int(sum(p.numel() for p in self.model.parameters())),
                "infer_ms": round(infer_ms, 3) if infer_ms is not None else None,
                "device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else self.device.type,
                "torch": torch.__version__,
            }
        )
        if hasattr(net, "mamba_impl") or hasattr(getattr(net, "base", None), "mamba_impl"):
            info["mamba_impl"] = getattr(getattr(net, "base", net), "mamba_impl", None)
        seq = self._last_checkpoint_seq if self._last_checkpoint_seq is not None else getattr(self.args, "checkpoint", 0)
        write_test_result(self.args, metrics, info, checkpoint=seq)
        if not int(getattr(self.args, "keep_checkpoint", 1)) and self._last_checkpoint_seq is not None:
            cleanup_checkpoint(self.args, self._last_checkpoint_seq)
        return metrics
