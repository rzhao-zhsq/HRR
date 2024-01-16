import argparse
import logging
import os
import random
import shutil
import sys
import warnings
from logging import Logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_provider.data_factory import data_provider
from models import (
    Transformer,
    TransformerDA,
    Informer,
    Autoformer,
    FEDformer,
    VFEDformer,
    MDFEDformer,
    DLinear,
    DConv,
    VAEDConv,
    VAELinear,
    PatchTST,
    PatchTST_resemble,
    TransformerEO,
    TransformerDA,
    TransformerDAEO,
)
from utils.metrics import metric
from utils.misc import load_config, move_to_device
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        # build model and set up logging dir
        self.model = self._build_model()
        args.model_dir = os.path.join(args.exp_out, args.dir_prefix + self.model.get_loginfo())
        self.model_dir = args.model_dir
        self.logger = self._make_logger(args.model_dir)
        self.tb_writer = SummaryWriter(log_dir=os.path.join(args.model_dir, "tensorboard"))

        # training params
        train_cfg = cfg['training']
        self.use_cuda = train_cfg['use_cuda']
        self.device = self._acquire_device()
        self.model = self.model.to(self.device)
        self._log_paramemters(self.model)

        # optimizer and learning rate scheduler
        self.epochs = train_cfg['epochs']
        self.criterion = self._select_criterion()
        # TODO: learning rate scheduler for epochs.
        self.optimizer = build_optimizer(
            config=train_cfg,
            parameters=self.model.parameters()
        )
        self.scheduler, self.scheduler_freq = build_scheduler(
            config=train_cfg,
            scheduler_mode="min",
            optimizer=self.optimizer
        )

    def _acquire_device(self):
        if self.use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu
            ) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            self.logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.logger.info('Use CPU')
        return device

    def _log_paramemters(self, model):
        total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info("Total parameters = {}".format(total_params))
        self.logger.info("Total trainable parameters = {}".format(total_params_trainable))


    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'MDFEDformer': MDFEDformer,
            'VFEDformer': VFEDformer,
            'Autoformer': Autoformer,
            'Informer': Informer,
            'Transformer': Transformer,
            'TransformerDA': TransformerDA,
            'TransformerEO': TransformerEO,
            'TransformerDAEO': TransformerDAEO,
            'PatchTST': PatchTST,
            'PatchTST_resemble': PatchTST_resemble,
            'DLinear': DLinear,
            'DConv': DConv,
            'VAEDConv': VAEDConv,
            'VAELinear': VAELinear,
        }
        model = model_dict[self.args.model].Model(self.args).float()  # float?
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _make_logger(self, model_dir: str, overwrite: bool = True, log_file: str = "train.log", ) -> Logger:
        """
        Create a logger for logging the training process.

        :param overwrite: flag to create the dir.
        :param model_dir: path to logging directory
        :param log_file: path to logging file
        :return: logger object
        """
        # global logger
        if os.path.isdir(model_dir):
            if not overwrite:
                raise FileExistsError("Model directory exists and overwriting is disabled.")
            # delete previous directory to start with empty dir again
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(level=logging.DEBUG)
            fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
            fh.setLevel(level=logging.DEBUG)
            logger.addHandler(fh)
            formatter = logging.Formatter("%(asctime)s %(message)s")
            fh.setFormatter(formatter)
            if sys.platform == "linux":
                sh = logging.StreamHandler()
                sh.setLevel(logging.INFO)
                sh.setFormatter(formatter)
                logging.getLogger("").addHandler(sh)
            logger.info("Logger prepared:  {}/{}!".format(model_dir, log_file))
            return logger

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # def _select_optimizer(self):
    #     model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    #     return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _mse_loss(self, pred, y):

        return F.mse_loss(pred, y)

    def run_batch(self, batch, step) -> dict:
        # batch_x, batch_y, batch_x_mark, batch_y_mark = (
        #     batch.prev_x, batch.target_y, batch.prev_mark, batch.target_mark
        # )
        batch = move_to_device(batch, self.device, torch.float32)
        f_dim = -1 if self.args.features == 'MS' else 0
        target = batch['target_y'][:, -self.args.pred_len:, f_dim:]

        # encoder - decoder
        self.optimizer.zero_grad()
        model_output_dict = self.model.forward(**batch)
        batch_loss = mse_loss = self._mse_loss(model_output_dict['output'][..., f_dim:], target)
        for k in model_output_dict:
            if "_loss" in k:
                batch_loss += model_output_dict[k]
        model_output_dict['mse_loss'] = mse_loss
        model_output_dict['batch_loss'] = batch_loss
        return model_output_dict

    def train_and_validate(self):
        train_data, train_loader = data_provider(self.args, flag='train')
        valid_data, vali_loader = data_provider(self.args, flag='valid', real_time=True)
        test_data, test_loader = data_provider(self.args, flag='test', real_time=True)
        # valid_data, vali_loader = self._get_data(flag='valid')
        # test_data, test_loader = self._get_data(flag='test')
        self.logger.info(
            "Train samples: %d, Dev samples: %d, Test samples: %d",
            len(train_data), len(valid_data), len(test_data)
        )
        self.logger.info("checkpoint save directory: %s", self.model_dir)

        early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=False, logger=self.logger)
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        # for debug
        # dev_loss = self.evaluate(valid_data, vali_loader)
        global_step = 0
        for epoch in range(self.epochs):
            epoch_train_loss = []
            self.model.train()
            for i, batch in enumerate(train_loader):
                global_step += 1
                self.optimizer.zero_grad()
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output_dict = self.run_batch(batch, global_step)
                    loss = model_output_dict['batch_loss']
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    model_output_dict = self.run_batch(batch, global_step)
                    loss = model_output_dict['batch_loss']
                    loss.backward()
                    self.optimizer.step()

                epoch_train_loss.append(loss.item())

                for k, v in model_output_dict.items():
                    if '_loss' in k:
                        self.tb_writer.add_scalar('train/' + k, v, global_step)
                self.tb_writer.add_scalar('train/optim_lr', self.optimizer.param_groups[0]['lr'], global_step)

                if self.scheduler_freq == "step":
                    self.scheduler.step(metrics=loss)
                    # adjust_learning_rate(self.optimizer, epoch + 1, self.args)

                if len(train_loader) // 3 == 0 or global_step % (len(train_loader) // 3) == 0:
                    self.logger.info(
                        "[Training] at Epoch: %03d, Step: %08d | Train Loss: %2.6f",
                        epoch + 1, global_step, loss.item(),
                    )

            # validate at each epoch.
            train_loss = np.average(epoch_train_loss)
            dev_loss = self.evaluate(valid_data, vali_loader)
            test_loss = self.evaluate(test_data, test_loader)
            self.logger.info(
                "[Evaluate] at Epoch: %03d, Step: %08d | "
                "Averaged Running Train Loss: %2.6f, "
                "Averaged Validation Loss: %2.6f, "
                "Averaged Test Loss: %2.6f",
                epoch + 1, global_step, train_loss, dev_loss.item(), test_loss.item()
            )
            # self.logger.info("*" * 40)
            self.tb_writer.add_scalar('Evaluate/train_loss', train_loss, epoch)
            self.tb_writer.add_scalar('Evaluate/dev_loss', dev_loss, epoch)
            self.tb_writer.add_scalar('Evaluate/test_loss', test_loss, epoch)


            early_stopping(dev_loss, self.model, self.model_dir)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            # update learning rate at each epoch.
            if self.scheduler_freq == "epoch":
                self.scheduler.step(metrics=dev_loss)
                # adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        self.test(data=data_provider(self.args, flag='train', real_time=True))
        self.test(data=data_provider(self.args, flag='valid', real_time=True))

    def evaluate(self, valid_data, vali_loader):
        step_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                self.optimizer.zero_grad()
                if self.args.use_amp:

                    with torch.cuda.amp.autocast():
                        model_output_dict = self.run_batch(batch, i)
                    loss = model_output_dict['mse_loss']

                else:
                    model_output_dict = self.run_batch(batch, i)
                    loss = model_output_dict['mse_loss']
                step_loss.append(loss.item())

        avg_loss = np.average(step_loss)
        self.model.train()
        return avg_loss

    def test(self, data: [tuple] = None):
        if data is not None:
            test_data, test_loader = data[0], data[1]
        else:
            test_data, test_loader = data_provider(self.args, flag='test', real_time=True)
        self.logger.info(
            "[{}] Test samples: {}".format(test_data.set_type.title(), len(test_data))
        )
        # self.logger.info("Loading Model for Test..")
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'checkpoint.pth')))

        preds, trues = [], []
        # result save
        folder_path = os.path.join(self.model_dir, "test")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = move_to_device(batch, self.device, torch.float32)

                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch['target_y'][:, -self.args.pred_len:, f_dim:]
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # output_dict = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        model_output_dict = self.model(**batch)
                else:
                    # output_dict = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    model_output_dict = self.model(**batch)

                pred = model_output_dict['output'][..., f_dim:].detach().cpu().numpy()
                true = target.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

        preds, trues = np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
        if test_data.real_time:
            preds, trues = preds.reshape(-1, preds.shape[-1]), trues.reshape(-1, trues.shape[-1])
            overlapped_size = preds.shape[0] - test_data.whole_pred_len()
            ind1 = list(range(self.args.pred_len * (len(test_data) - 1)))
            ind2 = list(range(preds.shape[0]))[overlapped_size - self.args.pred_len:]
            ind = ind1 + ind2
            preds, trues = preds[ind, :], trues[ind, :]

        denormalized_preds = test_data.inverse_transform(preds)
        denormalized_trues = test_data.inverse_transform(trues)
        mae, mse, rmse, mape, mspe, da = metric(denormalized_preds, denormalized_trues)
        self.logger.info('{} Metrics: MSE:{:.4f}, MAE:{:.4f}, RMSE:{:.4f}, MAPE:{:.4f}, DA:{:.4f}'.format(
            test_data.set_type.title(), mse, mae, rmse, mape, da
        ))
        results_file = "{} MSE_{:.4f} MAE_{:.4f} MAPE_{:.4f} MSPE_{:.4f} DA_{:.4f}".format(
            test_data.set_type.title(), mse, mae, mape, mspe, da
        )
        with open(os.path.join(self.model_dir, results_file), "w") as f:
            f.write("")
        np.save(
            folder_path + '/{} metrics.npy'.format(test_data.set_type.title()),
            np.array([mae, mse, rmse, mape, mspe])
        )
        np.save(
            folder_path + '/{} pred.npy'.format(test_data.set_type.title()),
            denormalized_preds
        )
        np.save(
            folder_path + '/{} true.npy'.format(test_data.set_type.title()),
            denormalized_trues
        )

        mae, mse, rmse, mape, mspe, da = metric(preds, trues)
        self.logger.info(
            '[Normalized_Metrics {}]: MSE:{:.4f}, MAE:{:.4f}, RMSE:{:.4f}, MAPE:{:.4f}, DA:{:.4f}'.format(
                test_data.set_type.title(), mse, mae, rmse, mape, da
            )
        )
        results_file = "{} MSE_{:.4f} MAE_{:.4f} MAPE_{:.4f} MSPE_{:.4f} DA_{:.4f} Normalized".format(
            test_data.set_type.title(), mse, mae, mape, mspe, da
        )
        with open(os.path.join(self.model_dir, results_file), "w") as f:
            f.write("")
        np.save(
            folder_path + '/{} Normalized_metrics.npy'.format(test_data.set_type.title()),
            np.array([mae, mse, rmse, mape, mspe])
        )
        # np.save(folder_path + '/Normalized_pred.npy', preds)
        # np.save(folder_path + '/Normalized_true.npy', trues)


    def predict(self, load=False):
        # TODO, unnormlization.
        pred_data, pred_loader = self._get_data(flag='pred')
        self.logger.info("[Predict] Test samples: %d", len(pred_data))
        if load:
            best_model_path = os.path.join(self.model_dir, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch_y[:, -self.args.pred_len:, f_dim:]

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = os.path.join(self.args.model_dir, "predict")
        if not os.path.exists(folder_path):
            os.makedirs

        np.save(folder_path + 'real_prediction.npy', preds)

        return


def main():
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--train', action='store_true', help='train or predict only')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file')
    parser.add_argument('--model', type=str, default='FEDformer',
                        help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--scaler', type=str, default='std', help='data scaler type')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--exp_out', type=str, default='./exp_out/', help='directory of experiment output')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    # model define
    parser.add_argument('--lag', type=int, default=0, help='lag')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--moving_avg', type=int, default=24, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='whether to use distilling in encoder, using this argument means not using distilling')
    parser.add_argument('--individual', action='store_true', help='train or predict only')
    parser.add_argument('--revin', action='store_true', help='revin or not')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fc_dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--subtract_last', action='store_true', help='self supervised learning')
    parser.add_argument('--ssl', action='store_true', help='self supervised learning')
    parser.add_argument('--load_ckpt', action='store_true', help='whether load ckpt from pretrained VAE model.')
    parser.add_argument('--pretrained_model', type=str, help='pretrained model dir of ssl')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='early stopping early_stop_patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    # parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    parser.add_argument('--latent-dim', type=int, default=8, help='latent dimension of gaussian variables')
    parser.add_argument('--sub_space', action='store_true', help='whether to sub_disagreement')
    parser.add_argument('--sub_space_residual', action='store_true', help='whether to sub_disagreement')
    parser.add_argument('--out_space', action='store_true', help='whether to out_disagreement')
    parser.add_argument('--out_space_residual', action='store_true', help='whether to out_disagreement')
    parser.add_argument('--diversity_weight', type=float, default=1.0, help='weight of disagreement loss')
    parser.add_argument('--diversity_metric', type=str, default='bcv', help='diversity metric')


    args = parser.parse_args()
    cfg = load_config(args.config)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # print('Args in experiment:')
    # for k, v in vars(args).items():
    #     print("{}:{}".format(k, v))

    dir_prefix = f'{args.data}_{args.model}_features_{args.features}'
    args.dir_prefix = dir_prefix

    if args.train:
        # args.model_dir = os.path.join(args.exp_out, setting)
        exp: Trainer = Trainer(args, cfg=cfg)  # set experiments
        exp.logger.info('Training: {}'.format(args.model_dir))
        exp.train_and_validate()
        exp.logger.info('Testing: {}'.format(args.model_dir))
        exp.test()
        if args.do_predict:
            exp.logger.info('Predicting: {}'.format(args.model_dir))
            exp.predict(True)

        torch.cuda.empty_cache()
    else:
        # args.model_dir = os.path.join(args.exp_out, setting)
        exp: Trainer = Trainer(args, cfg)  # set experiments
        exp.logger.info('Testing: {}'.format(args.model_dir))
        exp.test()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
