from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Pred,
    Dataset_Gefcom,
    Dataset_Gefcom2017_Reg,
    Dataset_Gefcom2012_Reg,
    # Dataset_Gefcom_Reg_Lag,
)

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'gefcom': Dataset_Gefcom,
    'gefcom2017_reg': Dataset_Gefcom2017_Reg,
    'gefcom2012_reg': Dataset_Gefcom2012_Reg,
# 'gefcom_reg_lag': Dataset_Gefcom_Reg_Lag,
}


def data_provider(args, split, real_time=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if split in ['test', 'valid']:
        shuffle_flag = False
        drop_last = False  # why drop last?
        batch_size = args.batch_size
        freq = args.freq
    elif split == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.detail_freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True & ~real_time
        drop_last = False  #
        batch_size = args.batch_size
        freq = args.freq

    # TODO(rzhao): combine the branches.
    if "gefcom" in args.data:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=split,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            lag=args.lag,
            scaler=args.scaler,
            real_time=real_time,
            zone=args.zone,
            attack_rate=args.attack_rate,
            attack_form=args.attack_form,
            attack_increase=args.attack_increase,
            dist_param_a=args.dist_param_a,
            dist_param_b=args.dist_param_b,
            cols=args.cols,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=split,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            lag=args.lag,
            real_time=real_time,
            scaler=args.scaler,
        )
    # print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
