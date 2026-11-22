from torch.utils.data import DataLoader

from src.data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_PEMS, Dataset_Solar

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "PEMS": Dataset_PEMS,
    "Solar": Dataset_Solar,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    is_test = flag == "test"
    shuffle_flag = not is_test
    drop_last = bool(getattr(args, "drop_last", False)) and not is_test
    batch_size = args.batch_size
    if is_test and getattr(args, "test_batch_size", 0):
        batch_size = args.test_batch_size
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last)
    return data_set, data_loader
