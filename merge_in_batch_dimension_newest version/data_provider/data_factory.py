import torch

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader, ConcatDataset

import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    # 'merge' : Dataset_Custom_Merge,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}

#MERGE 的所有相关操作都在my_collate这个函数里面，然后在后面data_provider的时候, 如果需要split training, 那就忽略这个my_collate函数。
#相反,则需要在DataLoader里面声明"collate_fn=my_collate".

def my_collate(batch):
    seqx, seqy,seqxmark, seqymark = zip(*batch)

    x_tensor = [torch.tensor(x) for x in seqx]
    # print(type(x_tensor))
    # print(type(x_tensor[0]))
    # print(x_tensor[0].size())
    # exit()
    x_processed = torch.cat(x_tensor,dim = 1)
    x_processed = x_processed.permute(1,0)
    x_processed = x_processed.unsqueeze(-1)

    y_tensor = [torch.tensor(y) for y in seqy]
    y_processed = torch.cat(y_tensor, dim=1)
    y_processed = y_processed.permute(1, 0)
    y_processed = y_processed.unsqueeze(-1)


    seqxmark_tensor = [torch.tensor(x) for x in seqxmark]
    seqxmark_tensor_combine = []

    for i in range(len(x_tensor)):
        seqxmark_tensor_combine.append(torch.stack([seqxmark_tensor[i] for _ in range(x_tensor[i].size()[1])],dim = 0))

    seqymark_tensor = [torch.tensor(y) for y in seqymark]

    seqymark_tensor_combine = []

    for i in range(len(y_tensor)):
        seqymark_tensor_combine.append(torch.stack([seqymark_tensor[i] for _ in range(y_tensor[i].size()[1])], dim=0))

    seqx_mark_tensor = torch.cat(seqxmark_tensor_combine,dim = 0)
    seqy_mark_tensor = torch.cat(seqymark_tensor_combine,dim = 0)

    return x_processed,y_processed,seqx_mark_tensor,seqy_mark_tensor

# the data paths of the train is different from that of the test/validation. so I devided them into to cases
def data_provider(args, flag, tdatapath = None, tdata = None, datapath = None, data = None):
    if (flag == 'test' or flag == 'val'):
        Data = data_dict[data]
        timeenc = 0 if args.embed != 'timeF' else 1

        shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

        if args.task_name == 'anomaly_detection':
            drop_last = False
            data_set = Data(
                args = args,
                root_path=args.root_path,
                win_size=args.seq_len,
                flag=flag,
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
            return data_set, data_loader
        elif args.task_name == 'classification':
            drop_last = False
            data_set = Data(
                args = args,
                root_path=args.root_path,
                flag=flag,
            )

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )
            return data_set, data_loader
        else:
            if args.data == 'm4':
                drop_last = False
            data_set = Data(
                args = args,
                root_path=args.root_path,
                data_path= datapath,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                collate_fn=my_collate,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
            return data_set, data_loader

    else:
        Filepaths = tdatapath.split(',')
        Datass = tdata.split(',')
        Datas=[]
        for i in Datass:
            Datas.append(data_dict[i])
        timeenc = 0 if args.embed != 'timeF' else 1

        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

        if args.task_name == 'anomaly_detection':
            drop_last = False
            Datasets = []
            for Data in Datas:
                data_set = Data(
                    args=args,
                    root_path=args.root_path,
                    win_size=args.seq_len,
                    flag=flag,
                )
                Datasets.append(data_set)
            print(flag, len(Datasets[0]) * len(Datasets))
            Dataset = ConcatDataset(Datasets)
            data_loader = DataLoader(
                Dataset,
                batch_size=batch_size ,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
            return Dataset, data_loader

        elif args.task_name == 'classification':
            drop_last = False
            Datasets = []
            for Data in Datas:
                data_set = Data(
                    args=args,
                    root_path=args.root_path,
                    flag=flag,
                )
                Datasets.append(data_set)
            print(flag, len(Datasets[0]) * len(Datasets))
            Dataset = ConcatDataset(Datasets)
            data_loader = DataLoader(
                Dataset,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
            return Dataset, data_loader
        else:
            if args.data == 'm4':
                drop_last = False
            Datasets = []
            length = len(Datas)
            lengsum = 0
            for i in range(length):
                Data = Datas[i]
                Path = Filepaths[i]
                data_set = Data(
                    args=args,
                    root_path=args.root_path,
                    data_path=Path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    timeenc=timeenc,
                    freq=freq,
                    seasonal_patterns=args.seasonal_patterns
                )
                Datasets.append(data_set)
                lengsum += len(Datasets[i])
            print(flag, lengsum)

            Dataset = ConcatDataset(Datasets)
            data_loader = DataLoader(
                Dataset,
                batch_size=batch_size,
                collate_fn= my_collate,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
            return Dataset, data_loader




# def data_provider(args, flag, tdatapath = None, tdata = None, datapath = None, data = None):
#     if (flag == 'test' or flag == 'val'):
#         Data = data_dict[data]
#         timeenc = 0 if args.embed != 'timeF' else 1
#
#         shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
#         drop_last = False
#         batch_size = args.batch_size
#         freq = args.freq
#
#         if args.task_name == 'anomaly_detection':
#             drop_last = False
#             data_set = Data(
#                 args = args,
#                 root_path=args.root_path,
#                 win_size=args.seq_len,
#                 flag=flag,
#             )
#             print(flag, len(data_set))
#             data_loader = DataLoader(
#                 data_set,
#                 batch_size=batch_size,
#                 shuffle=shuffle_flag,
#                 num_workers=args.num_workers,
#                 drop_last=drop_last)
#             return data_set, data_loader
#         elif args.task_name == 'classification':
#             drop_last = False
#             data_set = Data(
#                 args = args,
#                 root_path=args.root_path,
#                 flag=flag,
#             )
#
#             data_loader = DataLoader(
#                 data_set,
#                 batch_size=batch_size,
#                 shuffle=shuffle_flag,
#                 num_workers=args.num_workers,
#                 drop_last=drop_last,
#                 collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
#             )
#             return data_set, data_loader
#         else:
#             if args.data == 'm4':
#                 drop_last = False
#             data_set = Data(
#                 args = args,
#                 root_path=args.root_path,
#                 data_path= datapath,
#                 flag=flag,
#                 size=[args.seq_len, args.label_len, args.pred_len],
#                 features=args.features,
#                 target=args.target,
#                 timeenc=timeenc,
#                 freq=freq,
#                 seasonal_patterns=args.seasonal_patterns
#             )
#             print(flag, len(data_set))
#             data_loader = DataLoader(
#                 data_set,
#                 batch_size=batch_size,
#                 shuffle=shuffle_flag,
#                 num_workers=args.num_workers,
#                 drop_last=drop_last)
#             return data_set, data_loader
#
#     else:
#         Filepaths = tdatapath.split(',')
#         Data = data_dict[tdata]
#         timeenc = 0 if args.embed != 'timeF' else 1
#         shuffle_flag = True
#         drop_last = False
#         batch_size = args.batch_size
#         freq = args.freq
#         if args.task_name == 'anomaly_detection':
#             drop_last = False
#             data_set = Data(
#                 args=args,
#                 root_path=args.root_path,
#                 win_size=args.seq_len,
#                 flag=flag,
#             )
#             print(flag, len(data_set))
#             data_loader = DataLoader(
#                 data_set,
#                 batch_size=batch_size ,
#                 shuffle=shuffle_flag,
#                 num_workers=args.num_workers,
#                 drop_last=drop_last)
#             return data_set, data_loader
#         elif args.task_name == 'classification':
#             drop_last = False
#             data_set = Data(
#                 args=args,
#                 root_path=args.root_path,
#                 win_size=args.seq_len,
#                 flag=flag,
#             )
#             print(flag, len(data_set))
#             data_loader = DataLoader(
#                 data_set,
#                 batch_size=batch_size,
#                 shuffle=shuffle_flag,
#                 num_workers=args.num_workers,
#                 drop_last=drop_last)
#             return data_set, data_loader
#         else:
#             if args.data == 'm4':
#                 drop_last = False
#             drop_last = False
#             data_set = Data(
#                     args=args,
#                     root_path=args.root_path,
#                     data_path=Filepaths,
#                     flag=flag,
#                     size=[args.seq_len, args.label_len, args.pred_len],
#                     features=args.features,
#                     target=args.target,
#                     timeenc=timeenc,
#                     freq=freq,
#                     seasonal_patterns=args.seasonal_patterns
#                 )
#             print(flag, len(data_set))
#             data_loader = DataLoader(
#                 data_set,
#                 batch_size=batch_size,
#                 collate_fn=my_collate,
#                 shuffle=shuffle_flag,
#                 num_workers=args.num_workers,
#                 drop_last=drop_last)
#             return data_set, data_loader





