import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import UEAloader

class DataPipeline:
    def __init__(self, args):
        self.args = args
        self.data_dict = {
            'ETTh1': self._load_ett_hour,
            'ETTh2': self._load_ett_hour,
            'ETTm1': self._load_ett_minute,
            'ETTm2': self._load_ett_minute,
            'custom': self._load_custom,
            'm4': self._load_m4,
            'UEA': self._load_uea
        }

    def load_data(self, flag='train'):
        assert flag in ['train', 'val', 'test'], "Flag must be either train, val, or test"
        if self.args.data in self.data_dict:
            data_set = self.data_dict[self.args.data](flag)
        else:
            raise ValueError(f"Unknown dataset: {self.args.data}")
        
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=(flag == 'train'),
            num_workers=self.args.num_workers,
            drop_last=False
        )
        
        return data_loader, getattr(data_set, 'scaler', None)

    def _load_ett_hour(self, flag):
        from data_provider.data_loader import Dataset_ETT_hour
        return Dataset_ETT_hour(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            timeenc=self.args.timeenc,
            freq=self.args.freq
        )

    def _load_ett_minute(self, flag):
        from data_provider.data_loader import Dataset_ETT_minute
        return Dataset_ETT_minute(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            timeenc=self.args.timeenc,
            freq=self.args.freq
        )

    def _load_custom(self, flag):
        from data_provider.data_loader import Dataset_Custom
        return Dataset_Custom(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            timeenc=self.args.timeenc,
            freq=self.args.freq
        )

    def _load_m4(self, flag):
        from data_provider.data_loader import Dataset_M4
        return Dataset_M4(
            root_path=self.args.root_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            scale=self.args.scale,
            timeenc=self.args.timeenc,
            freq=self.args.freq,
            seasonal_patterns=self.args.seasonal_patterns
        )

    def _load_uea(self, flag):
        return UEAloader(
            self.args,
            self.args.root_path,
            flag=flag
        )

def load_dataset(args, flag='train'):
    """
    Wrapper function to load different types of datasets for TimesNet using a pipeline approach.
    
    Args:
    - args: argparse.Namespace object containing various parameters
    - flag: str, 'train', 'val', or 'test'
    
    Returns:
    - data_loader: DataLoader object
    - scaler: StandardScaler object (if applicable)
    """
    pipeline = DataPipeline(args)
    return pipeline.load_data(flag)