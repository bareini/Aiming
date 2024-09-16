import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl
import u_loss

# Constants
BASE_SEQ_LEN = 496

class SimDataset(Dataset):
    def __init__(self, dataidx=0, mode='train', save_trends=False, series_len=BASE_SEQ_LEN):
        self.mode = mode
        self.save_trends = save_trends
        self.series_len = series_len

        self._load_data(dataidx)
        self._process_data()

    def _load_data(self, dataidx):
        if self.mode in ['train', 'test']:
            file_path = 'sim_data.pkl'
        elif self.mode == 'val':
            file_path = 'sim_data_val.pkl'
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        with open(file_path, 'rb') as f:
            self.datasets = pkl.load(f)

    def _process_data(self):
        if self.mode in ['train', 'test']:
            prefix = '' if self.mode == 'train' else 'test_'
            split_size = int(self.datasets[f"{prefix}targets"][0].shape[0] / self.series_len)
        else:  # val mode
            prefix = 'val_'
            split_size = 200

        self._split_data(split_size, prefix)
        self._initialize_data_structures()
        self._prepare_data()

    def _split_data(self, split_size, prefix):
        self.X = np.array_split(self.datasets[f"{'data_' if self.mode == 'val' else ''}{prefix}{'s' if self.mode != 'val' else ''}"][0], split_size, axis=0)
        self.y = np.array_split(self.datasets[f"{'targets_' if self.mode == 'val' else ''}{prefix}targets"][0], split_size)
        self.onset1 = np.array_split(self.datasets[f"onset1s_{self.mode}"][0], split_size)
        self.onset2 = np.array_split(self.datasets[f"onset2s_{self.mode}"][0], split_size)
        self.trend_events = np.array_split(self.datasets[f"trend_events_{self.mode}"][0], split_size)
        self.ids = np.array_split(self.datasets[f"{prefix}ids"][0], split_size)
        self.mean = self.datasets[f"{prefix}{'targets' if self.mode != 'val' else 'ids'}"][0].mean()

    def _initialize_data_structures(self):
        self.data = {}
        self.y_t = {}
        self.onset1_dict, self.onset2_dict, self.trend_events_dict, self.ids_dict = {}, {}, {}, {}
        if self.save_trends:
            self.trends_true = {}
            self.trends_expected = {}

    def _prepare_data(self):
        trend_loss = u_loss.TrendLoss()

        for i, (x, y_val, onset1_val, onset2_val, trend_events_val, ids_val) in enumerate(zip(
            self.X, self.y, self.onset1, self.onset2, self.trend_events, self.ids
        )):
            self.data[i] = torch.from_numpy(x)
            self.y_t[i] = torch.from_numpy(y_val)
            self.onset1_dict[i] = onset1_val.astype(int)
            self.onset2_dict[i] = onset2_val.astype(int)
            self.trend_events_dict[i] = trend_events_val
            self.ids_dict[i] = ids_val

            if self.save_trends:
                self._calculate_trends(i, trend_loss)

    def _calculate_trends(self, i, trend_loss):
        self.trends_true[i] = torch.squeeze(trend_loss.calc_trend(
            torch.unsqueeze(self.y_t[i], dim=0), k=3))
        self.trends_expected[i] = torch.squeeze(trend_loss.calc_trend(
            torch.unsqueeze(self.y_t[i], dim=0), k=2))[:-1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        item = (
            self.data[index],
            self.y_t[index],
            self.onset1_dict[index],
            self.onset2_dict[index],
            self.trend_events_dict[index],
            self.ids_dict[index],
        )

        if self.save_trends:
            item += (self.trends_true[index], self.trends_expected[index])
        else:
            item += (None, None)

        return item
