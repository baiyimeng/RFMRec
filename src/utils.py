import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


def setup_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_arguments(args):
    return "\n".join(f"{key}: {value}" for key, value in vars(args).items())


def format_training_time(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.1f}s"
    else:
        return f"{seconds:.2f}s"


class MyDataset(Dataset):
    def __init__(self, id2seq, max_len):
        self.id2seq = id2seq
        self.max_len = max_len

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self.id2seq[index]
        history_seq = seq[:-1]
        history_seq = history_seq[-self.max_len :]

        pad_length = self.max_len - len(history_seq)
        input_ids = [0] * pad_length + history_seq
        target_ids = [0] * pad_length + seq[-len(history_seq) :]

        assert sum([i > 0 for i in input_ids]) == sum([i > 0 for i in target_ids])
        return torch.LongTensor(input_ids), torch.LongTensor(target_ids)


class DataTrain:
    def __init__(self, data_train, max_len, batch_size):
        self.id2seq = data_train
        self.max_len = max_len
        self.batch_size = batch_size

    def get_dataloader(self):
        dataset = MyDataset(self.id2seq, self.max_len)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )


class DataVal:
    def __init__(self, data_train, data_val, max_len, batch_size):
        self.id2seq = [x + y for x, y in zip(data_train, data_val)]
        self.max_len = max_len
        self.batch_size = batch_size

    def get_dataloader(self):
        dataset = MyDataset(self.id2seq, self.max_len)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        return dataloader


class DataTest:
    def __init__(self, data_train, data_val, data_test, max_len, batch_size):
        self.id2seq = [x + y + z for x, y, z in zip(data_train, data_val, data_test)]
        self.max_len = max_len
        self.batch_size = batch_size

    def get_dataloader(self):
        dataset = MyDataset(self.id2seq, self.max_len)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        return dataloader
