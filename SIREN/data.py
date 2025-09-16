import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from utils import get_mgrid_2d


class Soundfield(Dataset):
    def __init__(self, rir, num_x, time_len):
        super().__init__()
        self.rir = torch.Tensor(rir).to(torch.device("cuda"))
        self.coords = get_mgrid_2d(num_x, time_len).cuda()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.rir
