from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import double_tuple
import os.path as osp
from PIL import Image
import nibabel as nib
import numpy as np
import torch
import os

class Mosaic_Dataset(Dataset):
    def __init__(self, root_dir, final_resolution=7, input_resolution=224, num_modals=4, is_train=True):
        super().__init__()
        self.final_resolution = final_resolution
        self.input_resolution = input_resolution
        self.num_modals = num_modals
        self.root_dir = root_dir
        if is_train:
            self.dataset = os.listdir(osp.join(root_dir, 'train'))
        else:
            self.dataset = os.listdir(osp.join(root_dir, 'val'))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        mask = torch.randint(0, self.num_modals, double_tuple(self.final_resolution)).float()
        nifti_img = nib.load(self.dataset[index])
        total_img = nifti_img.get_fdata()



if __name__ == '__main__':
    a = torch.randint(0, 4, (1, 1, 7, 7))
    print(F.interpolate(a.float(), scale_factor=32, mode='nearest'))