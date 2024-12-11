from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import double_tuple
import os.path as osp
from PIL import Image
import nibabel as nib
import numpy as np
import torch
import os
def get_file_list(config):
    train_path = config['dataset']['train_list']
    val_path = config['dataset']['val_list']
    train_list = []
    val_list = []
    for entry in os.scandir(train_path):
        if entry.is_file():
            train_list.append(entry.path)
    for entry in os.scandir(val_path):
        if entry.is_file():
            val_list.append(entry.path)
    return train_list, val_list

def random_mirror_flip(imgs_array, prob=0.5):
    """
    Perform flip along each axis with the given probability; Do it for all voxels
    labels should also be flipped along the same axis.
    :param imgs_array:
    :param prob:
    :return:
    """
    for axis in range(1, len(imgs_array.shape)):
        random_num = np.random.random()
        if random_num >= prob:
            if axis == 1:
                imgs_array = imgs_array[:, ::-1, :]
            if axis == 2:
                imgs_array = imgs_array[:, :, ::-1]
    return imgs_array
def preprocess_img(img):
    c = img.shape[0]
    for i in range(0, c-1):
        a = img[i].max()
        img[i] = img[i] / (a + 1e-5)
    return img
def random_crop(img, crop_size):
    c, h, w = img.shape
    min_size = min(h, w)
    assert min_size > crop_size[0]
    h_margin = (h - crop_size[0]) // 2
    w_margin = (w - crop_size[1]) // 2
    random_h = int(np.random.uniform(0, h_margin, 1))
    random_w = int(np.random.uniform(0, w_margin, 1))
    new_img = img[:, h_margin-random_h:h_margin+crop_size[0]-random_h, w_margin-random_w:w_margin+crop_size[1]-random_w]
    return new_img
def preprocess_label(img, single_label=None):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """

    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET) - orange
    ed = img == 2  # Peritumoral Edema (ED) - yellow
    et = img == 4  # GD-enhancing Tumor (ET) - blue
    bg = (img!=1)*(img!=2)*(img!=4)
    # print("ed",et.shape)
    if not single_label:
        # return np.array([ncr, ed, et], dtype=np.uint8)
        return np.array([ed, bg, ncr, et], dtype=np.uint8)
    elif single_label == "WT":
        img[ed] = 1
        img[et] = 1
    elif single_label == "TC":
        img[ncr] = 0
        img[ed] = 1
        img[et] = 1
    elif single_label == "ET":
        img[ncr] = 0
        img[ed] = 0
        img[et] = 1
    else:
        raise RuntimeError("the 'single_label' type must be one of WT, TC, ET, and None")
    # print("image", img.shape)
    return img[np.newaxis, :]
class BTS_data(Dataset):
    def __init__(self, config, file, is_train=True):
        super(BTS_data, self).__init__()
        self.crop_size = config['dataset']['random_crop']
        #self.slice = opt.slice
        self.file = file
        self.is_train = is_train
        if config['dataset']['num_modals'] == 'full':
            self.num_modals = 4
        elif isinstance(config['dataset']['num_modals'], int):
            self.num_modals = config['dataset']['num_modals']
        self.input_resolution = config['dataset']['random_crop']
        self.final_resolution = config['model']['network']['down_scaling_factor']
        self.normalize = config['dataset']['normalize']
        self.format = config['dataset']['format']
        self.path = config['dataset']['data_folder']
        self.path = 'datasets/dataset_npy'
        self.year = config['dataset']['year']
        year = self.year
        '''if year == 2018:
            self.train = 'train'
            self.val = 'val'
            self.dataset = 'dataset'
        if year == 2019:
            self.train = 'train'
            self.val = 'train'
            self.dataset = 'dataset_2019'
        if year == 2021:
            self.train = 'train'
            self.val = 'train'
            self.dataset = 'dataset_2021'
            '''
    def __len__(self):
        return len(self.file)
    def __getitem__(self, index):
        mask = torch.randint(0, self.num_modals, double_tuple(self.final_resolution)).float()
        full_resolution_mask = F.interpolate(mask, size=double_tuple(self.input_resolution), mode='nearest')
        mask = full_resolution_mask.numpy()
        if self.format == 'npy':
            if self.mode == 'train':
                path = os.path.join(self.path, 'train', self.file[index])
            else:
                path = os.path.join(self.path, 'val', self.file[index])
            imgs_npy = np.load(path)[0]
        elif self.format == 'nii':
            if self.mode == 'train':
                path = self.file[index]
            else:
                path = self.file[index]
            imgs_npy = nib.load(path).get_fdata()
        if self.normalize:
            # normalize the image
            imgs_npy = preprocess_img(imgs_npy)
        '''file = os.listdir(path)
        flair = (nib.load(os.path.join(path, file[0]))).get_fdata().transpose(2,0,1)[None]
        seg = (nib.load(os.path.join(path, file[1]))).get_fdata().transpose(2,0,1)[None]
        t1c = (nib.load(os.path.join(path, file[3]))).get_fdata().transpose(2,0,1)[None]
        t2 = (nib.load(os.path.join(path, file[4]))).get_fdata().transpose(2,0,1)[None]
        flair, t2, t1c, seg = self.get_slices(flair, t2, t1c, seg)
        imgs_npy = np.concatenate((t1c, t2, flair, seg), axis=0)'''
        cur_with_label = imgs_npy.copy()
        cur_with_label = random_crop(cur_with_label, self.crop_size)
        cur_with_label = random_mirror_flip(cur_with_label)
        target_image = np.zeros(shape=double_tuple(self.input_resolution))
        flair = cur_with_label[1]
        t2 = cur_with_label[4]
        t1c = cur_with_label[7]
        t1 = cur_with_label[10]
        for modal, lab in zip([flair, t2, t1c, t1], range(self.num_modals)):
            bin_mask = mask == lab
            target_image = target_image + np.uint8(bin_mask) * modal

        source_data = target_image
        #source_data = cur_with_label[0:12]
        label = preprocess_label(cur_with_label[-1])
        #source_data = preprocess_img(source_data)
        return torch.from_numpy(source_data.copy()), torch.from_numpy(label.copy())
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