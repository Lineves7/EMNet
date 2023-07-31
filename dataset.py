import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch,Augment_RGB_torch_nocrop,is_image_file
import random

from torchvision import transforms as t

augment   = Augment_RGB_torch()
augment_nocrop   = Augment_RGB_torch_nocrop()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]

transforms_aug_nocrop = [method for method in dir(augment_nocrop) if callable(getattr(augment_nocrop, method)) if not method.startswith('_')]


##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'high'
        input_dir = 'low'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        degarded_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x)    for x in clean_files if is_png_file(x)]
        self.degarded_filenames = [os.path.join(rgb_dir, input_dir, x) for x in degarded_files if is_png_file(x)]
        
        self.img_options=img_options
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        degarded = torch.from_numpy(np.float32(load_img(self.degarded_filenames[tar_index])))
        
        clean = clean.permute(2,0,1)
        degarded = degarded.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        degarded_filename = os.path.split(self.degarded_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        degarded = degarded[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        # change
        degarded_mem = self.norm(degarded)
        degarded_mem = getattr(augment, apply_trans)(degarded_mem)
        clean = getattr(augment, apply_trans)(clean)
        degarded = getattr(augment, apply_trans)(degarded)
        idx = index + 0.0
        idx = np.array(([idx])).astype(np.float32)[0]

        return clean, degarded, clean_filename, degarded_filename,idx,degarded_mem



class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'high'
        input_dir = 'low'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        degarded_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_image_file(x)]
        self.degarded_filenames = [os.path.join(rgb_dir, input_dir, x) for x in degarded_files if is_image_file(x)]
        

        self.tar_size = len(self.clean_filenames)
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        degarded = torch.from_numpy(np.float32(load_img(self.degarded_filenames[tar_index])))

                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        degarded_filename = os.path.split(self.degarded_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        degarded = degarded.permute(2,0,1)

        # change
        degarded__mem = self.norm(degarded)

        idx = index + 0.0
        idx = np.array(([idx])).astype(np.float32)[0]

        return clean, degarded, clean_filename, degarded_filename,idx,degarded__mem


class DataLoaderVal_pic(Dataset):
    def __init__(self, rgb_dir, output_dir, target_transform=None):
        super(DataLoaderVal_pic, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'high'
        input_dir = 'low'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        degarded_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        output_files = sorted(os.listdir(output_dir))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.degarded_filenames = [os.path.join(rgb_dir, input_dir, x) for x in degarded_files if is_png_file(x)]
        self.output_filenames = [os.path.join(output_dir, x) for x in output_files if is_png_file(x)]


        self.tar_size = len(self.clean_filenames)
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size


        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        degarded = torch.from_numpy(np.float32(load_img(self.degarded_filenames[tar_index])))
        output = torch.from_numpy(np.float32(load_img(self.output_filenames[tar_index])))


        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        degarded_filename = os.path.split(self.degarded_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        degarded = degarded.permute(2,0,1)
        output = output.permute(2,0,1)

        # change
        degarded__mem = self.norm(degarded)

        idx = index + 0.0
        idx = np.array(([idx])).astype(np.float32)[0]

        return clean, degarded, clean_filename, degarded_filename,idx,degarded__mem,output


