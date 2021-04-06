import os
import time 

import nibabel as nib
from nilearn import plotting
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from skimage.transform import resize
from tqdm import tqdm


import functools
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torch.nn.functional as F
# from torchsummary import summary
import torchio
import pathlib
from torchvision.transforms import *
import random
%matplotlib inline


class MriData(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(MriData, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y).long()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HCP_MRI(data.Dataset):
    """
    Arguments:
        paths: paths to data folders
        target_path: path to file with targets and additional information
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
    """

    def __init__(self, paths, target_path, load_online=False, transform=None):

        self.load_online = load_online
        self.transform = transform
        
        self.mri_paths = {
            "participant_id": [],
            "path": []
        }
        #         behavioral_data = pd.read_csv(target_path)
        behavioral_data = target_path
        behavioral_data.set_index('Subject',inplace=True)
        y_gender = []
        
        for patient_folder_name in os.listdir(paths):
            if 'nii' in patient_folder_name:
                sub = patient_folder_name[:6]
                if int(sub) in behavioral_data.index.values:
                    full_path = '{}/{}_3T_T1w_MPR1.nii.gz'.format(paths, sub)
                
                    if os.path.exists(full_path):
                        self.mri_paths["participant_id"].append(sub)
                        self.mri_paths["path"].append(full_path)
                        y_gender.append(behavioral_data['Gender'][int(sub)])

        self.mri_paths = pd.DataFrame(self.mri_paths)
    
        self.mri_paths['Gender'] = y_gender
        self.labels = np.asarray(pd.get_dummies(self.mri_paths, columns=['Gender'])['Gender_M'].astype('long').values) #F=0,M=1
        self.mri_paths = self.mri_paths["path"].tolist()
        self.pids = behavioral_data.index.values
        assert len(set(self.pids)) == len(self.pids)
        del behavioral_data
        
        if not self.load_online:
            self.mri_files = [self.get_image(i) for i in tqdm(range(len(self.mri_paths)))]


    def get_image(self, index):
        def load_mri(mri_path):
            if "nii" in mri_path:
                img = load_nii_to_array(mri_path)  # 2.5s
            else:
                img = np.load(mri_path)  # 1s
            return img
        
        img = load_mri(self.mri_paths[index])
        return img

    def __getitem__(self, index):
        if not self.load_online:
            item = self.mri_files[index]
        else:
            item = self.get_image(index)
        if self.transform is not None:
            item = self.transform(item)
        item = torch.from_numpy(item).transpose(0,1).transpose(1,2)[None, :, :, :].type(torch.float32)
   
        return (item, self.labels[index])

    def __len__(self):
        return len(self.mri_paths)
    
    
def data_preproc(PATH_TO_MRI,behavioral_path, save_path):
    
    behavioral_data = pd.read_csv(behavioral_path)
    y_id = []
    for file_name in os.listdir(PATH_TO_MRI):
        if '.nii' in file_name:
            y_id.append(int(file_name[:6]))
    available_id = pd.DataFrame()
    available_id['id'] = y_id
    behavioral_data.set_index('Subject',inplace=True)
    y_gender = []
    for i in available_id['id']:
        y_gender.append(behavioral_data['Gender'][i])
    available_id['Gender'] = y_gender
    y = pd.get_dummies(available_id, columns=['Gender'])['Gender_M'].values #F=0,M=1
    np.save(os.path.join(save_path, 'labels'), y)
    print('Labels saved')
    print(y.shape)

    transform = CropOrPad(
     (180, 180, 180))
    imgs = []
    for i, idd in enumerate(available_id['id']):
        mri_dir = PATH_TO_MRI
        file= '{}.nii'.format(idd)
        full_path=os.path.join(mri_dir,file)
        img = torchio.Image(full_path, torchio.INTENSITY).data
        img_crop = transform(img)
        imgs.append(img_crop)
        if i%25 == 0:
            print('{} iteration is finished.'.format(i))

    X = np.stack(imgs,axis=0)
#     np.savez_compressed(os.path.join(save_path, 'tensors_cut'), X)
    np.save(os.path.join(save_path, 'tensors_cut'), X)
    del imgs #deleting for freeing space on disc
    print(X.shape)