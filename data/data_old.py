import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm


def load_nii_to_array(nii_path):
    return nib.load(nii_path).get_data()



class HCP_MRI(data.Dataset):
    """
    Arguments:
        paths: paths to data folders
        target_path: path to file with targets and additional information
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
    """

    def __init__(self, paths, target_path, load_online=False, hcp_type = '', coord_min=(20, 20, 20,),
                 img_shape=(152, 188, 152,), transform=None):

        self.load_online = load_online
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.transform = transform
        
        self.mri_paths = {
            "participant_id": [],
            "path": []
        }

#         for patient_folder_name in os.listdir(paths):
#             sub = patient_folder_name[:6]
#             full_path = '{}/{}_3T_T1w_MPR1_bet_mask.nii.gz'.format(paths, sub)
#             if os.path.exists(full_path):
#                 self.mri_paths["participant_id"].append(sub)
#                 self.mri_paths["path"].append(full_path)

        for patient_folder_name in os.listdir(paths):
            if len(patient_folder_name) == 6:
                    path_to_patient_folder = paths + '/' + patient_folder_name
                    full_path = '{}/unprocessed/3T/T1w_{}/{}_3T_T1w_{}.nii.gz'.format(path_to_patient_folder, hcp_type, patient_folder_name, hcp_type)
                    if os.path.exists(full_path):
                        self.mri_paths["participant_id"].append(patient_folder_name)
                        self.mri_paths["path"].append(full_path)

        self.mri_paths = pd.DataFrame(self.mri_paths)
        behavioral_data = pd.read_csv(target_path)
        behavioral_data.set_index('Subject',inplace=True)
        y_gender = []
        
        for i in self.mri_paths["participant_id"]:
            if int(i) in behavioral_data.index.values:
                y_gender.append(behavioral_data['Gender'][int(i)])
            else:
                print('Not in data')
                print(i)
    
        self.mri_paths['Gender'] = y_gender
#         print(self.mri_paths['Gender'])
        self.labels = np.asarray(pd.get_dummies(self.mri_paths, columns=['Gender'])['Gender_M'].astype('long').values) #F=0,M=1
        self.mri_paths = self.mri_paths["path"].tolist()
        self.pids = behavioral_data.index.values
        assert len(set(self.pids)) == len(self.pids)
        del behavioral_data
        
        if not self.load_online:
            self.mri_files = [self.get_image(i) for i in tqdm(range(len(self.mri_paths)))]

    def reshape_image(self, img, coord_min, img_shape):
        img = img[
              coord_min[0]:coord_min[0] + img_shape[0],
              coord_min[1]:coord_min[1] + img_shape[1],
              coord_min[2]:coord_min[2] + img_shape[2],
              ]
        if img.shape[:3] != img_shape:
            print("Current image shape: {}".format(img.shape[:3]))
            print("Desired image shape: {}".format(img_shape))
            raise AssertionError
#         img = img.reshape((1,) + img_shape)
        return img

    def get_image(self, index):
        def load_mri(mri_path):
            if "nii" in mri_path:
                img = load_nii_to_array(mri_path)  # 2.5s
            else:
                img = np.load(mri_path)  # 1s
            return img

        img = load_mri(self.mri_paths[index])
#         print('curent img shape', img.shape)
        img = self.reshape_image(img, self.coord_min, self.img_shape)
        return img

    def __getitem__(self, index):
        if not self.load_online:
            item = self.mri_files[index]
        else:
            item = self.get_image(index)
        if self.transform is not None:
            item = self.transform(item)
        item = item.transpose(0,1).transpose(1,2)[None, :, :, :].type(torch.float32)
        return (item, self.labels[index])

    def __len__(self):
        return len(self.mri_paths)