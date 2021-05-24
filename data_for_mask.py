import os
import time 

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from skimage.transform import resize
from tqdm import tqdm


def load_nii_to_array(nii_path):
    return nib.load(nii_path).get_data()

   
    

class HCP_MRI_mask(data.Dataset):
    """
    Arguments:
        paths: paths to data folders
        target_path: path to file with targets and additional information
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
    """

    def __init__(self, paths, target_path,fold_path = None, mask_dir = None, load_online=False, hcp_type = '', coord_min=(20, 20, 20,),
                 img_shape=(152, 188, 152,), transform=None):

        self.load_online = load_online
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.transform = transform
        self.mask_path = mask_path
        self.fold_path = fold_path
        
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
                    full_path = '{}/{}_3T_T1w_MPR1_bet_mask_crop.nii.gz'.format(paths, sub)
                
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
#                 print('nii')
                img = load_nii_to_array(mri_path)  # 2.5s
            else:
#                 print('np')
                img = np.load(mri_path)  # 1s
            return img
#         start_time = time.time()
        if self.fold_path:
            img = load_mri(self.fold_paths[index])
        else:
            img = load_mri(self.mri_paths[index])
        if self.mask_dir:
            mask = load_mri(os.path.join(self.mask_dir,'{}_3T_T1w_MPR1_bet_binary_gradcam_mask.nii.gz'.format(self.fold_paths[index][-34:-28])))
            img = img.mul(mask)
#         print('curent img shape', img.shape)
#         print( time.time() - start_time)
#         img = self.reshape_image(img, self.coord_min)
#         img = resize(img, self.img_shape)
        return img

    def __getitem__(self, index):
        if not self.load_online:
            item = self.mri_files[index]
        else:
            item = self.get_image(index)
#         item = item[None, :, :, :]
#         print(item.shape)
#         start_time = time.time()
        if self.transform is not None:
            item = self.transform(item)
#         print(item.shape)
#         print( time.time() - start_time)
#         item = torch.from_numpy(item).type(torch.float32)
        item = item.transpose(0,1).transpose(1,2)[None, :, :, :].type(torch.float32)
   
        return (item, self.labels[index])

    def __len__(self):
        return len(self.mri_paths)
    
 
    
    
    
class HCP_MRI_age(data.Dataset):
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

        for patient_folder_name in os.listdir(paths):
            sub = patient_folder_name[:6]
            full_path = '{}/{}_3T_T1w_MPR1_bet_mask.nii.gz'.format(paths, sub)
            if os.path.exists(full_path):
                self.mri_paths["participant_id"].append(sub)
                self.mri_paths["path"].append(full_path)

#         for patient_folder_name in os.listdir(paths):
#             if len(patient_folder_name) == 6:
#                     path_to_patient_folder = paths + '/' + patient_folder_name
#                     full_path = '{}/unprocessed/3T/T1w_{}/{}_3T_T1w_{}.nii.gz'.format(path_to_patient_folder, hcp_type, patient_folder_name, hcp_type)
#                     if os.path.exists(full_path):
#                         self.mri_paths["participant_id"].append(patient_folder_name)
#                         self.mri_paths["path"].append(full_path)

        self.mri_paths = pd.DataFrame(self.mri_paths)
        behavioral_data = pd.read_csv(target_path)
        behavioral_data.set_index('Subject',inplace=True)
        y_age = []
        behavioral_data.loc[(behavioral_data['Age']=='22-25') | (behavioral_data['Age']=='26-30'), 'lower_30'] = 1
        behavioral_data.loc[(behavioral_data['Age']=='31-35') | (behavioral_data['Age']=='36+'), 'lower_30'] = 0
        for i in self.mri_paths["participant_id"]:
            if int(i) in behavioral_data.index.values:
                y_age.append(behavioral_data['lower_30'][int(i)])
            else:
                print('Not in data')
                print(i)
    
        self.mri_paths['Age'] = y_age
#         print(self.mri_paths['Gender'])
#         self.labels = np.asarray(pd.get_dummies(self.mri_paths, columns=['Age'])['Ager_M'].astype('long').values) #F=0,M=1
        self.labels = y_age
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