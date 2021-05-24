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
        behavioral_data = target_path
        behavioral_data.set_index('Subject',inplace=True)
        y_age = []
        behavioral_data.loc[(behavioral_data['Age']=='22-25') | (behavioral_data['Age']=='26-30'), 'lower_30'] = 0
        behavioral_data.loc[(behavioral_data['Age']=='31-35') | (behavioral_data['Age']=='36+'), 'lower_30'] = 1
        
        
        for patient_folder_name in os.listdir(paths):
            if 'nii' in patient_folder_name:
                sub = patient_folder_name[:6]
                if int(sub) in behavioral_data.index.values:
                    full_path = '{}/{}_3T_T1w_MPR1_bet_scale.nii.gz'.format(paths, sub)
                
                    if os.path.exists(full_path):
                        self.mri_paths["participant_id"].append(sub)
                        self.mri_paths["path"].append(full_path)
                        y_age.append(behavioral_data['lower_30'][int(sub)].astype('long'))

        self.mri_paths = pd.DataFrame(self.mri_paths)
    
        self.mri_paths['Age'] = y_age
        self.labels = y_age
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
#         print('curent img shape', img.shape)
        
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
    
    
class HCP_MRI_domain_scale(data.Dataset):
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
        #         behavioral_data = pd.read_csv(target_path)
        behavioral_data = target_path
        behavioral_data.set_index('Subject',inplace=True)
        y_gender = []
        
        for patient_folder_name in os.listdir(paths):
            if 'nii' in patient_folder_name:
                sub = patient_folder_name[:6]
                if int(sub) in behavioral_data.index.values:
                    full_path = '{}/{}_3T_T1w_MPR1_bet_scale.nii.gz'.format(paths, sub)
                
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

#     def reshape_image(self, img, coord_min):
#         subj_bool = img != 0
# #         subj_bool = subj_bool.reshape( (256, 320, 320,1))
#         ax_zero_cut = torch.from_numpy(subj_bool).max(dim=2).values.max(dim=1).values.data.numpy()
#         ax_one_cut = torch.from_numpy(subj_bool).max(dim=2).values.max(dim=0).values.data.numpy()
#         ax_two_cut = torch.from_numpy(subj_bool).max(dim=1).values.max(dim=0).values.data.numpy()
        
#         ax_zero_min, ax_zero_max = np.where(ax_zero_cut)[0][[0, -1]]
#         ax_one_min, ax_one_max = np.where(ax_one_cut)[0][[0, -1]]
#         ax_two_min, ax_two_max = np.where(ax_two_cut)[0][[0, -1]]
#         img = img[
#               ax_zero_min:ax_zero_max + 1,
#               ax_one_min:ax_one_max + 1,
#               ax_two_min:ax_two_max + 1,
#               ]
# #         img = img[
# #               coord_min[0]:256 - coord_min[1],
# #               coord_min[2]:320 - coord_min[3],
# #               coord_min[4]:320 - coord_min[5],
# #               ]
# #         if img.shape[:3] != img_shape:
# #             print("Current image shape: {}".format(img.shape[:3]))
# #             print("Desired image shape: {}".format(img_shape))
# #             raise AssertionError
# #         img = img.reshape((1,) + img.shape)
#         return img

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
        img = load_mri(self.mri_paths[index])
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
    
    
class HCP_MRI_sp_aug(data.Dataset):
    """
    Arguments:
        paths: paths to data folders
        target_path: path to file with targets and additional information
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
    """

    def __init__(self, paths, target_path, load_online=False, hcp_type = '', coord_min=(20, 20, 20,),
                 img_shape=(152, 188, 152,), transform_man=None, transform_woman=None, transform = None ):

        self.load_online = load_online
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.transform_man = transform_man
        self.transform_woman = transform_woman
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

#     def reshape_image(self, img, coord_min):
#         subj_bool = img != 0
# #         subj_bool = subj_bool.reshape( (256, 320, 320,1))
#         ax_zero_cut = torch.from_numpy(subj_bool).max(dim=2).values.max(dim=1).values.data.numpy()
#         ax_one_cut = torch.from_numpy(subj_bool).max(dim=2).values.max(dim=0).values.data.numpy()
#         ax_two_cut = torch.from_numpy(subj_bool).max(dim=1).values.max(dim=0).values.data.numpy()
        
#         ax_zero_min, ax_zero_max = np.where(ax_zero_cut)[0][[0, -1]]
#         ax_one_min, ax_one_max = np.where(ax_one_cut)[0][[0, -1]]
#         ax_two_min, ax_two_max = np.where(ax_two_cut)[0][[0, -1]]
#         img = img[
#               ax_zero_min:ax_zero_max + 1,
#               ax_one_min:ax_one_max + 1,
#               ax_two_min:ax_two_max + 1,
#               ]
# #         img = img[
# #               coord_min[0]:256 - coord_min[1],
# #               coord_min[2]:320 - coord_min[3],
# #               coord_min[4]:320 - coord_min[5],
# #               ]
# #         if img.shape[:3] != img_shape:
# #             print("Current image shape: {}".format(img.shape[:3]))
# #             print("Desired image shape: {}".format(img_shape))
# #             raise AssertionError
# #         img = img.reshape((1,) + img.shape)
#         return img

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
        img = load_mri(self.mri_paths[index])
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
#         print(self.labels[index])
        if self.labels[index] == 0:
                if self.transform_woman is not None:
#                     print('check woman')
                    item = self.transform_woman(item)
        elif self.labels[index] == 1:
                if self.transform_man is not None:
#                     print('check man')
                    item = self.transform_man(item)
        if self.transform is not None:
                    item = self.transform(item)
#         print(item.shape)
#         print( time.time() - start_time)
#         item = torch.from_numpy(item).type(torch.float32)
#         print(item.shape)
        item = item.transpose(0,1).transpose(1,2)[None, :, :, :].type(torch.float32)
#         print(item.shape)
        return (item, self.labels[index])

    def __len__(self):
        return len(self.mri_paths)
    
    

class HCP_MRI_reshape(data.Dataset):
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

#     def reshape_image(self, img, coord_min):
#         subj_bool = img != 0
# #         subj_bool = subj_bool.reshape( (256, 320, 320,1))
#         ax_zero_cut = torch.from_numpy(subj_bool).max(dim=2).values.max(dim=1).values.data.numpy()
#         ax_one_cut = torch.from_numpy(subj_bool).max(dim=2).values.max(dim=0).values.data.numpy()
#         ax_two_cut = torch.from_numpy(subj_bool).max(dim=1).values.max(dim=0).values.data.numpy()
        
#         ax_zero_min, ax_zero_max = np.where(ax_zero_cut)[0][[0, -1]]
#         ax_one_min, ax_one_max = np.where(ax_one_cut)[0][[0, -1]]
#         ax_two_min, ax_two_max = np.where(ax_two_cut)[0][[0, -1]]
#         img = img[
#               ax_zero_min:ax_zero_max + 1,
#               ax_one_min:ax_one_max + 1,
#               ax_two_min:ax_two_max + 1,
#               ]
# #         img = img[
# #               coord_min[0]:256 - coord_min[1],
# #               coord_min[2]:320 - coord_min[3],
# #               coord_min[4]:320 - coord_min[5],
# #               ]
# #         if img.shape[:3] != img_shape:
# #             print("Current image shape: {}".format(img.shape[:3]))
# #             print("Desired image shape: {}".format(img_shape))
# #             raise AssertionError
# #         img = img.reshape((1,) + img.shape)
#         return img

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
        img = load_mri(self.mri_paths[index])
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
    
    

class HCP_MRI_crop_resize(data.Dataset):
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
        #         behavioral_data = pd.read_csv(target_path)
        behavioral_data = target_path
        behavioral_data.set_index('Subject',inplace=True)
        y_gender = []
        
        for patient_folder_name in os.listdir(paths):
            if 'npy' in patient_folder_name:
                sub = patient_folder_name[:6]
                if int(sub) in behavioral_data.index.values:
                    full_path = '{}/{}_3T_T1w_MPR1_bet_mask_crop.npy'.format(paths, sub)
                
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

#     def reshape_image(self, img, coord_min):
#         subj_bool = img != 0
# #         subj_bool = subj_bool.reshape( (256, 320, 320,1))
#         ax_zero_cut = torch.from_numpy(subj_bool).max(dim=2).values.max(dim=1).values.data.numpy()
#         ax_one_cut = torch.from_numpy(subj_bool).max(dim=2).values.max(dim=0).values.data.numpy()
#         ax_two_cut = torch.from_numpy(subj_bool).max(dim=1).values.max(dim=0).values.data.numpy()
        
#         ax_zero_min, ax_zero_max = np.where(ax_zero_cut)[0][[0, -1]]
#         ax_one_min, ax_one_max = np.where(ax_one_cut)[0][[0, -1]]
#         ax_two_min, ax_two_max = np.where(ax_two_cut)[0][[0, -1]]
#         img = img[
#               ax_zero_min:ax_zero_max + 1,
#               ax_one_min:ax_one_max + 1,
#               ax_two_min:ax_two_max + 1,
#               ]
# #         img = img[
# #               coord_min[0]:256 - coord_min[1],
# #               coord_min[2]:320 - coord_min[3],
# #               coord_min[4]:320 - coord_min[5],
# #               ]
# #         if img.shape[:3] != img_shape:
# #             print("Current image shape: {}".format(img.shape[:3]))
# #             print("Desired image shape: {}".format(img_shape))
# #             raise AssertionError
# #         img = img.reshape((1,) + img.shape)
#         return img

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
        img = load_mri(self.mri_paths[index])
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
        item = item[None, :, :, :]
#         print(item.shape)
#         start_time = time.time()
        if self.transform is not None:
            item = self.transform(item)
#         print(item.shape)
#         print( time.time() - start_time)
        item = torch.from_numpy(item).type(torch.float32)
#         item = item.transpose(0,1).transpose(1,2)[None, :, :, :].type(torch.float32)
   
        return (item, self.labels[index])

    def __len__(self):
        return len(self.mri_paths)
    
    
    
class HCP_MRI_crop(data.Dataset):
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
        #         behavioral_data = pd.read_csv(target_path)
        behavioral_data = target_path
        behavioral_data.set_index('Subject',inplace=True)
        y_gender = []
        
        for patient_folder_name in os.listdir(paths):
            sub = patient_folder_name[:6]
            if int(sub) in behavioral_data.index.values:
                full_path = '{}/{}_3T_T1w_MPR1_bet_mask.nii.gz'.format(paths, sub)
                if os.path.exists(full_path):
                    self.mri_paths["participant_id"].append(sub)
                    self.mri_paths["path"].append(full_path)
                    y_gender.append(behavioral_data['Gender'][int(sub)])

#         for patient_folder_name in os.listdir(paths):
#             if len(patient_folder_name) == 6:
#                     path_to_patient_folder = paths + '/' + patient_folder_name
#                     full_path = '{}/unprocessed/3T/T1w_{}/{}_3T_T1w_{}.nii.gz'.format(path_to_patient_folder, hcp_type, patient_folder_name, hcp_type)
#                     if os.path.exists(full_path):
#                         self.mri_paths["participant_id"].append(patient_folder_name)
#                         self.mri_paths["path"].append(full_path)

        self.mri_paths = pd.DataFrame(self.mri_paths)
#         behavioral_data = pd.read_csv(target_path)
#         behavioral_data = target_path
#         behavioral_data.set_index('Subject',inplace=True)
#         y_gender = []
        
#         for i in self.mri_paths["participant_id"]:
#             if int(i) in behavioral_data.index.values:
#                 y_gender.append(behavioral_data['Gender'][int(i)])
#             else:
#                 print('Not in data')
#                 print(i)
    
        self.mri_paths['Gender'] = y_gender
#         print(self.mri_paths['Gender'])
        self.labels = np.asarray(pd.get_dummies(self.mri_paths, columns=['Gender'])['Gender_M'].astype('long').values) #F=0,M=1
        self.mri_paths = self.mri_paths["path"].tolist()
        self.pids = behavioral_data.index.values
        assert len(set(self.pids)) == len(self.pids)
        del behavioral_data
        
        if not self.load_online:
            self.mri_files = [self.get_image(i) for i in tqdm(range(len(self.mri_paths)))]

    def reshape_image(self, img, coord_min):
        img = img[
              coord_min[0]:256 - coord_min[1],
              coord_min[2]:320 - coord_min[3],
              coord_min[4]:320 - coord_min[5],
              ]
#         if img.shape[:3] != img_shape:
#             print("Current image shape: {}".format(img.shape[:3]))
#             print("Desired image shape: {}".format(img_shape))
#             raise AssertionError
#         img = img.reshape((1,) + img.shape[1:])
        return img

    def get_image(self, index):
        def load_mri(mri_path):
            if "nii" in mri_path:
                print('nii')
                img = load_nii_to_array(mri_path)  # 2.5s
            else:
                print('np')
                img = np.load(mri_path)  # 1s
            return img
        start_time = time.time()
        img = load_mri(self.mri_paths[index])
        print( time.time() - start_time)
#         print('curent img shape', img.shape)
        start_time = time.time()
        img = self.reshape_image(img, self.coord_min)
        print( time.time() - start_time)
        return img

    def __getitem__(self, index):
        if not self.load_online:
            item = self.mri_files[index]
        else:
            item = self.get_image(index)
        start_time = time.time()
        if self.transform is not None:
            item = self.transform(item)
        print( time.time() - start_time)
        start_time = time.time()
        item = item.transpose(0,1).transpose(1,2)[None, :, :, :].type(torch.float32)
        print( time.time() - start_time)
        return (item, self.labels[index])

    def __len__(self):
        return len(self.mri_paths)
    
    

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
        #         behavioral_data = pd.read_csv(target_path)
        behavioral_data = target_path
        behavioral_data.set_index('Subject',inplace=True)
        y_gender = []
        
        for patient_folder_name in os.listdir(paths):
            sub = patient_folder_name[:6]
            if int(sub) in behavioral_data.index.values:
                full_path = '{}/{}_3T_T1w_MPR1_bet_mask.nii.gz'.format(paths, sub)
                if os.path.exists(full_path):
                    self.mri_paths["participant_id"].append(sub)
                    self.mri_paths["path"].append(full_path)
                    y_gender.append(behavioral_data['Gender'][int(sub)])

#         for patient_folder_name in os.listdir(paths):
#             if len(patient_folder_name) == 6:
#                     path_to_patient_folder = paths + '/' + patient_folder_name
#                     full_path = '{}/unprocessed/3T/T1w_{}/{}_3T_T1w_{}.nii.gz'.format(path_to_patient_folder, hcp_type, patient_folder_name, hcp_type)
#                     if os.path.exists(full_path):
#                         self.mri_paths["participant_id"].append(patient_folder_name)
#                         self.mri_paths["path"].append(full_path)

        self.mri_paths = pd.DataFrame(self.mri_paths)
#         behavioral_data = pd.read_csv(target_path)
#         behavioral_data = target_path
#         behavioral_data.set_index('Subject',inplace=True)
#         y_gender = []
        
#         for i in self.mri_paths["participant_id"]:
#             if int(i) in behavioral_data.index.values:
#                 y_gender.append(behavioral_data['Gender'][int(i)])
#             else:
#                 print('Not in data')
#                 print(i)
    
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
    
    
