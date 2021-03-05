import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.append("../renderer/")

import nmr_test as nmr
import neural_renderer


class MyDataset(Dataset):
    def __init__(self, data_dir, img_size, texture_size, faces, vertices, distence=None, mask_dir='', ret_mask=False):
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        for file in files:
            if distence is None:
                self.files.append(file)
            else:
                data = np.load(os.path.join(self.data_dir, file))
                veh_trans = data['veh_trans']
                cam_trans = data['cam_trans']
                dis = (cam_trans - veh_trans)[0, :]
                dis = np.sum(dis ** 2)
                # print(dis)
                if dis <= distence:
                    self.files.append(file)
        print(len(self.files))
        self.img_size = img_size
        textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
        self.textures = torch.from_numpy(textures).cuda(device=0)
        self.faces_var = torch.from_numpy(faces[None, :, :]).cuda(device=0)
        self.vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
        self.mask_renderer = nmr.NeuralRenderer(img_size=self.img_size).cuda()
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
        # print(self.files)
    
    def set_textures(self, textures):
        self.textures = textures
    
    def __getitem__(self, index):
        # index = 5
        
        # print(index)
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file)
        img = data['img']
        veh_trans = data['veh_trans']
        cam_trans = data['cam_trans']
        
        eye, camera_direction, camera_up = nmr.get_params(cam_trans, veh_trans)
        
        self.mask_renderer.renderer.renderer.eye = eye
        self.mask_renderer.renderer.renderer.camera_direction = camera_direction
        self.mask_renderer.renderer.renderer.camera_up = camera_up 

        imgs_pred = self.mask_renderer.forward(self.vertices_var, self.faces_var, self.textures)
        # masks = imgs_pred[:, 0, :, :] | imgs_pred[:, 1, :, :] | imgs_pred[:, 2, :, :]
        # print(masks.size())
        
        img = img[:, :, ::-1]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0)
        # print(img.size())
        # print(imgs_pred.size())
        imgs_pred = imgs_pred / torch.max(imgs_pred)
        total_img = img + 255 * imgs_pred
        
        
        if self.ret_mask:
            mask_file = os.path.join(self.mask_dir, self.files[index][:-4] + '.png')
            mask = cv2.imread(mask_file)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
            mask = torch.from_numpy(mask.astype('float32')).cuda()
            # print(mask.size())
            # print(torch.max(mask))
            return index, total_img.squeeze(0) , imgs_pred.squeeze(0), mask
        return index, total_img.squeeze(0) , imgs_pred.squeeze(0)
    
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    obj_file = 'audi_et_te.obj'
    vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, load_texture=True)
    dataset = MyDataset('../data/phy_attack/train/', 608, 4, faces, vertices)
    loader = DataLoader(
        dataset=dataset,   
        batch_size=3,     
        shuffle=True,            
        #num_workers=2,              
    )
    
    for img, car_box in loader:
        print(img.size(), car_box.size())
