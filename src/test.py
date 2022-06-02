import os
import sys
from PIL import Image
import numpy as np
import tqdm
import torch
import cv2
import warnings
warnings.filterwarnings("ignore")


import torch.nn.functional as F

import nmr_test as nmr
import neural_renderer
from torchvision.transforms import Resize
from data_loader import MyDataset
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision import transforms

from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument("--texture", type=str, default='textures/smile_trained.npy', help="texture")
parser.add_argument("--obj", type=str, default='audi_et_te.obj')
parser.add_argument("--datapath", type=str)

args = parser.parse_args()

mask_dir = os.path.join(args.datapath, 'masks/')


obj_file = args.obj
texture_size = 6
vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size, load_texture=True)


log_dir = './'

img_size = 608
torch.autograd.set_detect_anomaly(True)



texture_param = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32') * -0.9# test 0
texture_param = torch.autograd.Variable(torch.from_numpy(texture_param).cuda(device=0), requires_grad=True)

texture_origin = torch.from_numpy(textures[None, :, :, :, :, :]).cuda(device=0)

texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
with open('all_faces.txt', 'r') as f:
    face_ids = f.readlines()
    # print(face_ids)
    for face_id in face_ids:
        if face_id != '\n':
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)





def cal_texture():
    
    textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    # return texture_origin
    return texture_origin * (1 - texture_mask) + texture_mask * textures
    # return texture_origin * (1 - texture_mask) + texture_mask * textures
    # return  texture_canny * texture_mask * (0.9 * textures + 0.1 * texture_origin) + \
    #         (1 - texture_canny) * texture_mask * (0.5 * textures + 0.5 * texture_origin) + \
    #         (1 - texture_mask) * texture_origin
    # return texture_mask_top * texture_top + texture_mask_front * texture_front + texture_mask_side * texture_side + (1 - texture_mask) * texture_origin

            


def preprocess_image(img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img
        for i in range(3):
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] - means[i]
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] / stds[i]
        
        return preprocessed_img

    
label_list = [468,511,609,817,581,751,627]#, 436]


def resnet152(data_dir, epoch, train=True, batch_size=1):
    
    model = models.resnet152(pretrained=True)
    model.eval().cuda()
    
    print(data_dir)
    dataset = MyDataset(data_dir, 224, texture_size, faces, vertices, distence=50, mask_dir=mask_dir, ret_mask=True)
    loader = DataLoader(
        dataset=dataset,     
        batch_size=batch_size,  
        shuffle=False,        
        # num_workers=2,     
    )
    
    textures = cal_texture()
    # print(textures) # wjk tested
    dataset.set_textures(textures)
    print(len(dataset))
    for _ in range(epoch):
        print('Epoch: ', _, '/', epoch)
        count = 0
        tqdm_loader = tqdm.tqdm(loader)
        for i, (index, total_img, texture_img, masks)  in enumerate(tqdm_loader):
            index = int(index[0])
            
            
            total_img_np = total_img.data.cpu().numpy()[0]

            total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))

            total_img_np.save(os.path.join(log_dir, 'test_total.jpg')) 

            Image.fromarray((255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(log_dir, 'texture2.png'))

            total_img = preprocess_image(total_img/255)
            output = model(total_img)
            prob = F.softmax(output)
            
            prob_index = np.argmax(output.cpu().data.numpy())
            
            if prob_index in label_list:
                count += 1
                with open('right.txt', 'a') as f:
                    f.write(str(index) + '\n')
            
            tqdm_loader.set_description('acc %d, index %d' % (count, prob_index))

            
            textures = cal_texture()
            dataset.set_textures(textures)
        
        with open("new_acc.txt", "a") as f:
            f.write('resnet152: ' + args.obj + '+' + args.texture + ': ' + str(count / len(loader)) + '\n')
        

        

def densenet(data_dir, epoch, train=True, batch_size=1):
    model = models.densenet201(pretrained=True)
    model.eval().cuda()
    image_transforms = transforms.Compose([
            transforms.Resize(size=224),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    print(data_dir)
    dataset = MyDataset(data_dir, 224, texture_size, faces, vertices, distence=50, mask_dir=mask_dir, ret_mask=True)
    loader = DataLoader(
        dataset=dataset,     
        batch_size=batch_size,  
        shuffle=False,        
        # num_workers=2,     
    )
    
    textures = cal_texture()

    dataset.set_textures(textures)
    print(len(dataset))
    for _ in range(epoch):
        print('Epoch: ', _, '/', epoch)
        count = 0
        tqdm_loader = tqdm.tqdm(loader)
        for i, (index, total_img, texture_img, masks) in enumerate(tqdm_loader):
            index = int(index[0])
            
            
            total_img_np = total_img.data.cpu().numpy()[0]

            total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))

            total_img_np.save(os.path.join(log_dir, 'test_total.jpg')) 

            Image.fromarray((255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(log_dir, 'texture2.png'))

            total_img = total_img.float()#.div(255.0)
            
            PIL_image = Image.fromarray(np.uint8(total_img_np))
            total_img = image_transforms(PIL_image).unsqueeze(0)

            total_img = total_img.cuda()
            
            
            output = model(total_img)
            prob = F.softmax(output)
            
            prob_index = np.argmax(output.cpu().data.numpy())
            
            if prob_index in label_list:
                count += 1
                with open('right.txt', 'a') as f:
                    f.write(str(index) + '\n')
            
            tqdm_loader.set_description('acc %d, index %d' % (count, prob_index))

            
            textures = cal_texture()
            dataset.set_textures(textures)
        
        with open("new_acc.txt", "a") as f:
            f.write('densenet201: '+args.obj + '+' + args.texture + ': ' + str(count / len(loader)) + '\n')

def vgg(data_dir, epoch, train=True, batch_size=1):
    
    model = models.vgg19(pretrained=True)
    model.eval().cuda()

    
    print(data_dir)
    dataset = MyDataset(data_dir, 224, texture_size, faces, vertices, distence=50, mask_dir=mask_dir, ret_mask=True)
    loader = DataLoader(
        dataset=dataset,     
        batch_size=batch_size,  
        shuffle=False,        
        # num_workers=2,     
    )
    
    textures = cal_texture()

    dataset.set_textures(textures)
    print(len(dataset))
    for _ in range(epoch):
        print('Epoch: ', _, '/', epoch)
        count = 0;
        tqdm_loader = tqdm.tqdm(loader)
        for i, (index, total_img, texture_img, masks) in enumerate(tqdm_loader):
            index = int(index[0])
            
            
            total_img_np = total_img.data.cpu().numpy()[0]

            total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))

            total_img_np.save(os.path.join(log_dir, 'test_total.jpg')) 

            Image.fromarray((255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(log_dir, 'texture2.png'))

            total_img = preprocess_image(total_img/255)
            output = model(total_img)
            prob = F.softmax(output)
            
            prob_index = np.argmax(output.cpu().data.numpy())
            
            if prob_index in label_list:
                count += 1
            
            tqdm_loader.set_description('acc %d, index %d' % (count, prob_index))

            
            textures = cal_texture()
            dataset.set_textures(textures)
        
        with open("new_acc.txt", "a") as f:
            f.write('vgg: '+args.obj + '+' + args.texture + ': ' + str(count / len(loader)) + '\n')

            
            
def iv3(data_dir, epoch, train=True, batch_size=1):
    model = models.inception_v3(pretrained=True)

    model.cuda().eval()
    image_transforms = transforms.Compose([
            transforms.Resize(size=224),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    print(data_dir)
    dataset = MyDataset(data_dir, 224, texture_size, faces, vertices, distence=50, mask_dir=mask_dir, ret_mask=True)
    loader = DataLoader(
        dataset=dataset,     
        batch_size=batch_size,  
        shuffle=train,        
        # num_workers=2,     
    )
    
    
    textures = cal_texture()
    dataset.set_textures(textures)
    print(len(dataset))
    
    for _ in range(epoch):
        print('Epoch: ', _, '/', epoch)
        count = 0
        i = 1
        tqdm_loader = tqdm.tqdm(loader)
        for index, total_img, texture_img, masks in tqdm_loader:
            
            index = index[0]
            
            total_img_np = total_img.data.cpu().numpy()[0]
            
            total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))

            total_img_np.save(os.path.join(log_dir, 'test_total.jpg')) 
            
            
            total_img = total_img.float()#.div(255.0)
            
            PIL_image = Image.fromarray(np.uint8(total_img_np))
            total_img = image_transforms(PIL_image).unsqueeze(0)
            
            total_img = total_img.cuda()
            
            
            out = model(total_img)
            out = out.squeeze()
            out_np = out.cpu().detach().numpy()
            #print(out_np)
            label = np.argmax(out_np)
            #print(label)
            
            if int(label) in label_list:
                count += 1
            '''
            else:
                with open('wrong_label.txt', 'a') as f:
                    f.write(str(i)+','+str(label) + '\n')
            # print(texture_param)
            '''
            with open('pre_label_resnet.txt', 'a') as f:
                    f.write(str(label) + '\n')
            i += 1
            tqdm_loader.set_description('acc: %d' % (count))
            # print(count)
            
            textures = cal_texture()
            dataset.set_textures(textures)
            
        with open('new_acc.txt', 'a') as f:
            f.write('iv3: '+args.obj + '+' + args.texture + ': ' + str(count / len(loader)) + '\n')
            
            
if __name__ == '__main__':
    
    print(args.texture, args.obj)

    train_dir = os.path.join(args.datapath, 'phy_attack/train/')
    test_dir = os.path.join(args.datapath, 'phy_attack/test/')
    
    texture_param = torch.autograd.Variable(torch.from_numpy(np.load(os.path.join(log_dir, args.texture))).cuda(device=0), requires_grad=True)

    resnet152(test_dir, 1, train=False, batch_size=1)
    densenet(test_dir, 1, train=False, batch_size=1)
    iv3(test_dir, 1, train=False, batch_size=1)
    vgg(test_dir, 1, train=False, batch_size=1)
        
   
