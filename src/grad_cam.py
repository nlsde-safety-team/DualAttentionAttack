import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import os

class GradCamPP():
    
    hook_a, hook_g = None, None
    
    hook_handles = []
    
    def __init__(self, model, conv_layer, use_cuda=True):
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda()
        
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        
        self._relu = True
        self._score_uesd = True
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))
        
    
    def _hook_a(self, module, input, output):
        self.hook_a = output
        
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def _hook_g(self, module, grad_in, grad_out):
        # print(grad_in[0].shape)
        # print(grad_out[0].shape)
        self.hook_g = grad_out[0]
    
    def _backprop(self, scores, class_idx):
        
        loss = scores[:, class_idx].sum() # .requires_grad_(True)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
    
    def _get_weights(self, class_idx, scores):
        
        self._backprop(scores, class_idx)
        
        grad_2 = self.hook_g.pow(2)
        grad_3 = self.hook_g.pow(3)
        alpha = grad_2 / (1e-13 + 2 * grad_2 + (grad_3 * self.hook_a).sum(axis=(2, 3), keepdims=True))

        # Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(axis=(1, 2))
    
    def __call__(self, input, class_idx):
        # print(input.shape)
        # if self.use_cuda:
        #     input = input.cuda()
        scores = self.model(input)
        pred = F.softmax(scores)[0, class_idx]
        # print(scores)
        weights = self._get_weights(class_idx, scores)
        # print(input.grad)
        # rint(weights)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        
        # print(cam.shape)
        # self.clear_hooks()
        cam_np = cam.data.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_np = cv2.resize(cam_np, input.shape[2:])
        cam_np = cam_np - np.min(cam_np)
        cam_np = cam_np / np.max(cam_np)
        return cam, cam_np, pred

class GradCam():
    
    hook_a, hook_g = None, None
    
    hook_handles = []
    
    def __init__(self, model, conv_layer, use_cuda=True):
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda()
        
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        
        self._relu = True
        self._score_uesd = True
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))
        
    
    def _hook_a(self, module, input, output):
        self.hook_a = output
        
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def _hook_g(self, module, grad_in, grad_out):
        # print(grad_in[0].shape)
        # print(grad_out[0].shape)
        self.hook_g = grad_out[0]
    
    def _backprop(self, scores, class_idx):
        
        loss = scores[:, class_idx].sum() # .requires_grad_(True)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
    
    def _get_weights(self, class_idx, scores):
        
        self._backprop(scores, class_idx)
        
        return self.hook_g.squeeze(0).mean(axis=(1, 2))
    
    def __call__(self, input, class_idx):
        # print(input.shape)
        # if self.use_cuda:
        #     input = input.cuda()
        scores = self.model(input)
        # class_idx = torch.argmax(scores, axis=-1)
        pred = F.softmax(scores)[0, class_idx]
        # print(class_idx, pred)
        # print(scores)
        weights = self._get_weights(class_idx, scores)
        # print(input.grad)
        # rint(weights)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        
        # print(cam.shape)
        # self.clear_hooks()
        cam_np = cam.data.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_np = cv2.resize(cam_np, input.shape[2:])
        cam_np = cam_np - np.min(cam_np)
        cam_np = cam_np / np.max(cam_np)
        return cam, cam_np, pred

class CAM:
    
    def __init__(self):
        model = models.resnet50(pretrained=True)
        self.grad_cam = GradCam(model=model, conv_layer='layer4', use_cuda=True)
        self.log_dir = "./"
        
    def __call__(self, img, index, log_dir, t_index=None):
        self.log_dir = log_dir
        self.t_index = t_index
        img = img / 255
        raw_img = img.data.cpu().numpy()[0].transpose((1, 2, 0))
        input = self.preprocess_image(img)
        target_index = [468,511,609,817,581,751,627]
        if t_index==None:
            ret, mask, pred = self.grad_cam(input, target_index[index % len(target_index)])
        else:
            ret, mask, pred = self.grad_cam(input, t_index)
        # print(img.shape)
        self.show_cam_on_image(raw_img, mask)
        return ret, pred
        
    def preprocess_image(self, img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img
        for i in range(3):
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] - means[i]
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] / stds[i]
        # preprocessed_img = \
        #     np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        # preprocessed_img = torch.from_numpy(preprocessed_img)
        # preprocessed_img.unsqueeze_(0)
        input = preprocessed_img.requires_grad_(True)
        return input


    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = np.float32(heatmap) / 255
        cam_pure = heatmap
        cam_pure = cam_pure / np.max(cam_pure)
        cam = np.float32(img) + heatmap
        cam = cam / np.max(cam)
        if self.t_index==None:
            Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam.jpg'))
            Image.fromarray(np.uint8(255 * mask)).save(os.path.join(self.log_dir, 'cam_b.jpg'))
            
        else:
            Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam_'+str(self.t_index)+'.jpg'))
        Image.fromarray(np.uint8(255 * cam_pure)).save(os.path.join(self.log_dir, 'cam_p.jpg'))
            
        # cv2.imwrite("cam.jpg", np.uint8(255 * cam))


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = models.resnet50(pretrained=True)
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # print(model._modules.items())
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)