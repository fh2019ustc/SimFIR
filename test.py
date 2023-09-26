from models_learn import SimFIR

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import numpy as np
import cv2
import os
from functools import partial
from PIL import Image
import argparse
import warnings
import time

warnings.filterwarnings('ignore')

    
class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.simfir = SimFIR(img_size=256,
                    patch_size=16, embed_dim=256, depth=10, num_heads=8,
                    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    def forward(self, x):
        bm, mask = self.simfir(x)
        mask = mask > 0.5
        bm = bm / 127.5 - 1

        return bm, mask


def reload_rec_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def rec(rec_model_path, distorrted_path, save_path, opt):
    print(torch.__version__)
    
    # distorted images list
    img_list = os.listdir(distorrted_path)
    
    # creat save path for rectified images
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    net = Net(opt)
    print(get_parameter_number(net))

    # reload rec model
    reload_rec_model(net.simfir, rec_model_path)
    
    net.eval()
    
    for img_path in img_list:
        name = img_path.split('.')[-2]
        im_ori = np.array(Image.open(distorrted_path + img_path))[:, :, :3] / 255.
        
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (256, 256))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)

        with torch.no_grad():
            bm_pred, mask_pred = net(im)
            bm_pred = bm_pred.cpu()
            mask_pred = mask_pred.cpu()
            
            bm0 = cv2.resize(bm_pred[0, 0].numpy(), (w, h))  # x flow
            bm1 = cv2.resize(bm_pred[0, 1].numpy(), (w, h))  # y flow
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
            
            mask = cv2.resize(np.float32(mask_pred[0, 0]), (w, h))  # x flow
            mask = torch.from_numpy(mask)  # h * w * 2
            
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
            io.imsave(save_path + name + '_rec' + '.png', ((out[0] * mask * 255).permute(1, 2, 0).numpy()).astype(np.uint8))


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec_model_path', default='./model_pretrained/SimFIR.pth')
    parser.add_argument('--distorrted_path', default='./distorted/')
    parser.add_argument('--save_path', default='./rectified/')
    opt = parser.parse_args()

    rec(rec_model_path=opt.rec_model_path,
        distorrted_path=opt.distorrted_path,
        save_path=opt.save_path,
        opt=opt)
    

if __name__ == "__main__":
    main()
