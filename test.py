
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import os, argparse
import cv2
from Code.lib.model import RFNet
from Code.lib.filter import anisodiff2D
from Code.utils.data import test_dataset
from Code.utils.options import opt

dataset_path = opt.test_path

# set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU {}'.format(opt.gpu_id))
ano_fil = anisodiff2D(num_iter=10, delta_t=1/7, kappa=50, option=2)
# load the model
model = RFNet()
model.cuda()
ch = torch.load('./Ours_best.pth')
print(ch.keys())
model.load_state_dict(ch)

model.eval()

# test
test_datasets = ['ReDWeb-S','come-test-e','come-test-h','NJU2K', 'NLPR', 'SIP', 'STERE','DES']

for dataset in test_datasets:
    save_path = '../RFNet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root  = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root,opt.trainsize)
    for i in range(test_loader.size):
        image,gt,depth, name, image_for_post = test_loader.load_data()

        image = image.cuda()
        depth   = depth.cuda()

        res = model(image,depth)
        #res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, res * 255)
    print('Test Done!')
