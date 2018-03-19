import numpy as np
import pickle
from PIL import Image
import time
import shutil
from random import randint
import random
import argparse
import scipy.io

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt


class Fusiondataset(Dataset):  
    def __init__(self, dic, use_Bbox, split, nb_per_stack=3):
        #Generate a 16 Frame clip
        self.keys=list(dic.keys())
        self.values=list(dic.values())
        self.use_Bbox=use_Bbox
        self.split=split
        self.nb_per_stack = nb_per_stack


    def __len__(self):
        return len(self.keys)

    def get_example(self, idx):
        if self.split == 'train':
            video, nb_clips = self.keys[idx].split('[@]')
            clips_idx = randint(1,int(nb_clips))
        elif self.split == 'val':
            video,clips_idx = self.keys[idx].split('[@]')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1

        # Get the rgb and opf data for model input
        rgb = self.read_image(video, clips_idx)
        opf = self.stack_opf(video, clips_idx)
        
        data = (rgb,opf)
        sample = (video, data, label)
        return sample

    def stack_opf(self, key, index):
        data_dir = '/home/ubuntu/data/PennAction/Penn_Action/flownet2.0/dense_opf/'
        out=np.zeros((2*self.nb_per_stack,224,224))
        for ii in range(self.nb_per_stack):
            flowx=Image.open(data_dir + key+'/x'+str(index+ii).zfill(6)+'.jpg')
            flowy=Image.open(data_dir + key+'/y'+str(index+ii).zfill(6)+'.jpg')

            if self.use_Bbox:
                flowx = self.crop_gt_Bbox(flowx, key, index)
                flowy = self.crop_gt_Bbox(flowy, key, index)

            out[2*(ii),:,:] = flowx.resize([224,224])
            out[2*(ii)+1,:,:] = flowy.resize([224,224])
    
        return torch.from_numpy(out).float().div(255)

    def read_image(self, key, index):
        data_dir = '/home/ubuntu/data/PennAction/Penn_Action/frames/'
        n = key+'/'+ str(index).zfill(6)+'.jpg'
        img = Image.open(data_dir+n)

        Rcrop=transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                ])

        transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

        #if self.use_Bbox:
        #    img = (self.crop_gt_Bbox(img,key,index)).resize([224,224])
        #else:
        img = Rcrop(img)

        return transform(img)

    def crop_gt_Bbox(self, img, key, index):
        anno_path = '/home/ubuntu/data/PennAction/Penn_Action/labels/'
        annotation = scipy.io.loadmat(anno_path+key+'.mat')
        x0,y0,x1,y1 = annotation['bbox'][index-1]
        crop_img = img.crop([x0,y0,x1,y1])

        return crop_img
    