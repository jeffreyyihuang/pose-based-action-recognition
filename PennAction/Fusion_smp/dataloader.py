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


class ResNet3D_dataset(Dataset):  
    def __init__(self, dic, root_dir, anno_path, mode, nb_per_stack, transform=None):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.root_img = '/home/ubuntu/data/PennAction/Penn_Action/frames/'
        self.anno_path=anno_path
        self.transform = transform
        self.mode=mode
        self.nb_per_stack = nb_per_stack
        self.img_transform = transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.keys)

    def stack_opf(self, key, index):
        opf_root = '/home/ubuntu/data/PennAction/Penn_Action/flownet2.0/dense_opf/'
        index = int(index)
        out=np.zeros((20,224,224))
        for ii in range(10):
            flowx=Image.open(opf_root + key+'/x'+str(index+ii).zfill(6)+'.jpg')
            flowy=Image.open(opf_root + key+'/y'+str(index+ii).zfill(6)+'.jpg')

            out[2*(ii),:,:] = (flowx).resize([224,224])
            out[2*(ii)+1,:,:] = (flowy).resize([224,224])
            flowx.close()
            flowy.close()
    
        return torch.from_numpy(out).float().div(255)

    def __getitem__(self, idx):
        if self.mode == 'train':
            video, nb_clips = self.keys[idx].split('[@]')
            clips_idx = randint(1,int(nb_clips))
        elif self.mode == 'val':
            video,clips_idx = self.keys[idx].split('[@]')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        data  = stack_joint_position(video, int(clips_idx), self.nb_per_stack, self.root_dir, self.anno_path, self.transform, self.mode)
        
        #index = randint(int(clips_idx),int(clips_idx)+self.nb_per_stack)
        n = video+'/'+ str(clips_idx).zfill(6)+'.jpg'
        image = Image.open(self.root_img+n)
        opf = self.stack_opf(video,clips_idx)
        
        if self.mode == 'train':
            img = (image).resize([224,224])
            img = self.img_transform(img)
            x = (img,opf,data)
            sample = (x,label)
        elif self.mode == 'val':
            img = self.img_transform(image.resize([224,224]))
            x = (img,opf,data)
            sample = (video,x,label)
        else:
            raise ValueError('There are only train and val mode')
        image.close()
        return sample

class ResNet3D_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, nb_per_stack, data_path, dic_path, anno_path):
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.data_path=data_path
        self.anno_path=anno_path
        self.nb_per_stack=nb_per_stack
        #load data dictionary
        with open(dic_path+'/train_video.pickle','rb') as f:
            self.train_video=pickle.load(f)
        f.close()
        with open(dic_path+'/test_video.pickle','rb') as f:
            self.test_video=pickle.load(f)
        f.close()
        with open(dic_path+'frame_count.pickle','rb') as f:
            self.frame_count=pickle.load(f)
        f.close()

    def run(self):
        self.test_video_segment()
        self.train_video_labeling()
        train_loader = self.train()
        val_loader = self.val()
        return train_loader, val_loader
    
    def test_video_segment(self):
        self.dic_test_idx = {}
        for video in self.test_video: # dic[video] = label
            nb_frame = int(self.frame_count[video])-self.nb_per_stack
            if nb_frame <= 0:
                raise ValueError('Invalid nb_per_stack number {} ').format(self.nb_per_stack)
            for clip_idx in range(nb_frame):
                if clip_idx % self.nb_per_stack ==0:
                    key = video + '[@]' + str(clip_idx+1)
                    self.dic_test_idx[key] = self.test_video[video]

    def train_video_labeling(self):
        self.dic_video_train={}
        for video in self.train_video: # dic[video] = label

            nb_clips = self.frame_count[video]-self.nb_per_stack
            if nb_clips <= 0:
                raise ValueError('Invalid nb_per_stack number {} ').format(self.nb_per_stack)
            key = video +'[@]' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video]
                            
    def train(self):
        training_set = ResNet3D_dataset(dic=self.dic_video_train, 
            root_dir=self.data_path,
            anno_path = self.anno_path,
            nb_per_stack=self.nb_per_stack,
            mode='train',
            transform=None 
            )
        print '==> Training data :',len(training_set),' videos'
        print training_set[1][0][0].size(),training_set[1][0][1].size(),training_set[1][0][2].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def val(self):
        validation_set = ResNet3D_dataset(
            dic= self.dic_test_idx, 
            root_dir=self.data_path,
            anno_path = self.anno_path,
            nb_per_stack=self.nb_per_stack,
            mode ='val',
            transform=None 
            )
        print '==> Validation data :',len(validation_set),' clips'
        print validation_set[1][1][0].size(),validation_set[1][1][1].size(),validation_set[1][1][2].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

def stack_joint_position(videoname, clip_idx, nb_per_stack, root_dir, anno_path, transform, mode):
    out=np.zeros((1,nb_per_stack,112,112))
    index=int(clip_idx)
    for i in range(nb_per_stack):
        n = videoname+'/'+ str(index+i).zfill(6)+'.mat'
        mat = scipy.io.loadmat(root_dir + n)['final_score']
        x0,y0,x1,y1=gt_bounding_box(videoname,index+i,anno_path)
        if mode =='train':
            data = crop_and_resize(mat,x0,y0,x1,y1)
            out[:,i,:,:] = data
        elif mode =='val':
            data = crop_and_resize(mat,x0,y0,x1,y1)
            out[:,i,:,:] = data
        else:
            raise ValueError('There are only train and val mode')
            

    return torch.from_numpy(out).float().div(255)


def gt_bounding_box(videoname,idx,anno_path):
    annotation = scipy.io.loadmat(anno_path+videoname+'.mat')
    x0,y0,x1,y1 = annotation['bbox'][idx-1]

    return int(x0),int(y0),int(x1),int(y1)

def crop_and_resize(mat,x0,y0,x1,y1):
    joint_postion = Image.fromarray(mat.sum(axis=2,dtype='uint8'))
    img = joint_postion.crop([x0,y0,x1,y1])
    resize = img.resize([112,112])

    return resize
'''
class GroupRandomFlip():
    def get_params(self):
        self.isflip = random.random()

    def hflip(self,img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def flip(self,img):
        if self.isflip < 0.5:
            return self.hflip(img)
        else:
            return img
'''
class RsizeRandomCrop():
    def __init__(self):
        H = [256,224,192,168]
        W = [256,224,192,168]
        id1 = randint(0,len(H)-1)
        id2 = randint(0,len(W)-1)
    
        self.h_crop = H[id1]
        self.w_crop = W[id2]
        
        self.h0 = randint(0,256-self.h_crop)
        self.w0 = randint(0,256-self.w_crop)
        

    def __call__(self,img):
        img = img.resize([256,256])
        crop = img.crop([self.h0,self.w0,self.h0+self.h_crop,self.w0+self.w_crop])
        resize = crop.resize([224,224])
        return resize   


if __name__ == '__main__':
    data_loader = ResNet3D_DataLoader(BATCH_SIZE=1,num_workers=1,nb_per_stack=16,
                                        dic_path='/home/ubuntu/data/PennAction/Penn_Action/train_test_split/',
                                        data_path='/home/ubuntu/data/PennAction/Penn_Action/heatmap/',
                                        anno_path='/home/ubuntu/data/PennAction/Penn_Action/labels/'
                                        )
    train_loader,val_loader = data_loader.run()
    print type(train_loader),type(val_loader)