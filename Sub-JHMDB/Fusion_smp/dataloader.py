import numpy as np
import pickle
from PIL import Image
import time
import shutil
from random import randint
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


# DATA LOADER FOR SUB JHMDB DATASET min nb frame: 15

class Stack_joint_position_dataset(Dataset):  
    def __init__(self, dic, root_dir, anno_path, mode, nb_per_stack, transform=None):
        
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.anno_path=anno_path
        self.transform = transform
        self.mode=mode
        self.nb_per_stack = nb_per_stack
        self.img_transform = transform=transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

    def stack_joint_position(self, key, index):
        out=np.zeros((1,self.nb_per_stack,112,112))
        for ii in range(self.nb_per_stack):
            n = key+'/'+ str(index+ii).zfill(5)+'.mat'
            mat = scipy.io.loadmat(self.root_dir + n)['final_score']
            data = self.gt_bbox_crop_and_resize(key,index+ii,mat)
            out[:,ii,:,:] = data
    
        return torch.from_numpy(out).float().div(255)

    def gt_bbox_crop_and_resize(self,key,idx,mat):
        annotation = scipy.io.loadmat(self.anno_path+key+'/bounding_box.mat')
        h0,h1,w0,w1 = annotation['bbox'][:,idx-1]

        joint_postion = Image.fromarray(mat.sum(axis=2,dtype='uint8'))
        crop_img = joint_postion.crop([w0,h0,w1,h1])
        
        return crop_img.resize([112,112])

    def stack_opf(self, key, index):
        index=int(index)
        annotation = scipy.io.loadmat(self.anno_path+key+'/bounding_box.mat')
        h0,h1,w0,w1 = annotation['bbox'][:,index-1]

        out=np.zeros((2*10,224,224))
        for ii in range(10):
            flowx=Image.open('/home/ubuntu/data/Sub_JHMDB/dense_opf/' + key+'/x'+str(index+ii).zfill(5)+'.jpg')
            flowy=Image.open('/home/ubuntu/data/Sub_JHMDB/dense_opf/' + key+'/y'+str(index+ii).zfill(5)+'.jpg')

            cropx = flowx.crop([w0,h0,w1,h1])
            cropy = flowy.crop([w0,h0,w1,h1])

            out[2*(ii),:,:] =(cropx).resize([224,224])
            out[2*(ii)+1,:,:] =(cropy).resize([224,224])
    
        return torch.from_numpy(out).float().div(255)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.mode == 'train':
            video, nb_clips = self.keys[idx].split('[@]')
            clip_idx = randint(1,int(nb_clips))
        elif self.mode == 'val':
            video,clip_idx = self.keys[idx].split('[@]')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        clip_idx = int(clip_idx)
        #pose
        data = self.stack_joint_position(video,int(clip_idx))
        #rgb
        img_name = video+'/'+ str(clip_idx).zfill(5)+'.png'
        img = Image.open('/home/ubuntu/data/JHMDB/Rename_Images/'+img_name)
        #flow
        opf = self.stack_opf(video,clip_idx)

        x = (self.img_transform(img),data,opf)


        if self.mode == 'train':
            sample = (x,label)
        elif self.mode == 'val':
            sample = (video,x,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample

class Stack_joint_position_DataLoader():
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
        self.test_frame_sampling()
        self.train_video_labeling()
        train_loader = self.train()
        val_loader = self.val()
        return train_loader, val_loader
    
    def test_frame_sampling(self):  # uniformly sample 18 frames and  make a video level consenus
        self.dic_test_idx = {}
        for video in self.test_video: # dic[video] = label
            nb_frame = int(self.frame_count[video])-self.nb_per_stack+1
            for i in range(nb_frame):
                if i % self.nb_per_stack ==0:
                    key = video + '[@]' + str(i+1)
                    #print key
                    self.dic_test_idx[key] = self.test_video[video]

    def train_video_labeling(self):
        self.dic_video_train={}
        for video in self.train_video: # dic[video] = label

            nb_clips = self.frame_count[video]-self.nb_per_stack+1
            if nb_clips <= 0:
                raise ValueError('Invalid nb_per_stack number {} ').format(self.nb_per_stack)
            key = video +'[@]' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video]
                            
    def train(self):
        training_set = Stack_joint_position_dataset(dic=self.dic_video_train, 
            root_dir=self.data_path,
            anno_path = self.anno_path,
            nb_per_stack=self.nb_per_stack,
            mode='train',
            transform=None
            )
        print '==> Training data :',len(training_set),' videos'
        print training_set[1][0][0].size(),training_set[1][0][1].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def val(self):
        validation_set = Stack_joint_position_dataset(
            dic= self.dic_test_idx, 
            root_dir=self.data_path,
            anno_path = self.anno_path,
            nb_per_stack=self.nb_per_stack,
            mode ='val',
            transform=None
            )
        print '==> Validation data :',len(validation_set),' clips'
        print validation_set[1][1][0].size(),validation_set[1][1][1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

if __name__ == '__main__':
    data_loader = Stack_joint_position_DataLoader(BATCH_SIZE=1,num_workers=1,nb_per_stack=10,
                                        dic_path='/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/get_train_test_split/dict/',
                                        data_path='/home/ubuntu/data/JHMDB/pose_estimation/pose_estimation/',
                                        anno_path='/home/ubuntu/data/JHMDB/bounding_box/'
                                        )
    train_loader,val_loader = data_loader.run()
    print type(train_loader),type(val_loader)