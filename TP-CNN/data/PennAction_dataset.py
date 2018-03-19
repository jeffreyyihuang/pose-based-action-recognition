from utils.config import opt
from random import randint
from PIL import Image
import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

class PennActionDataset(Dataset):
    def __init__(self, dic, use_Bbox, split, input_type, nb_per_stack=3):
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.use_Bbox = use_Bbox
        self.split = split
        self.nb_per_stack = nb_per_stack
        self.input_type = input_type

        self.input_type_zoo = {
            'pose': 'stack_joint_position',
            'opf': 'stack_opf',
            'rgb': 'read_image',
            '3d_pose': 'stack_joint_position_3d'
        }

        if input_type not in self.input_type_zoo:
            raise ValueError('These mode only support the input type of [ pose, opf, rgb, 3d_pose ]')

    def get_example(self, i):
        get_fn = getattr(self,self.input_type_zoo[self.input_type])
        # print (type(self.values))
        label = int(self.values[i])-1

        if self.split == 'train':
            videoname, nb_clips = self.keys[i].split('[@]')
            clip_idx = randint(1, int(nb_clips))            
            item = get_fn(videoname, int(clip_idx))
            return (item,label)

        elif self.split == 'test':
            videoname, clip_idx = self.keys[i].split('[@]')
            item = get_fn(videoname, int(clip_idx))
            return (videoname, item, label)

        else:
            raise ValueError('There are only train and test split')
        
    
    def stack_joint_position(self, key, index):
        data_dir = '/home/ubuntu/data/PennAction/Penn_Action/heatmap/'
        out=np.zeros((self.nb_per_stack,224,224))
        for ii in range(self.nb_per_stack):
            n = key+'/'+ str(index+ii).zfill(6)+'.mat'
            mat = scipy.io.loadmat(data_dir + n)['final_score']
            joint_postion = Image.fromarray(mat.sum(axis=2,dtype='uint8'))
            if self.use_Bbox:
                data = self.crop_gt_Bbox(joint_postion,key,index+ii)
            else:
                data = joint_postion

            out[ii,:,:] = data.resize([224,224])
    
        return torch.from_numpy(out).float().div(255)

    def stack_joint_position_3d(self, key, index):
        data_dir = '/home/ubuntu/data/PennAction/Penn_Action/heatmap/'
        out=np.zeros((1,self.nb_per_stack,112,112))
        for ii in range(self.nb_per_stack):
            n = key+'/'+ str(index+ii).zfill(6)+'.mat'
            mat = scipy.io.loadmat(data_dir + n)['final_score']
            joint_postion = Image.fromarray(mat.sum(axis=2,dtype='uint8'))
            if self.use_Bbox:
                data = self.crop_gt_Bbox(joint_postion,key,index+ii)
            else:
                data = joint_postion

            out[:,ii,:,:] = data.resize([112,112])
    
        return torch.from_numpy(out).float().div(255)
    
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
                transforms.Resize(256),
                transforms.RandomCrop(224),
                ])

        transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

        if self.use_Bbox:
            img = (self.crop_gt_Bbox(img,key,index)).resize([224,224])
        else:
            img = Rcrop(img)

        return transform(img)

    def crop_gt_Bbox(self, img, key, index):
        anno_path = '/home/ubuntu/data/PennAction/Penn_Action/labels/'
        annotation = scipy.io.loadmat(anno_path+key+'.mat')
        x0,y0,x1,y1 = annotation['bbox'][index-1]
        crop_img = img.crop([x0,y0,x1,y1])

        return crop_img
            
    def __len__(self):
        return len(self.keys)



