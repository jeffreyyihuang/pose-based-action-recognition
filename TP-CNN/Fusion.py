from tqdm import tqdm
import shutil
import torch.nn as nn
import torch
import time
import datetime
import os
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from utils.config import opt
from data.dataloader import DataLoader as DLoader
import model.resnet_2d as models_2d
import model.resnet_3d as models_3d
#import model.resnet3d_conv1_10 as model_dev
from utils.extension import *


def main(**kwargs):

    # opt config
    opt._parse(kwargs)

    # Data Loader
    data_loader = DLoader(opt)
    train_loader, test_loader, test_video = data_loader.run()

    # Train my model
    model = Resnet2D(opt, train_loader, test_loader, test_video)
    model.run()




class Resnet2D():

    def __init__(self, opt, train_loader, test_loader, test_video):
        self.opt = opt
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_video = test_video
        self.best_prec1 = 0

        # Check for rgb input
        if opt.input_type == 'rgb' and opt.nb_per_stack != 1:
            raise ValueError('rgb data only support nb_per_stack = 1')

        # tensorboard name
        current_time = datetime.datetime.now().strftime('%b%d_%H')
        log_dir = os.path.join(
            'runs', current_time+'_SM_Fusion_L_'+str(opt.nb_per_stack))

        # Bounding Box
        if opt.use_Bbox:
            log_dir = log_dir+'_Bbox'
        self.tensorboard = SummaryWriter(log_dir=log_dir)

    def build_model(self):
        self.model = Fusion_net(
            RGB_weight='/home/ubuntu/cvlab/pytorch/icme2018/record/rgb_L1/model_best.pth.tar',
            OPF_weight='/home/ubuntu/cvlab/pytorch/icme2018/record/opf_L15_Bbox/model_best.pth.tar',
            opt=self.opt
        )
        # To Cuda
        self.model = self.model.cuda()

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), self.opt.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=1, verbose=True)
        print ('==> Build %s model and setup loss function and optimizer' %
               self.opt.model)

    def run(self):
        self.build_model()
        #self.resume_and_evaluate()

        cudnn.benchmark = True
        for self.epoch in range(self.opt.start_epoch, self.opt.nb_epochs):
            # Train
            self.train_1epoch()

            # Test
            prec1, val_loss = self.validate_1epoch()
            self.scheduler.step(val_loss)

            # save training model and video level prediction
            save_folder = self.opt.record_path+'/' + \
                '_SM_Fusion_L_'+str(opt.nb_per_stack)

            # mkdir for record_path
            if not os.path.isdir(self.opt.record_path):
                os.mkdir(self.opt.record_path)
            # mkdir for save folder
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            # Save the current model and the best model
            is_best = prec1 > self.best_prec1
            if is_best:
                self.best_prec1 = prec1
                with open(save_folder+'/video_preds.pickle', 'wb') as f:
                    pickle.dump(self.dic_video_level_preds, f)
                f.close()

            # Save model and hyperparameter
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer': self.optimizer.state_dict(),
                'config': self.opt._state_dict()
            }, is_best, folder=save_folder)

    def train_1epoch(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()
        end = time.time()

        # tqdm display
        des = 'Epoch:[%d/%d][training stage]' % (
            self.epoch, self.opt.nb_epochs)
        progress = tqdm(self.train_loader, ascii=True, desc=des)

        # mini-batch training
        for i, (_key, data, label) in enumerate(progress):
            # measure data loading time
            data_time.update(time.time() - end)

            # To cuda()
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # Tensor to Variable
            R = Variable(data[0]).cuda()
            O = Variable(data[1]).cuda()
            input_var = (R,O)

            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], label.size(0))
            top1.update(prec1[0], label.size(0))
            top5.update(prec5[0], label.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # tqdm visuallization
            info = {
                'Prec@1': display_format(top1.avg),
                'Loss': display_format(losses.avg),
                'Data Time': display_format(data_time.avg)
            }
            progress.set_postfix(info)

        # tensorboard utils
        tb_info = {
            'Batch Time': batch_time.avg,
            'Data Time': data_time.avg,
            'Loss': losses.avg,
            'Prec@1': top1.avg,
            'Prec@5': top5.avg,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        for k, v in tb_info.items():
            self.tensorboard.add_scalar('train/'+k, v, self.epoch)

    def validate_1epoch(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        end = time.time()

        # tqdm display
        des = 'Epoch:[%d/%d][testing stage ]' % (
            self.epoch, self.opt.nb_epochs)
        progress = tqdm(self.test_loader, ascii=True, desc=des)
        # mini-batch training
        for i, (keys, data, label) in enumerate(progress):
            # TO cuda()
            label = label.cuda(async=True)
            label_var = Variable(label).cuda()

            # Tensor to Variable
            R = Variable(data[0]).cuda()
            O = Variable(data[1]).cuda()
            input_var = (R,O)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, label_var)
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))

            # measure loss
            losses.update(loss.data[0], label.size(0))
            top1.update(prec1[0], label.size(0))
            top5.update(prec5[0], label.size(0))

            # tqdm
            info = {
                'Prec@1': display_format(top1.avg),
                'Loss': display_format(losses.avg),
                'Batch Time': display_format(batch_time.avg),
            }
            progress.set_postfix(info)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j].split('/', 1)[0]
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j, :]
                else:
                    self.dic_video_level_preds[videoName] += preds[j, :]

        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
        #print type(video_loss)

        # Tensorboard visuallization
        info = {
            'Batch Time': batch_time.avg,
            'Video Loss': video_loss,
            'Video Prec@1': video_top1,
            'Video Prec@5': video_top5
        }

        for k, v in info.items():
            self.tensorboard.add_scalar('test/'+k, v, self.epoch)

        return video_top1, video_loss

    def frame2_video_level_accuracy(self):
        correct = 0
        video_level_preds = np.zeros(
            (len(self.dic_video_level_preds), self.opt.nb_classes))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii = 0
        for key in sorted(self.dic_video_level_preds.keys()):
            name = key

            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name])-1

            video_level_preds[ii, :] = preds
            video_level_labels[ii] = label
            ii += 1
            if np.argmax(preds) == (label):
                correct += 1

        # top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        loss = self.criterion(Variable(video_level_preds).cuda(),
                              Variable(video_level_labels).cuda())

        top1, top5 = accuracy(
            video_level_preds, video_level_labels, topk=(1, 5))

        top1 = float(top1.numpy())
        top5 = float(top5.numpy())

        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1, top5, loss.data.cpu().numpy()

class Fusion_net(nn.Module):
    def __init__(self, RGB_weight, OPF_weight, opt):
        super(Fusion_net, self).__init__()
        self.RGBnet = self.load_weight(
            models_2d.resnet50(
                pretrained=True,
                channel=3,
                nb_classes=opt.nb_classes,
                extract_feature = True
                ), 
            RGB_weight
            )
        self.OPFnet = self.load_weight(
            models_2d.resnet50(
                pretrained=True,
                channel=30,
                nb_classes=opt.nb_classes,
                extract_feature = True
                ), 
            OPF_weight
            )
        self.fusion_conv = nn.Sequential(
                nn.Conv2d(4096,2048,kernel_size=1, stride=1, bias=False),
                nn.Conv2d(2048,1024,kernel_size=1, stride=1, bias=False),
                nn.Conv2d(1024,512,kernel_size=1, stride=1, bias=False),
                nn.Conv2d(512,256,kernel_size=1, stride=1, bias=False),
            )
        self.fusion_Linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256,1024),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024,15)
        )

    def load_weight(self, model, weight_path):
        checkpoint = torch.load(weight_path)
        print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(weight_path, checkpoint['epoch'], checkpoint['best_prec1']))
        model_dict = checkpoint['state_dict']
        model.load_state_dict(model_dict)

        return model

    def forward(self, x):
        rx = self.RGBnet(x[0])
        ox = self.OPFnet(x[1])

        #print rx.size(),ox.size()

        
        in_ = [rx,ox]  # merge the input of two stream
        x = torch.cat(in_, 1)
        x = self.fusion_conv(x)
        x = x.view(x.size(0),-1)
        x = self.fusion_Linear(x)

        return x

if __name__ == '__main__':
    import fire

    fire.Fire(main)
