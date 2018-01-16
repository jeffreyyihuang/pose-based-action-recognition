from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
pose_preds = '/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/REVICE_CODE/3d_bbox_SJP/old_L15/rgb_video_preds.pickle'
rgb_preds = '/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/REVICE_CODE/rgb_stream/record/rgb_video_preds.pickle'
opf_preds = '/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/REVICE_CODE/opf+bbox/record/rgb_video_preds.pickle'

with open(pose_preds,'rb') as f:
    pose =pickle.load(f)
f.close()
with open(rgb_preds,'rb') as f:
    rgb =pickle.load(f)
f.close()
with open(opf_preds,'rb') as f:
    opf =pickle.load(f)
f.close()

with open('/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/get_train_test_split/dict/test_video.pickle','rb') as f:
    dic_video_label = pickle.load(f)
f.close()

video_level_preds = np.zeros((len(pose.keys()),12))
video_level_labels = np.zeros(len(pose.keys()))
correct=0
ii=0
for name in sorted(pose.keys()):

    p = pose[name]
    r = rgb[name]
    o = opf[name]
    #preds=p+r
    label = int(dic_video_label[name])-1
            
    video_level_preds[ii,:] = (p*4+r+o)
    video_level_labels[ii] = label
    ii+=1         
    if np.argmax(p+r) == (label):
        correct+=1

video_level_labels = torch.from_numpy(video_level_labels).long()
video_level_preds = torch.from_numpy(video_level_preds).float()
     
top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))     
                            
print top1,top5
