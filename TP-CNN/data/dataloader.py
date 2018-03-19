from .PennAction_dataset import PennActionDataset
from .Fusion_dataset import Fusiondataset
from utils.config import opt
from torch.utils.data import  DataLoader as _DataLoader
import pickle

class DataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.BATCH_SIZE=opt.batch_size
        self.num_workers = opt.num_workers
        self.nb_per_stack=opt.nb_per_stack
        #load data dictionary
        with open(opt.dic_path+'/train_video.pickle','rb') as f1:
            self.train_video=pickle.load(f1)
        f1.close()
        with open(opt.dic_path+'/test_video.pickle','rb') as f2:
            self.test_video=pickle.load(f2)
        f2.close()
        with open(opt.dic_path+'/frame_count.pickle','rb') as f3:
            self.frame_count=pickle.load(f3,encoding='latin1')
        f3.close()

    def run(self):
        self.test_frame_sampling()
        self.train_video_labeling()
        train_loader = self.train()
        test_loader = self.val()
        return train_loader, test_loader, self.test_video
    
    def test_frame_sampling(self):  # uniformly sample 18 frames and  make a video level consenus
        self.dic_test_idx = {}
        for video in self.test_video: # dic[video] = label
            nb_frame = int(self.frame_count[video])-self.nb_per_stack-1 # -1 for opf stream
            for i in range(nb_frame):
                if i % self.nb_per_stack ==0:
                    key = video + '[@]' + str(i+1)
                    #print key
                    self.dic_test_idx[key] = self.test_video[video]

    def train_video_labeling(self):
        self.dic_video_train={}
        for video in self.train_video: # dic[video] = label
            nb_clips = self.frame_count[video]-self.nb_per_stack-1 # -1 for opf stream
            if nb_clips <= 0:
                raise ValueError('Invalid nb_per_stack number {} ').format(self.nb_per_stack)
            key = video +'[@]' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video]
                            
    def train(self):
        training_set = Train_Dataset(dic_train=self.dic_video_train, opt=self.opt)
        if self.opt.Fusion:
            print ('==> Training data : %d videos,'%len(training_set), training_set[1][1][0].size(), training_set[1][1][1].size())
        else:
            print ('==> Training data : %d videos,'%len(training_set), training_set[1][1].size())

        train_loader = _DataLoader(
            dataset=training_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers
            )
        return train_loader

    def val(self):
        testing_set = Test_Dataset(dic_test= self.dic_test_idx, opt=self.opt)
        if self.opt.Fusion:
            print ('==> Testing data : %d clips,'%len(testing_set), testing_set[1][1][0].size(), testing_set[1][1][1].size())
        else:
            print ('==> Testing data : %d clips,'%len(testing_set), testing_set[1][1].size())
        test_loader = _DataLoader(
            dataset=testing_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers
            )
        return test_loader

class Train_Dataset:
    def __init__(self, opt, dic_train):
        self.opt = opt

        if opt.Fusion:
            self.db = Fusiondataset(
                dic=dic_train,
                use_Bbox=opt.use_Bbox,
                split='train',
                nb_per_stack=opt.nb_per_stack
            )
        else:
            self.db = PennActionDataset(
                dic= dic_train,
                use_Bbox = opt.use_Bbox,
                split='train',
                input_type = opt.input_type,
                nb_per_stack = opt.nb_per_stack
                    )

    def __getitem__(self, idx):
        return self.db.get_example(idx)

    def __len__(self):
        return len(self.db)

class Test_Dataset:
    def __init__(self, opt, dic_test):
        self.opt = opt

        if opt.Fusion:
            self.db = Fusiondataset(
                dic=dic_test,
                use_Bbox=opt.use_Bbox,
                split='train',
                nb_per_stack=opt.nb_per_stack
            )
        else:
            self.db = PennActionDataset(
                dic= dic_test,
                use_Bbox = opt.use_Bbox,
                split='test',
                input_type = opt.input_type,
                nb_per_stack = opt.nb_per_stack
                    )

    def __getitem__(self, idx):
        return self.db.get_example(idx)

    def __len__(self):
        return len(self.db)

def main(**kwarg):

    # opt config
    opt._parse(kwargs)

    # Data Loader
    data_loader = DLoader(opt)
    train_loader, test_loader, test_video = data_loader.run()
    

if __name__ == '__main__':
    import fire
    fire.Fire(main)
