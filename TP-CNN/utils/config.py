from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    #hyperparameter
    nb_epochs = 50
    batch_size = 32
    lr = 5e-3
    evaluate = False
    resume = False
    start_epoch = 0

    # data
    dataset = 'PennAction'
    input_type = 'pose'
    use_Bbox = False
    nb_per_stack = 15

    #model
    model = 'resnet18'
    nb_classes = 15
    Fusion = False

    #record
    record_path = 'record'
    dic_path = '/mnt/home/htchen-dev/home/ubuntu/data/PennAction/Penn_Action/train_test_split/'

    #utils
    num_workers = 8

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('--------user config--------')
        pprint(self._state_dict())
        print('------------end------------')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()

# verify function
def main(**kwargs):
    opt._parse(kwargs)

if __name__ == '__main__':
    main(**kwargs)
