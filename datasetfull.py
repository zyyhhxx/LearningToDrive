import os
from PIL import Image
import pandas as pd
import numpy as np
from random import shuffle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler


class SubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class Drive360Loader(DataLoader):

    def __init__(self, config, phase):

        self.drive360 = Drive360(config, phase)
        batch_size = config['data_loader'][phase]['batch_size']
        sampler = SubsetSampler(self.drive360.indices)
        num_workers = config['data_loader'][phase]['num_workers']

        super().__init__(dataset=self.drive360,
                         batch_size=batch_size,
                         sampler=sampler,
                         num_workers=num_workers
                         )

class Drive360(object):
    ## takes a config json object that specifies training parameters and a
    ## phase (string) to specifiy either 'train', 'test', 'validation'
    def __init__(self, config, phase):
        self.config = config
        self.data_dir = config['data_loader'][phase]['data_dir']
        self.csv_name = config['data_loader'][phase]['csv_name']
        self.shuffle = config['data_loader'][phase]['shuffle']
        self.history_number = config['data_loader']['historic']['number']
        self.history_frequency = config['data_loader']['historic']['frequency']
        self.normalize_targets = config['target']['normalize']
        self.target_mean = {}
        target_mean = config['target']['mean']
        for k, v in target_mean.items():
            self.target_mean[k] = np.asarray(v, dtype=np.float32)
        self.target_std = {}
        target_std = config['target']['std']
        for k, v in target_std.items():
            self.target_std[k] = np.asarray(v, dtype=np.float32)

        self.front = self.config['front']
        self.tomtom = config['tomtom']
        self.right_left = config['multi_camera']['right_left']
        self.rear = config['multi_camera']['rear']
        self.here_number = self.config['here']['number']
        self.here_frequency = self.config['here']['frequency']
        self.truncate_test = config['truncate']

        #### reading in dataframe from csv #####
        self.dataframe = pd.read_csv(os.path.join(self.data_dir, self.csv_name),
                                     dtype={'cameraFront': object,
                                            'cameraRear': object,
                                            'cameraRight': object,
                                            'cameraLeft': object,
                                            'tomtom': object,
                                            'hereMmLatitude': np.float32,
                                            'hereMmLongitude': np.float32,
                                            'hereSpeedLimit': np.float32,
                                            'hereSpeedLimit_2': np.float32,
                                            'hereFreeFlowSpeed': np.float32,
                                            'hereSignal': np.float32,
                                            'hereYield': np.float32,
                                            'herePedestrian': np.float32,
                                            'hereIntersection': np.float32,
                                            'hereMmIntersection': np.float32,
                                            'hereSegmentExitHeading': np.float32,
                                            'hereSegmentEntryHeading': np.float32,
                                            'hereSegmentOthersHeading': object,
                                            'hereCurvature': np.float32,
                                            'hereCurrentHeading': np.float32,
                                            'here1mHeading': np.float32,
                                            'here5mHeading': np.float32,
                                            'here10mHeading': np.float32,
                                            'here20mHeading': np.float32,
                                            'here50mHeading': np.float32,
                                            'hereTurnNumber': np.float32,
                                            'canSpeed': np.float32,
                                            'canSteering': np.float32
                                            })
        #### scaling curvature values to within approximately -1.0 to 1.0
        self.dataframe['hereCurvature'] = self.dataframe['hereCurvature'].values * 20.0

        #### filtering and fixing here data #####
        for k in ['hereSignal', 'hereYield', 'herePedestrian', 'hereIntersection', 'hereMmIntersection']:
            self.dataframe[k].fillna(-100.0, inplace=True)
            self.dataframe.loc[self.dataframe[k] < 0.0, k] = -100.0
            self.dataframe.loc[self.dataframe[k] > 100.0, k] = -100.0
            self.dataframe[k] = self.dataframe[k] / 100.0

        for k in ['hereSpeedLimit', 'hereFreeFlowSpeed', 'hereCurrentHeading',
                  'here1mHeading', 'here5mHeading', 'here10mHeading',
                  'here20mHeading', 'here50mHeading',
                  'hereSegmentEntryHeading', 'hereSegmentExitHeading']:
            self.dataframe[k] = self.dataframe[k].ffill().bfill()
            if 'Heading' in k:
                self.dataframe[k] = self.dataframe[k] / 180.0
            else:
                self.dataframe[k] = self.dataframe[k] / 120.0

        # Here we calculate the temporal offset for the starting indices of each chapter. As we cannot cross chapter
        # boundaries but would still like to obtain a temporal sequence of images, we cannot start at index 0 of each chapter
        # but rather at some index i such that the i-max_temporal_history = 0
        # To explain see the diagram below:
        #
        #             chapter 1    chapter 2     chapter 3
        #           |....-*****| |....-*****| |....-*****|
        # indices:   0123456789   0123456789   0123456789
        #
        # where . are ommitted indices and - is the index. This allows using [....] as temporal input.
        #
        # Thus the first sample will consist of images:     [....-]
        # Thus the second sample will consist of images:    [...-*]
        # Thus the third sample will consist of images:     [..-**]
        # Thus the fourth sample will consist of images:    [.-***]
        # Thus the fifth sample will consist of images:     [-****]
        # Thus the sixth sample will consist of images:     [*****]

        self.sequence_length = self.history_number*self.history_frequency
        self.here_sequence_length = self.here_frequency*self.here_number
        max_temporal_history = max(self.sequence_length, self.here_sequence_length)
        self.indices = self.dataframe.groupby('chapter').apply(lambda x: x.iloc[max_temporal_history:]).index.droplevel(level=0).tolist()

        #### phase specific manipulation #####
        if phase == 'train':
            self.dataframe['canSteering'] = np.clip(self.dataframe['canSteering'], a_max=360, a_min=-360)

            ##### If you want to use binning on angle #####
            ## START ##
            # self.dataframe['bin_canSteering'] = pd.cut(self.dataframe['canSteering'],
            #                                            bins=[-360, -20, 20, 360],
            #                                            labels=['left', 'straight', 'right'])
            # gp = self.dataframe.groupby('bin_canSteering')
            # min_group = min(gp.apply(lambda x: len(x)))
            # bin_indices = gp.apply(lambda x: x.sample(n=min_group)).index.droplevel(level=0).tolist()
            # self.indices = list(set(self.indices) & set(bin_indices))
            ## END ##

        elif phase == 'validation':
            self.dataframe['canSteering'] = np.clip(self.dataframe['canSteering'], a_max=360, a_min=-360)

        elif phase == 'test':
            # IMPORTANT: for the test phase indices will start 10s (100 samples) into each chapter
            # this is to allow challenge participants to experiment with different temporal settings of data input.
            # If challenge participants have a greater temporal length than 10s for each training sample, then they
            # must write a custom function here.

            self.indices = self.dataframe.groupby('chapter').apply(
                lambda x: x.iloc[self.truncate_test:]).index.droplevel(
                level=0).tolist()
            if 'canSteering' not in self.dataframe.columns:
                self.dataframe['canSteering'] = [0.0 for _ in range(len(self.dataframe))]
            if 'canSpeed' not in self.dataframe.columns:
                self.dataframe['canSpeed'] = [0.0 for _ in range(len(self.dataframe))]


        if self.normalize_targets and not phase == 'test':
            self.dataframe['canSteering'] = (self.dataframe['canSteering'].values -
                                            self.target_mean['canSteering']) / self.target_std['canSteering']
            self.dataframe['canSpeed'] = (self.dataframe['canSpeed'].values -
                                            self.target_mean['canSpeed']) / self.target_std['canSpeed']

        if self.shuffle:
            shuffle(self.indices)



        print('Phase:', phase, '# of data:', len(self.indices))

        front_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'validation': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ])}
        sides_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'validation': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ]),
            'test': transforms.Compose([
                transforms.Resize((320, 180)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config['image']['norm']['mean'],
                                     std=config['image']['norm']['std'])
            ])}
        tomtom_transforms = {
            'train': transforms.Compose([
                transforms.Resize((375, 250)),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6841898743159287, 0.6942467588984359, 0.63074608385988107],
                                     std=[0.054145950202715391, 0.056086449019928965, 0.080349866693504177])
            ]),
            'validation': transforms.Compose([
                transforms.Resize((375, 250)),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6841898743159287, 0.6942467588984359, 0.63074608385988107],
                                     std=[0.054145950202715391, 0.056086449019928965, 0.080349866693504177])
            ]),
            'test': transforms.Compose([
                transforms.Resize((375, 250)),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6841898743159287, 0.6942467588984359, 0.63074608385988107],
                                     std=[0.054145950202715391, 0.056086449019928965, 0.080349866693504177])
            ])}

        self.imageFront_transform = front_transforms[phase]
        self.imageSides_transform = sides_transforms[phase]
        self.tomtom_transform = tomtom_transforms[phase]


    def __getitem__(self, index):
        inputs = {}
        labels = {}
        end = index - self.sequence_length
        skip = int(-1 * self.history_frequency)
        rows = self.dataframe.iloc[index:end:skip].reset_index(drop=True, inplace=False)

        if self.front:
            inputs['cameraFront'] = {}
            for row_idx, (_, row) in enumerate(rows.iterrows()):
                inputs['cameraFront'][row_idx] = (self.imageFront_transform(Image.open(self.data_dir + row['cameraFront'])))

        if self.tomtom:
            inputs['tomtom'] = self.tomtom_transform(Image.open(self.data_dir + rows['tomtom'].iloc[0]))
        if self.right_left:
            inputs['cameraRight'] = self.imageSides_transform(Image.open(self.data_dir + rows['cameraRight'].iloc[0]))
            inputs['cameraLeft'] = self.imageSides_transform(Image.open(self.data_dir + rows['cameraLeft'].iloc[0]))
        if self.rear:
            inputs['cameraRear'] = self.imageSides_transform(Image.open(self.data_dir + rows['cameraRear'].iloc[0]))

        ##### getting here data #####
        here_end = index - self.here_sequence_length
        here_skip = int(-1 * self.here_frequency)
        here_rows = self.dataframe.iloc[index:here_end:here_skip].reset_index(drop=True, inplace=False)
        #### Road Feature Information #####
        inputs['hereSpeedLimit'] = here_rows['hereSpeedLimit'].values
        inputs['hereFreeFlowSpeed'] = here_rows['hereFreeFlowSpeed'].values
        inputs['hereSignal'] = here_rows['hereSignal'].values
        inputs['hereYield'] = here_rows['hereYield'].values
        inputs['herePedestrian'] = here_rows['herePedestrian'].values
        inputs['hereCurvature'] = here_rows['hereCurvature'].values
        inputs['hereMmIntersection'] = here_rows['hereMmIntersection'].values
        inputs['hereIntersection'] = here_rows['hereIntersection'].values
        inputs['hereTurnNumber'] = here_rows['hereTurnNumber'].values
        inputs['hereSegmentEntryHeading'] = here_rows['hereSegmentEntryHeading'].values
        inputs['hereSegmentExitHeading'] = here_rows['hereSegmentExitHeading'].values
        #### Routing Information #####
        inputs['here1mHeading'] = here_rows['here1mHeading'].values
        inputs['here5mHeading'] = here_rows['here5mHeading'].values
        inputs['here10mHeading'] = here_rows['here10mHeading'].values
        inputs['here20mHeading'] = here_rows['here20mHeading'].values
        inputs['here50mHeading'] = here_rows['here50mHeading'].values

        labels['canSteering'] = self.dataframe['canSteering'].iloc[index]
        labels['canSpeed'] = self.dataframe['canSpeed'].iloc[index]

        return inputs, labels

