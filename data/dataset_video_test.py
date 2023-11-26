import torch.utils.data as data
from pathlib import Path
import random
import torch
import data.augmentations as augmentations
import utils.utils_video as utils_video
import numpy as np


class VideoTest_NR_Dataset(data.Dataset):
   
    def __init__(self, opt):
        super(VideoTest_NR_Dataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(self.lq_root / 'meta_info.txt', 'r') as fin:
            self.keys = np.unique([line.split('/')[1] for line in fin])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [i for i in range(opt['num_frame'])]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        self.mirror_sequence = opt['mirror_sequence']
        self.pad_sequence = opt['pad_sequence']
        self.dataset_name = opt['name']

    def __getitem__(self, index):
        #print('GET_ITEM')
        
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        #gt_size = self.opt['gt_size']
        key = self.keys[index]

        # get the neighboring LQ and  GT frames
        img_lqs = []
        #img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{self.dataset_name}/{key}/im{neighbor:05d}'
                #img_gt_path = f'GT/{key}/im{neighbor:05d}'
            else:
                img_lq_path = self.lq_root / key / f'im{neighbor:05d}.png'
                #img_gt_path = self.gt_root / key / f'im{neighbor:05d}.png'
            # LQ
            #print(img_lq_path)
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)

            # GT
            #img_bytes = self.file_client.get(img_gt_path, 'gt')
            #img_gt = utils_video.imfrombytes(img_bytes, float32=True)

            img_lqs.append(img_lq)
            #img_gts.append(img_gt)
        
        # randomly crop
        #img_gts, img_lqs = augmentations.paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)
        
        # video augmentation - flip, rotate
        #img_lqs.extend(img_gts)
        img_results = augmentations.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        img_lqs = torch.stack(img_results[:7], dim=0)
        #img_gts = torch.stack(img_results[7:], dim=0)
        #img_gts, img_lqs = augmentations.cutblur(img_gts, img_lqs)


        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'key': key}

    def __len__(self):
        return len(self.keys)
    

class Video_FR_Dataset(data.Dataset):
   
    def __init__(self, opt):
        super(Video_FR_Dataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.keys = {}
        with open(self.gt_root / 'meta_info.txt', 'r') as fin:
            self.keys['gt'] = np.unique([line.split('/im')[0] for line in fin])
        with open(self.lq_root / 'meta_info.txt', 'r') as fin:
            self.keys['lq'] = np.unique([line.split('/im')[0] for line in fin])
        assert len(self.keys['lq']) == len(self.keys['gt']), "LQ and GT meta_info files contain different number of lines"

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
        
        self.opt['name_zero_pad'] = 5 if self.opt['name_zero_pad'] is None else self.opt['name_zero_pad']
        self.opt['num_frame'] = 7 if self.opt['num_frame'] is None else self.opt['num_frame']

        # indices of input images
        self.neighbor_list = [i for i in range(*opt['frames_interval_list'])]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        self.mirror_sequence = opt['mirror_sequence']
        self.pad_sequence = opt['pad_sequence']
        self.random_crop = opt['random_crop']
        self.cutblur = opt['cutblur']

    def __getitem__(self, index):
        #print('GET_ITEM')
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        key = [self.keys['lq'][index], self.keys['gt'][index]]

        # get the neighboring LQ and  GT frames
        img_lqs = []
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{key[0]}/im{str(neighbor).zfill(self.opt["name_zero_pad"])}'
                img_gt_path = f'{key[1]}/im{str(neighbor).zfill(self.opt["name_zero_pad"])}'
            else:
                img_lq_path = self.lq_root / key / f'im{str(neighbor).zfill(self.opt["name_zero_pad"])}.png'
                img_gt_path = self.gt_root / key / f'im{str(neighbor).zfill(self.opt["name_zero_pad"])}.png'
            # LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)

            # GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)

            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        # randomly crop
        if self.random_crop:
            scale = self.opt['scale']
            gt_size = self.opt['gt_size']
            img_gts, img_lqs = augmentations.paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)
        elif not self.opt['gt_size'] is None:
            img_gts, img_lqs = augmentations.paired_center_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)
        
        # video augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augmentations.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_lqs)
        img_lqs = torch.stack(img_results[:self.opt["num_frame"]], dim=0)
        img_gts = torch.stack(img_results[self.opt["num_frame"]:], dim=0)
        
        if self.cutblur:
            img_gts, img_lqs = augmentations.cutblur(img_gts, img_lqs)
        if self.mirror_sequence:  # mirror the sequence: 7 frames to 14 frames
            img_lqs = torch.cat([img_lqs, img_lqs.flip(0)], dim=0)
            img_gts = torch.cat([img_gts, img_gts.flip(0)], dim=0)
        elif self.pad_sequence:  # pad the sequence: 7 frames to 8 frames
            img_lqs = torch.cat([img_lqs, img_lqs[-1:,...]], dim=0)
            img_gts = torch.cat([img_gts, img_gts[-1:,...]], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys['lq'])