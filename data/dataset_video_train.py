import torch.utils.data as data
from pathlib import Path
import random
import torch
import data.augmentations as augmentations
import utils.utils_video as utils_video
import numpy as np
from torchvision import transforms


class VideoRecurrentTrainRealVSRDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoRecurrentTrainRealVSRDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(self.gt_root / 'meta_info.txt', 'r') as fin:
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

    def __getitem__(self, index):
        #print('GET_ITEM')
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]

        # get the neighboring LQ and  GT frames
        img_lqs = []
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'LQ/{key}/im{neighbor:05d}'
                img_gt_path = f'GT/{key}/im{neighbor:05d}'
            else:
                img_lq_path = self.lq_root / key / f'im{neighbor:05d}.png'
                img_gt_path = self.gt_root / key / f'im{neighbor:05d}.png'
            # LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)

            # GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)

            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = augmentations.paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)
        
        # video augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augmentations.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        img_lqs = torch.stack(img_results[:7], dim=0)
        img_gts = torch.stack(img_results[7:], dim=0)
        #img_gts, img_lqs = augmentations.cutblur(img_gts, img_lqs)

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
        return len(self.keys)
    


class VideoRecurrentTrainVimeoDataset(data.Dataset):
    """Vimeo90K dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, separated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VideoRecurrentTrainVimeoDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.temporal_scale = opt.get('temporal_scale', 1)

        #with open(opt['meta_info_file'], 'r') as fin:
        #    self.keys = [line.split(' ')[0] for line in fin]
        with open(self.gt_root / 'meta_info.txt', 'r') as fin:
            self.keys = [line.split('/im')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])][::self.temporal_scale]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        print(f'Random reverse is {self.random_reverse}.')

        self.mirror_sequence = opt.get('mirror_sequence', False)
        self.pad_sequence = opt.get('pad_sequence', False)


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring LQ and  GT frames
        img_lqs = []
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            # LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            # GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)

            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        # randomly crop
        if self.opt['random_crop']:
            img_gts, img_lqs = augmentations.paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)
        elif self.opt['gt_size']:
            img_gts, img_lqs = augmentations.paired_center_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)
        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augmentations.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        img_lqs = torch.stack(img_results[:7], dim=0)
        img_gts = torch.stack(img_results[7:], dim=0)

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
        return len(self.keys)


