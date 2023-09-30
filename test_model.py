import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import utils_test
from utils import utils_option

from data.dataset_video_test import VideoTrain_FR_Dataset
from data.dataset_video_test import VideoTrain_NR_Dataset
from models.select_network import select_net

import time
import os
import json
import re
import yaml

def main():

    # ----------------------------------------
    # options
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option yaml file.')

    args = parser.parse_args()
    opt_path = args.opt
    #opt = utils_option.parse(args.opt, is_train=False)
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    with open(opt_path, "r") as file:
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        opt = yaml.load(file, Loader=loader)
    opt = utils_option.dict_to_nonedict(opt)
    
    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = int(time.time())
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # ----------------------------------------
    # model (netG)
    # ----------------------------------------
    model = select_net(opt)
    model_path = opt['path']['pretrained_netG']
    
    if os.path.exists(model_path):
        print(f'loading model from ./model_zoo/vrt/{model_path}')

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)
    device = torch.device(f'cuda:{opt["gpu_idx"]}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    model.device = device

    for mode, dataset_opt in opt['datasets'].items():
        if mode == 'no-reference':
            # ----------------------------------------
            # prepare data
            # ----------------------------------------
            test_set = VideoTrain_NR_Dataset(dataset_opt)
            test_nr_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
            # ----------------------------------------
            # compute metrcs
            # ----------------------------------------
            stat = utils_test.test_metrics(model, test_nr_loader, dataset_opt, mode, opt)   
            

        elif mode == 'full-reference':
            # ----------------------------------------
            # prepare data
            # ----------------------------------------
            test_set = VideoTrain_FR_Dataset(dataset_opt)
            test_fr_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
            # ----------------------------------------
            # compute metrcs
            # ----------------------------------------
            stat = utils_test.test_metrics(model, test_fr_loader, dataset_opt, mode, opt)
        else:
            raise NotImplementedError("Mode [%s] is not recognized." % mode)
        
        # ----------------------------------------
        # save stat
        # ----------------------------------------
        stat = {vid: {loss: stat[vid][loss].detach().cpu().mean().item() for loss in stat[vid]} for vid in stat}
        means = {}
        for metric in dataset_opt['metrics']:
            mean = []
            for vid in stat:
                if metric in stat[vid].keys():
                    mean.append(stat[vid][metric])
            #print(metric)
            means[metric] = np.array(mean).mean()
        stat['mean'] = means

        with open(os.path.join(dataset_opt['save_stat_path'], f'{opt["task"]}.json'), 'w') as f:
            json.dump(stat, f, indent=2)
                  

if __name__ == '__main__':
    main()
