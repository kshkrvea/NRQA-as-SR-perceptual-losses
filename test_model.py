import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import utils_test
from utils import utils_option

from models.select_network import select_net
from data.select_dataset import select_dataset

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
        print(f'loading model from {model_path}')

    pretrained_model = torch.load(model_path)  
    
    weights = pretrained_model
    if 'params' in pretrained_model.keys():
        weights = pretrained_model['params']
    elif 'state_dict' in pretrained_model.keys():
        weights = pretrained_model['state_dict']
    
    """from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        if k == 'step_counter':
            continue
        if k[:10] == 'generator.':
            name = k[10:] # remove `module.`
        else:
            name = k

        if '1.upsample_conv' in name:
            name = 'upconv1.' + name.split('.')[-1]
        elif '2.upsample_conv' in name:
            name = 'upconv2.' + name.split('.')[-1]
        else:
            pass
        
        if 'spynet.basic_module' in name:
            name = name.replace('.conv', '')
            parts = name.split('.')
            parts[-2] = str(int(parts[-2]) * 2)
            name = '.'.join(parts)

        new_state_dict[name] = v """
        

    model.load_state_dict(weights, strict=True)
    
    device = torch.device(f'cuda:{opt["gpu_idx"]}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    model.device = device
    print(opt['datasets'])
    for mode, dataset_opt in opt['datasets'].items():
        print(1)
        continue
        # ----------------------------------------
        # prepare data
        # ----------------------------------------
        test_set = select_dataset(dataset_opt)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=12)

        # ----------------------------------------
        # compute metrcs
        # ----------------------------------------
        stat = {}#utils_test.test_metrics(model, test_loader, dataset_opt, mode, opt)   
        
        # ----------------------------------------
        # save stat
        # ----------------------------------------
        stat = {vid: {loss: stat[vid][loss].detach().cpu().mean().item() for loss in stat[vid]} for vid in stat}
        means = {}
        for metric in opt['metrics']:
            mean = []
            for vid in stat:
                if metric in stat[vid].keys():
                    mean.append(stat[vid][metric])
            #print(metric)
            means[metric] = np.array(mean).mean()
        stat['mean'] = means

        if not os.path.exists(dataset_opt['save_stat_path']):
            os.makedirs(dataset_opt['save_stat_path'])
        with open(os.path.join(dataset_opt['save_stat_path'], f'{opt["task"]}.json'), 'w') as f:
            json.dump(stat, f, indent=2)
                  

if __name__ == '__main__':
    main()
