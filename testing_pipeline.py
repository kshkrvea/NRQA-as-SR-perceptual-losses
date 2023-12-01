import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import utils_test
from utils import utils_option

from models.select_network import select_net
from data.select_dataset import select_dataset
from metrics.select_metric import select_metric

import os
import json
import pandas as pd

def main():

    # ----------------------------------------
    # options
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option yaml file.')

    args = parser.parse_args()
    opt_path = args.opt
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = utils_option.simplified_parse(opt_path)
    

    # ----------------------------------------
    # model (netG)
    # ----------------------------------------
    model = select_net(opt)
    device = torch.device(f'cuda:{opt["gpu_idx"]}' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # metrics
    # ----------------------------------------
    metrics = utils_option.simplified_parse(opt['metrics_config'])
    metric_fns = {}
    for metric_name, parameters in metrics.items():  
        metric_fns[metric_name] = (select_metric(metric_name=metric_name, args=parameters['args'], device=device), parameters)

    
    # ----------------------------------------
    # collected stats
    # ----------------------------------------
    result_stats = {}
    if not opt['stats_path'] is None:
        xl = pd.ExcelFile(os.path.join(opt['collected_stats_root'], opt['stats_path']))
        for dataset_name in opt['datasets'].keys():
            if dataset_name in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=dataset_name, index_col=0) 
                result_stats[dataset_name] = df.to_dict(orient='index')

    for dataset_name in opt['datasets'].keys():
        if not dataset_name in result_stats.keys():
            result_stats[dataset_name] = {}
    
    for version, version_opt in opt['versions'].items():
        model_path = version_opt['pretrained_netG']
        if os.path.exists(model_path):
            print(f'Loading model from: {model_path}')

        pretrained_model = torch.load(model_path)  
        
        weights = pretrained_model
        if 'params' in pretrained_model.keys():
            weights = pretrained_model['params']
        elif 'state_dict' in pretrained_model.keys():
            weights = pretrained_model['state_dict']
        

        if not version_opt['weights_mismatch'] is None:
            weights = utils_test.resolve_weights_mismatch(weights, version_opt['weights_mismatch'])
        
        if version_opt['metrics'] == 'all':
            needed_metric_fns = metric_fns
        else:
            needed_metric_fns = {k: v for k, v in metric_fns.items() if k in version_opt['metrics']}

        model.load_state_dict(weights, strict=True)
        
        model.eval()
        model = model.to(device)
        model.device = device
        
        for dataset_name, dataset_opt in opt['datasets'].items():
            # ----------------------------------------
            # prepare data
            # ----------------------------------------
            test_set = select_dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=12)

            # ----------------------------------------
            # compute metrcs
            # ----------------------------------------
            stat = utils_test.test_metrics(model, test_loader, dataset_opt, opt, dataset_name, version, needed_metric_fns)   
            
            # ----------------------------------------
            # save stat
            # ----------------------------------------
            stat = {vid: {loss: stat[vid][loss].detach().cpu().mean().item() for loss in stat[vid]} for vid in stat}
            means = {}
            for metric in metrics:
                mean = []
                for vid in stat:
                    if metric in stat[vid].keys():
                        mean.append(stat[vid][metric])
                #print(metric)
                means[metric] = np.array(mean).mean()
            stat['mean'] = means
            
            if version in result_stats[dataset_name].keys():
                result_stats[dataset_name][version] = {**result_stats[dataset_name][version], **means} # merge stats, overwrited with the new ones
            else:
                result_stats[dataset_name][version] = means
   
            if not os.path.exists(os.path.join(opt['collected_stats_root'], dataset_name)):
                os.makedirs(os.path.join(opt['collected_stats_root'], dataset_name))
            with open(os.path.join(opt['collected_stats_root'], dataset_name, f'{version}.json') , 'w') as f:
                json.dump(stat, f, indent=2)
    
    stat_path = os.path.join(opt['collected_stats_root'], opt['save_stats_name'])
    with pd.ExcelWriter(stat_path) as writer:  
        for dataset_name, results in result_stats.items():
            def h_max(s):
                reversed = metrics[s.name]['reversed']
                is_max = s == s.max() if not reversed else s == s.min()
                return ['color: red' if cell else '' for cell in is_max]
            
            dfd = pd.DataFrame.from_dict(data=results, orient='index')
            dfd.style.apply(h_max).to_excel(writer, sheet_name=dataset_name, engine='openpyxl')
                  
if __name__ == '__main__':
    main()
