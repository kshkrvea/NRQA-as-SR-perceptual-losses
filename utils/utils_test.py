from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
from metrics.select_metric import select_metric

def save_video(vid, vid_path):
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    for i, frame in enumerate(vid):
        frame = frame.detach().cpu().numpy()
        frame = np.moveaxis(frame, 0, 2)
        cv2.imwrite(os.path.join(vid_path, f'{i:03d}.png'), frame[..., ::-1] * 255)

def test_metrics(model, test_loader, dataset_opt, opt, dataset_name):
    
    need_H = dataset_opt['mode'] == 'FR'
    measure_values = {}
    n_tested_videos = 0

    metric_fns = {}
    for metric_name, parameters in opt['metrics'].items():  
        metric_fns[metric_name] = (select_metric(metric_name=metric_name, args=parameters['args'], device=model.device), parameters)

    for n_vid, test_data in enumerate(tqdm(test_loader)):
        measure_values[f'{n_vid:03d}'] = {}
        measure_values[f'gt_{n_vid:03d}'] = {}
        measure_values[f'lq_{n_vid:03d}'] = {}
        output = model.test_video(test_data['L'].to(device=model.device), opt['args'])[0]
        output = output.clamp(0, 1)
        lq = test_data['L'][0]
        gt = test_data['H'][0] if need_H else None

        if dataset_opt['save_results']:
            save_video(output, os.path.join(dataset_opt['save_path'], dataset_name, f'{n_vid:03d}', opt['task']))
            
            gt_path = os.path.join(dataset_opt['save_path'], dataset_name, f'{n_vid:03d}', 'gt')
            lq_path = os.path.join(dataset_opt['save_path'], dataset_name, f'{n_vid:03d}', 'lq')
            if gt is not None and not os.path.exists(gt_path):
                save_video(gt, gt_path)
            if not os.path.exists(lq_path):
                save_video(lq, lq_path)

        output_gpu = output.to(device=model.device)
        gt_gpu = None if gt is None else gt.to(device=model.device)
        lq_gpu = lq.to(device=model.device)

        for metric_name, metric_info in metric_fns.items():
            metric_fn, parameters = metric_info
            gt_ = gt_gpu if parameters['gpu_support'] else gt
            output_ = output_gpu if parameters['gpu_support'] else output
            with torch.no_grad():
                if parameters['mode'] == 'NR':
                    lq_ = lq_gpu if parameters['gpu_support'] else lq
                    measure_values[f'{n_vid:03d}'][metric_name] = metric_fn(output_)                  
                    measure_values[f'lq_{n_vid:03d}'][metric_name] = metric_fn(lq_)
                    if gt is not None: # FR dataset 
                        measure_values[f'gt_{n_vid:03d}'][metric_name] = metric_fn(gt_)      
                    else: #NR dataset, no gt
                        pass
                
                elif parameters['mode'] == 'FR': # FR dataset   
                    if gt is not None: # FR dataset 
                        measure_values[f'{n_vid:03d}'][metric_name] = metric_fn(gt_, output_)
                    else: #NR dataset, no gt
                        pass

                else:
                    raise NotImplementedError("Metric mode [%s] is not recognized." % parameters['mode'])
                
        n_tested_videos += 1
        if dataset_opt['n_videos'] != 'all' and n_tested_videos >= dataset_opt['n_videos']:
            break
    
    return measure_values
