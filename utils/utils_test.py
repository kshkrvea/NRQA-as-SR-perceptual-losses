from tqdm import tqdm
import numpy as np
import cv2
import os

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
import lpips
from DISTS_pytorch import DISTS
import erqa
import torch

measure_functions_fr = {
        'ssim': StructuralSimilarityIndexMeasure(data_range=1.0),
        'psnr': PeakSignalNoiseRatio(),
        'lpips_alex': lpips.LPIPS(net='alex'),
        'lpips_vgg': lpips.LPIPS(net='vgg'),
        'dists': DISTS(),
        'erqa': erqa.ERQA()
    }

def get_paq2piq_model(device):
    from metrics.paq2piq import InferenceModel, RoIPoolModel
    paq2piq_model = InferenceModel(RoIPoolModel(), 'metrics/data/RoIPoolModel-fit.10.bs.120.pth', device)
    paq2piq_model.blk_size = (3, 5)
    return paq2piq_model.predict

measure_functions_nr = {
        'paq2piq': get_paq2piq_model
    }


def save_video(vid, vid_path):
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    for i, frame in enumerate(vid):
        frame = frame.detach().cpu().numpy()
        frame = np.moveaxis(frame, 0, 2)
        cv2.imwrite(os.path.join(vid_path, f'{i:03d}.png'), frame[..., ::-1] * 255)



def test_metrics(model, test_loader, dataset_opt, mode, opt):

    need_H = mode == 'full-reference'

    measure_values = {}
    n_tested_videos = 0
    for n_vid, test_data in enumerate(tqdm(test_loader)):
        measure_values[f'{n_vid:03d}'] = {}
        measure_values[f'gt_{n_vid:03d}'] = {}
        measure_values[f'lq_{n_vid:03d}'] = {}
        output = model.test_video(test_data['L'].to(device=model.device), opt['args'])[0]
        lq = test_data['L'][0]
        gt = test_data['H'][0] if need_H else None

        if dataset_opt['save_results']:
            save_video(output, os.path.join(dataset_opt['save_path'], f'{n_vid:03d}', opt['task']))
            
            gt_path = os.path.join(dataset_opt['save_path'], f'{n_vid:03d}', 'gt')
            lq_path = os.path.join(dataset_opt['save_path'], f'{n_vid:03d}', 'lq')
            if gt is not None and not os.path.exists(gt_path):
                save_video(gt, gt_path)
            if not os.path.exists(lq_path):
                save_video(lq, lq_path)

        if gt is not None: # FR testing mode
            for metric_name, metric_fn in measure_functions_fr.items():
                if metric_name in dataset_opt['metrics']:
                    if metric_name != 'erqa': #ERQA doesn't support GPU
                        metric_fn = metric_fn.to(device=model.device)
                        measure_values[f'{n_vid:03d}'][metric_name] = metric_fn(output.to(device=model.device), gt.to(device=model.device))
                    else:
                        measure_values[f'{n_vid:03d}'][metric_name] = metric_fn(output.to(device=model.device), gt.to(device=model.device))
            
            for metric_name, get_metric_fn in measure_functions_nr.items():
                if metric_name in dataset_opt['metrics']:
                    metric_fn = get_metric_fn(model.device)
                    #print(gt.shape)
                    measure_values[f'{n_vid:03d}'][metric_name] = metric_fn(output[0])
                    #print("---")
                    measure_values[f'gt_{n_vid:03d}'][metric_name] = metric_fn(gt[0])                    
                    measure_values[f'lq_{n_vid:03d}'][metric_name] = metric_fn(lq[0])

        else: #NR testing mode
            for metric_name, get_metric_fn in measure_functions_nr.items():
                if metric_name in dataset_opt['metrics']:
                    metric_fn = get_metric_fn(model.device)
                    measure_values[f'{n_vid:03d}'][metric_name] = metric_fn(output[0])
                    measure_values[f'lq_{n_vid:03d}'][metric_name] = metric_fn(lq[0])

        n_tested_videos += 1
        if dataset_opt['n_videos'] != 'all' and n_tested_videos >= dataset_opt['n_videos']:
            break
        
    
    return measure_values