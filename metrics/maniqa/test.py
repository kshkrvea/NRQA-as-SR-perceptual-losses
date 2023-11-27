#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import numpy as np
from src.model import MetricModel
import skvideo.io
import cv2
from tqdm import tqdm
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_path", type=str)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    bps = 3
    if args.width * args.height <= 0:
       raise RuntimeError("unsupported resolution")


    model = MetricModel('cuda:0', 'ckpt_koniq10k.pt')
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])    
    
    print("value")
    '''
    with open(args.dist_path, 'rb') as dist_rgb24, torch.no_grad():
        while True:
            dist = dist_rgb24.read(args.width * args.height * bps)
            print('qwe')
            if len(dist) == 0:
                break
            if len(dist) != args.width * args.height * bps:
                raise RuntimeError("unexpected end of stream dist_path")

            dist = np.frombuffer(dist, dtype='uint8').reshape((args.height,args.width,bps)) / 255.
            score = model(torch.unsqueeze(transform(dist), 0).type(torch.FloatTensor).to('cuda:0')).item()
            print(score)
    '''
    cap = cv2.VideoCapture(args.dist_path)
    scores = []
    ret = True
    frame_num = 0
    metric_values = []
    def get_dur(filename):
        video = cv2.VideoCapture(filename)
        return int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = get_dur(args.dist_path)
    scores = []
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dist = np.frombuffer(frame, dtype='uint8').reshape((args.height,args.width,bps)) / 255.
            score = model(torch.unsqueeze(transform(dist), 0).type(torch.FloatTensor).to('cuda:0')).item()
            new_elem = {"frame" : i, "maniqa" : score}
            #print(new_elem)
            metric_values.append(new_elem)
            scores.append(score)
    

    print('mean score: ', np.mean(scores))

if __name__ == "__main__":
   main()
