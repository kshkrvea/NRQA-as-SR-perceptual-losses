import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_image as util
from utils import utils_option
from utils.utils_dist import get_dist_info, init_dist
from data.select_dataset import select_dataset
from models.select_model import define_Model

from tqdm import tqdm
import torchvision
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure

psnr = PeakSignalNoiseRatio()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)


def main():

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)


    args = parser.parse_args()
    opt = utils_option.parse(args.opt, is_train=True)
    opt = utils_option.dict_to_nonedict(opt)
    
    logger = SummaryWriter(f"runs/{opt['model']}/{opt['task']}")

    opt['dist'] = args.dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------

    if opt['dist']:
        init_dist('pytorch')

    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # init start iterations
    # ----------------------------------------
    init_iter_G = utils_option.get_iteration(opt["path"]['pretrained_netG'], 'G')
    init_iter_optimizerG = utils_option.get_iteration(opt["path"]['pretrained_optimizerG'], 'optimizerG')
    current_step = max(init_iter_G, init_iter_optimizerG)
    opt['current_step'] = current_step

    
    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        #TODO

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        import time
        seed = int(time.time())
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    test_loader = None

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = select_dataset(dataset_opt)
            if opt['rank'] == 0:
                pass
                # logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'],
                                                   drop_last=True, seed=seed)
                
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                if 'test' not in opt['datasets']:
                    train_size = int(len(train_set) * 0.95)
                    test_size = len(train_set) - train_size
                    train_set, test_set = torch.utils.data.random_split(train_set, [train_size, test_size])
                    test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
                
                train_loader = DataLoader(train_set,
                                            batch_size=dataset_opt['dataloader_batch_size'],
                                            shuffle=dataset_opt['dataloader_shuffle'],
                                            num_workers=dataset_opt['dataloader_num_workers'],
                                            drop_last=True,
                                            pin_memory=True)
                

        elif phase == 'test':
            test_set = select_dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    example_data = next(iter(train_loader))
    img_grid = torchvision.utils.make_grid(example_data['L'][:, 0, ...])
    logger.add_image('dataset test images: L', img_grid)
    logger.close()

    

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    

    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(tqdm(train_loader)):
            current_step += 1
            
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            
            model.feed_data(train_data)
            
            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logger.add_scalar(f'current learning rate', model.current_learning_rate(), current_step)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                model.save(current_step)

            if opt['use_static_graph'] and (current_step == opt['train']['fix_iter'] - 1):
                current_step += 1
                model.update_learning_rate(current_step)
                model.save(current_step)
                current_step -= 1
            
            # -------------------------------
            # 6) testing
            # -------------------------------
              
            
            if None != test_loader and  current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                
                # train loss
                for loss in model.log_dict:
                    logger.add_scalar(f'train {loss} loss', model.log_dict[loss] / opt['train']['checkpoint_test'], current_step)
                    model.log_dict[loss] = 0
                
                # test loss
                test_loss = {loss: 0 for loss in model.G_lossfn_types}
                test_loss['ssim'] = 0
                test_loss['psnr'] = 0
                for _, test_data in enumerate(tqdm(test_loader)):
                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    output = visuals['E']
                    gt = visuals['H'] if 'H' in visuals else None
                    lq = visuals['L'] if 'L' in visuals else None
                    if gt is not None: # FR testing mode
                        for loss_fn, loss_type in zip(model.G_lossfns, model.G_lossfn_types):
                            if loss_type in ['tv']:
                               test_loss[loss_type] += loss_fn(output.to(device=model.device)) 
                            else:
                                test_loss[loss_type] += loss_fn(output.to(device=model.device), gt.to(device=model.device))
                        
                        test_loss['psnr'] += psnr(output, gt)
                        test_loss['ssim'] += ssim(output, gt)
                    else: #NR testing mode
                        pass
                        #TODO
                
                for loss in model.log_dict:
                    logger.add_scalar(f'test {loss} loss', test_loss[loss] / opt['train']['checkpoint_test'], current_step)
                
                logger.add_scalar(f'test psnr metric', test_loss['psnr'] / len(test_loader), current_step)
                logger.add_scalar(f'test ssim metric', test_loss['ssim'] / len(test_loader), current_step)

                img_grid = torchvision.utils.make_grid(lq[0])
                logger.add_image('LQ', img_grid)
                
                img_grid = torchvision.utils.make_grid(output[0])
                logger.add_image('SR', img_grid)
                
                img_grid = torchvision.utils.make_grid(gt[0])
                logger.add_image('GT', img_grid)
                
                logger.close()
                

            if current_step > opt['train']['total_iter']:
                model.save(current_step)
                #torch.save(model.netG.state_dict(), f'{opt["task"]}/models/{current_step}_G.pth')
                sys.exit()
        #sys.exit()

if __name__ == '__main__':
    main()
