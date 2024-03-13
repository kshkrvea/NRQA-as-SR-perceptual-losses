from collections import OrderedDict
import torch
from torch.optim import Adam

from models.select_network import define_G
from models.model_plain import ModelPlain

from archs.iSeeBetter.dataset import get_flow
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from models.select_loss import select_loss
from models.spynet import SpyNet
import torch.nn.functional as F

from archs.iSeeBetter.SRGAN.model import Discriminator
from archs.iSeeBetter.rbpn import GeneratorLoss

# vid2img => changed training and inference scheme
class Model_iSeeBetter(ModelPlain):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(Model_iSeeBetter, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)

        if self.is_train:
            self.netD = Discriminator()
            self.netD = self.model_to_device(self.netD)
            if self.opt['netD']['freeze']:
                for child in list(self.netD.children()):
                    for param in child.parameters():
                        param.requires_grad = False
            if self.opt_train['E_decay'] > 0:
                self.netE = define_G(opt).to(self.device).eval()
        
        self.middle_frame_idx = int(self.netG.module.nFrames / 2)
        self.spynet = SpyNet(load_path=self.opt['netG']['spynet_path'])

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """
    
    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.netG.train()                     # set training mode,for BN
        self.netD.train()
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = {loss: 0 for loss in self.G_lossfn_types + ['adv_iseebetter', 'd_iseebetter']}
        self.spynet = self.spynet.to(self.device)

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

        load_path_optimizerD = self.opt['path']['pretrained_optimizerD']
        if load_path_optimizerD is not None and self.opt_train['D_optimizer_reuse']:
            print('Loading optimizerD [{:s}] ...'.format(load_path_optimizerD))
            self.load_optimizer(load_path_optimizerD, self.D_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.netD, 'D', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        if self.opt_train['D_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.D_optimizer, 'optimizerD', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_types = self.opt_train['G_lossfn_types']
        self.G_lossfn_types = []
        self.G_lossfn_weights = []       
        self.G_lossfns = []
        
        for loss, params in G_lossfn_types.items():
            mode = 'FR' if 'mode' not in params.keys() else params['mode']
            args = dict() if 'args' not in params.keys() else params['args']
            self.G_lossfns.append([select_loss(loss, args=args, device=self.device), mode])
            self.G_lossfn_weights.append(params['weight'])
            self.G_lossfn_types.append(loss)
        
        self.generatorCriterion = GeneratorLoss().to(device=self.device)

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
            #if True:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError
        
        self.D_optimizer = Adam(self.netD.parameters(), 
                                lr=self.opt_train['D_optimizer_lr'],
                                betas=self.opt_train['D_optimizer_betas'],
                                weight_decay=0)

    # ----------------------------------------
    # define scheduler
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler']['type'] == 'MultiStepLR':
            from torch.optim.lr_scheduler import MultiStepLR as scheduler
        elif self.opt_train['G_scheduler']['type'] == 'CosineAnnealingWarmRestarts':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as scheduler
        elif self.opt_train['G_scheduler']['type'] == 'CosineCycleAnnealingWarmRestarts':
            from utils.utils_schedulers import CosineCycleAnnealingWarmRestarts as scheduler
        else:
            raise NotImplementedError
        self.schedulers.append(scheduler(self.G_optimizer, **self.opt_train['G_scheduler']['params']))
        self.schedulers.append(scheduler(self.D_optimizer, **self.opt_train['D_scheduler']['params']))
    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.input = self.L[:, self.middle_frame_idx].to(self.device)
        self.neigbor = torch.cat((self.L[:,:self.middle_frame_idx], self.L[:,self.middle_frame_idx+1:]), axis=1).to(self.device)
        #self.neigbor = self.L.to(self.device)
        input_stack = self.input[:, None, ...].repeat(1, self.neigbor.size()[1], 1, 1, 1)
        b, t, c, h, w = input_stack.size()
        input_stack = input_stack.reshape(-1, c, h, w).to(self.device)
        neigbor = self.neigbor.reshape(-1, c, h, w).to(self.device)
        flow = self.spynet(input_stack, neigbor)
        self.flow = flow.reshape(b, t, 2, h, w).detach()        
        self.bicubic = F.interpolate(self.input, (h*self.opt['scale'], w*self.opt['scale']))
        if need_H:
            self.H = data['H'].to(self.device)
            self.target = self.H[:, self.middle_frame_idx]
            self.H = self.target
        self.L = self.input
        #print(self.target.size(), self.H.size(), self.bicubic.size(), self.L.size(), self.neigbor.size(), self.flow.size())

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):    
        #print(self.input.size(), self.neigbor.size(), self.flow.size())
        self.E = self.netG(self.input, self.neigbor, self.flow)
        #print(self.E.size(), self.H.size(), self.bicubic.size(), self.L.size())
        self.E = self.bicubic + self.E
        #print(self.E.size(), self.H.size(), self.bicubic.size(), self.L.size())

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        with torch.autograd.set_detect_anomaly(False):
        ################################################################################################################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ################################################################################################################
            for p in self.netD.parameters():
                p.requires_grad = True

            self.netD.zero_grad()
            self.netG_forward()

            real_labels = self.netD(self.target).mean()
            fake_labels = self.netD(self.E.detach()).mean()

            D_loss = 1 - real_labels + fake_labels
            #D_loss = D_loss / self.L.size()[0]

            D_loss.backward()
            self.D_optimizer.step()
            self.log_dict['d_iseebetter'] += D_loss.item()
            del D_loss
            del real_labels
            del fake_labels
            #del self.E

            ################################################################################################################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ################################################################################################################
            for p in self.netD.parameters():
                p.requires_grad = False
            self.netG.zero_grad()
            fake_labels = self.netD(self.E).mean()
            adv_loss = self.generatorCriterion(fake_labels, self.E, self.target, 0)
            del fake_labels
            adv_loss = adv_loss / self.L.size()[0]
            self.log_dict['adv_iseebetter'] += adv_loss.item()
            G_loss = adv_loss * self.opt['train']['adv_loss_weight']
            del adv_loss
            #G_loss = 0
            for weight, loss, loss_name in zip(self.G_lossfn_weights, self.G_lossfns, self.G_lossfn_types):
                loss_fn, loss_mode = loss
                
                if loss_mode == 'NR':
                    loss_val = loss_fn(self.E)
                elif loss_mode == 'FR':
                    loss_val = loss_fn(self.E, self.target)
                elif loss_mode == 'pseudo_FR':
                    loss_val = loss_fn(self.E) - loss_fn(self.target) # в self.target нет градиентов чзх
                else:
                    raise ValueError("Loss mode [%s] is not recognized." % loss_mode)
                
                mult = -1 if self.opt_train['G_lossfn_types'][loss_name]['reverse'] else 1
                G_loss += weight * mult * loss_val
                self.log_dict[loss_name] += loss_val.item()
                del loss_val
            #G_loss.requires_grad = True
            G_loss.backward()
            # ------------------------------------
            # clip_grad
            # ------------------------------------
            # `clip_grad_norm` helps prevent the exploding gradient problem.
            G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
            if G_optimizer_clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

            self.G_optimizer.step()
            del G_loss
            # ------------------------------------
            # regularizer
            # ------------------------------------
            G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
            if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
                self.netG.apply(regularizer_orth)
            G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
            if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
                self.netG.apply(regularizer_clip)

            # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
            #self.log_dict['G_loss'] = G_losses #G_loss.item()
                
            D_regularizer_orthstep = self.opt_train['D_regularizer_orthstep'] if self.opt_train['D_regularizer_orthstep'] else 0
            if D_regularizer_orthstep > 0 and current_step % D_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
                self.netD.apply(regularizer_orth)
            D_regularizer_clipstep = self.opt_train['D_regularizer_clipstep'] if self.opt_train['D_regularizer_clipstep'] else 0
            if D_regularizer_clipstep > 0 and current_step % D_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
                self.netG.apply(regularizer_clip)

            if self.opt_train['E_decay'] > 0:
                self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
            self.E = self.E[None, ...]
            self.H = self.H[None, ...]
            self.L = self.L[None, ...]
            #print(self.E.size(), self.H.size(), self.bicubic.size(), self.L.size())
        self.netG.train()

