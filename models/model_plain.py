from collections import OrderedDict
import torch
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase

from utils.utils_regularizers import regularizer_orth, regularizer_clip

from models.select_loss import select_loss


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option

        self.netG = define_G(opt)
        

        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

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
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = {loss: 0 for loss in self.G_lossfn_types}

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

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

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            #if v.requires_grad:
            if True:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

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
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):    
        self.E = self.netG(self.L)
        #print(self.E.size(), self.H.size(), self.L.size())

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        
        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_loss = 0
        for weight, loss, loss_name in zip(self.G_lossfn_weights, self.G_lossfns, self.G_lossfn_types):
            loss_fn, loss_mode = loss
            
            if loss_mode == 'NR':
                loss_val = loss_fn(self.E)
            elif loss_mode == 'FR':
                loss_val = loss_fn(self.E, self.H)
            elif loss_mode == 'pseudo_FR':
                loss_val = loss_fn(self.E) - loss_fn(self.H)
            else:
                raise ValueError("Loss mode [%s] is not recognized." % loss_mode)
            
            mult = -1 if self.opt_train['G_lossfn_types'][loss_name]['reverse'] else 1
            G_loss += weight * mult * loss_val
            self.log_dict[loss_name] += loss_val
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

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

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()


    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
