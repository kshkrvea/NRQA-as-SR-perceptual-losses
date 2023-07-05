import functools
import torch
from torch.nn import init
import os

"""
# --------------------------------------------
# select the network of G and D
# --------------------------------------------
"""

# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']
    args = opt_net['arguments']
    # ----------------------------------------
    # SwinIR
    # ----------------------------------------
    if net_type == 'swinir':
        from archs.swinir.network_swinir import SwinIR as net
        netG = net(**args)    

    # ----------------------------------------
    # VRT
    # ----------------------------------------
    elif net_type == 'vrt':
        from archs.vrt.network_vrt import VRT as net
        netG = net(**args)
        
    # ----------------------------------------
    # RVRT
    # ----------------------------------------
    elif net_type == 'rvrt':
        from archs.rvrt.network_rvrt import RVRT as net
        netG = net(**args)
    
    # ----------------------------------------
    # EDVR: cuda extension required
    # ----------------------------------------
    elif net_type == 'edvr':
        from archs.edvr.basicsr.archs.edvr_arch import EDVR as net
        netG = net(**args)

    # ----------------------------------------
    # RBPN
    # ----------------------------------------
    elif net_type == 'rbpn':
        from archs.rbpn.network_rbpn import RBPN as net
        netG = net(**args)

    # ----------------------------------------
    # SRCNN
    # ----------------------------------------
    elif net_type == 'srcnn':
        from archs.srcnn.network_srcnn import SRCNN as net
        netG = net(**args)

    # ----------------------------------------
    # others
    # ----------------------------------------
    # TODO

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------    
    if opt['is_train']:
        # ----------------------------------------
        # load pre-trained G model
        # ----------------------------------------
        model_path = opt["path"]["pretrained_netG"]
        if model_path and os.path.exists(model_path):
            print(f'loading model from {model_path}')        
            # original saved file with DataParallel
            state_dict = torch.load(model_path)
        else:
            state_dict = netG.state_dict()
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == 'module.':
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v 

        netG.load_state_dict(new_state_dict['params'] if 'params' in new_state_dict.keys() else new_state_dict, strict=True)
        
        if opt['scale'] != opt['weight_scale']:
            from archs.vrt.network_vrt import Upsample
            netG.upsample = Upsample(opt['weights_scale'], 64)
        # ----------------------------------------
        # freeze layers
        # ----------------------------------------
        for child in list(netG.children()):
            for param in child.parameters():
                param.requires_grad = True

        for child in list(netG.children())[:opt_net['unfreeze_blocks']]:
            for param in child.parameters():
                param.requires_grad = False
    
    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt):
    opt_net = opt['netD']
    net_type = opt_net['net_type']
    from archs.gan.network_discriminator import Discriminator_VGG_128_SN as discriminator
    netD = discriminator()

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])
    
    # ----------------------------------------
    # load pre-trained D model
    # ----------------------------------------
    model_path = opt["path"]["pretrained_netD"]
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    pretrained_model = torch.load(model_path)
    netD.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)
    if opt['scale'] != opt['weight_scale']:
        pass
        #TODO
        #netD.upsample = Upsample(opt['weight_scale'], 64)

    # ----------------------------------------
    # freeze layers
    # ----------------------------------------
    for child in list(netD.children()):
        for param in child.parameters():
            param.requires_grad = True

    for child in list(netD.children())[:opt_net['unfreeze_blocks']]:
        for param in child.parameters():
            param.requires_grad = False

    return netD



"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
