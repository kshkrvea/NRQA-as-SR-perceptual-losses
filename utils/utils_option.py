import re
import os
import yaml


def simplified_parse(opt_path):
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
    opt = dict_to_nonedict(opt)
    return opt


def parse(opt_path, is_train=True):

    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
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

    opt['opt_path'] = opt_path
    opt['is_train'] = is_train

    # ----------------------------------------
    # set default
    # ----------------------------------------
    if 'merge_bn' not in opt:
        opt['merge_bn'] = False
        opt['merge_bn_startpoint'] = -1

    if 'scale' not in opt:
        opt['scale'] = 1

    # ----------------------------------------
    # datasets
    # ----------------------------------------
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = opt['scale']  # broadcast
        dataset['n_channels'] = opt['n_channels']  # broadcast
        if 'dataroot_H' in dataset and dataset['dataroot_H'] is not None:
            dataset['dataroot_H'] = os.path.expanduser(dataset['dataroot_H'])
        if 'dataroot_L' in dataset and dataset['dataroot_L'] is not None:
            dataset['dataroot_L'] = os.path.expanduser(dataset['dataroot_L'])

    # ----------------------------------------
    # path
    # ----------------------------------------
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)

    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['task'] = path_task
    opt['path']['log'] = path_task
    opt['path']['options'] = os.path.join(path_task, 'options')

    if is_train:
        opt['path']['models'] = os.path.join(path_task, 'models')
        opt['path']['images'] = os.path.join(path_task, 'images')
    else:  # test
        opt['path']['images'] = os.path.join(path_task, 'test_images')

    # ----------------------------------------
    # network
    # ----------------------------------------
    opt['netG']['scale'] = opt['scale'] if 'scale' in opt else 1

    # ----------------------------------------
    # GPU devices
    # ----------------------------------------
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # ----------------------------------------
    # default setting for distributeddataparallel
    # ----------------------------------------
    if 'find_unused_parameters' not in opt:
        opt['find_unused_parameters'] = True
    if 'use_static_graph' not in opt:
        opt['use_static_graph'] = False
    if 'dist' not in opt:
        opt['dist'] = False
    opt['num_gpu'] = len(opt['gpu_ids'])
    print('number of GPUs is: ' + str(opt['num_gpu']))

    # ----------------------------------------
    # default setting for perceptual loss
    # ----------------------------------------
    if 'F_feature_layer' not in opt['train']:
        opt['train']['F_feature_layer'] = 34  # 25; [2,7,16,25,34]
    if 'F_weights' not in opt['train']:
        opt['train']['F_weights'] = 1.0  # 1.0; [0.1,0.1,1.0,1.0,1.0]
    if 'F_lossfn_type' not in opt['train']:
        opt['train']['F_lossfn_type'] = 'l1'
    if 'F_use_input_norm' not in opt['train']:
        opt['train']['F_use_input_norm'] = True
    if 'F_use_range_norm' not in opt['train']:
        opt['train']['F_use_range_norm'] = False

    # ----------------------------------------
    # default setting for optimizer
    # ----------------------------------------
    if 'G_optimizer_type' not in opt['train']:
        opt['train']['G_optimizer_type'] = "adam"
    if 'G_optimizer_betas' not in opt['train']:
        opt['train']['G_optimizer_betas'] = [0.9,0.999]
    if 'G_scheduler_restart_weights' not in opt['train']:
        opt['train']['G_scheduler_restart_weights'] = 1
    if 'G_optimizer_wd' not in opt['train']:
        opt['train']['G_optimizer_wd'] = 0
    if 'G_optimizer_reuse' not in opt['train']:
        opt['train']['G_optimizer_reuse'] = False
    if 'netD' in opt and 'D_optimizer_reuse' not in opt['train']:
        opt['train']['D_optimizer_reuse'] = False

    # ----------------------------------------
    # default setting of strict for model loading
    # ----------------------------------------
    if 'G_param_strict' not in opt['train']:
        opt['train']['G_param_strict'] = True
    if 'netD' in opt and 'D_param_strict' not in opt['path']:
        opt['train']['D_param_strict'] = True
    if 'E_param_strict' not in opt['path']:
        opt['train']['E_param_strict'] = True

    # ----------------------------------------
    # Exponential Moving Average
    # ----------------------------------------
    if 'E_decay' not in opt['train']:
        opt['train']['E_decay'] = 0

    # ----------------------------------------
    # default setting for discriminator
    # ----------------------------------------
    if 'netD' in opt:
        if 'net_type' not in opt['netD']:
            opt['netD']['net_type'] = 'discriminator_patchgan'  # discriminator_unet
        if 'in_nc' not in opt['netD']:
            opt['netD']['in_nc'] = 3
        if 'base_nc' not in opt['netD']:
            opt['netD']['base_nc'] = 64
        if 'n_layers' not in opt['netD']:
            opt['netD']['n_layers'] = 3
        if 'norm_type' not in opt['netD']:
            opt['netD']['norm_type'] = 'spectral'


    return opt




def get_iteration(file_, file_type='G'):
    file_ = file_ if not file_ is None else ''
    iter_current = re.findall(r"(\d+)_{}.pth".format(file_type), file_)
    return 0 if not iter_current else int(iter_current[0])


'''
# --------------------------------------------
# convert OrderedDict to NoneDict,
# return None for missing key
# --------------------------------------------
'''
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None
