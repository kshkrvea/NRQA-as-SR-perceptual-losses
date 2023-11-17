
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""
def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'gan':     # one input: L
        from models.model_gan import ModelGAN as M

    elif model == 'vrt':     # one video input L, for VRT
        from models.model_vrt import ModelVRT as M

    elif model in ['edvr', 'rbpn', 'srcnn', 'vggsr', 'basicvsrpp']: 
        from models.model_plain import ModelPlain as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
