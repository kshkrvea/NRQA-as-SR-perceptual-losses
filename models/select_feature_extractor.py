

def select_FE(model='vgg', device='cpu'):
    
    if model == 'vgg':
        import torchvision
        return torchvision.models.vgg19(pretrained=True)
    
    elif model == 'maniqa':
        from metrics.maniqa.src.model import MetricModel
        return MetricModel(model_path='metrics/maniqa/ckpt_koniq10k.pt', device=device)
    
    else:
        raise NotImplementedError("Feature extraction model name [%s] is not recognized." % model)
