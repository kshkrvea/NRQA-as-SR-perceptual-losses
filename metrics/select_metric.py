import torch

def select_metric(metric_name, args=dict(), device='cpu'):

    if metric_name == 'ssim':
        from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
        Loss = ssim(**args)

    elif metric_name == 'psnr':
        from torchmetrics.image import PeakSignalNoiseRatio as psnr
        Loss = psnr()

    elif metric_name == 'charbonnier':
        from models.loss import CharbonnierLoss
        Loss = CharbonnierLoss(**args)

    elif metric_name == 'maniqa':
        from metrics.maniqa.src.model import MetricModel
        Loss = MetricModel(device=device, **args)
    
    elif metric_name in ['lpips_vgg', 'lpips_alex', 'lpips']:
        from models.loss import LpipsLoss
        Loss = LpipsLoss(**args)
    
    elif metric_name == 'dists':
        from DISTS_pytorch import DISTS
        Loss = DISTS()
    
    elif metric_name == 'paq2piq':
        from metrics.paq2piq.paq2piq import InferenceModel, RoIPoolModel
        paq2piq_model = InferenceModel(RoIPoolModel(), device=device, **args)
        paq2piq_model.blk_size = (3, 5)
        return paq2piq_model.predict

    elif metric_name == 'erqa':
        import erqa
        return erqa.ERQA()

    elif metric_name == 'mdtvsfa':
        from metrics.mdtvsfa import MDTVSFA

        return MDTVSFA(device=device, **args)

    elif metric_name == 'nima':
        from metrics import pyiqa_create_metric_wrapper, PadIfNeeded
        Loss = pyiqa_create_metric_wrapper('nima', device=device)
        # InceptionResNetV2 takes at least 75x75 images
        Loss = torch.nn.Sequential(PadIfNeeded(min_height=75, min_width=75), Loss)

    elif metric_name == 'hyperiqa':
        from metrics import pyiqa_create_metric_wrapper, PadIfNeeded
        Loss = pyiqa_create_metric_wrapper('hyperiqa', device=device)
        # HyperNet takes random crops of size 224x224 to evaluate images
        Loss = torch.nn.Sequential(PadIfNeeded(min_height=224, min_width=224), Loss)

    elif metric_name == 'clipiqa':
        from metrics import pyiqa_create_metric_wrapper
        Loss = pyiqa_create_metric_wrapper('clipiqa', device=device)

    else:
        raise NotImplementedError("Loss function name [%s] is not recognized." % metric_name)

    return Loss.to(device)