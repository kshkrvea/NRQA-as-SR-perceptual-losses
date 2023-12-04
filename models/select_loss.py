import torch.nn as nn
import torch

def select_loss(loss_name, args=dict(), mode='FR', device='cpu'):
    
    if loss_name == 'l1':
        Loss = nn.L1Loss()
    
    elif loss_name == 'l2':
        Loss = nn.MSELoss()
    
    elif loss_name == 'l2sum':
        Loss = nn.MSELoss(reduction='sum')

    elif loss_name == 'ssim':
        from models.loss_ssim import SSIMLoss
        Loss = SSIMLoss()

    elif loss_name == 'charbonnier':
        from models.loss import CharbonnierLoss
        Loss = CharbonnierLoss(**args)
    
    elif loss_name == 'perceptual':
        from models.loss import PerceptualLoss
        Loss = PerceptualLoss(device=device, **args)

    elif loss_name == 'tv':
        from models.loss import TVLoss
        Loss = TVLoss()

    elif loss_name == 'maniqa':
        from metrics.maniqa.src.model import MetricModel
        Loss = MetricModel(device=device, **args)
    
    elif loss_name == 'lpips':
        from models.loss import LpipsLoss
        Loss = LpipsLoss(**args)

    elif loss_name == 'mdtvsfa':
        from metrics.mdtvsfa import MDTVSFA

        Loss = MDTVSFA(device=device, **args)

    elif loss_name == 'paq2piq':
        # doesn't work
        #from metrics.paq2piq import InferenceModel, RoIPoolModel
        #model_state = torch.load('metrics/data/RoIPoolModel-fit.10.bs.120.pth', map_location=lambda storage, loc: storage)
        #model = RoIPoolModel()
        #model.load_state_dict(model_state["model"])
        #model = model.to(device)
        #model.eval()

        #paq2piq_model = InferenceModel(RoIPoolModel(), 'metrics/data/RoIPoolModel-fit.10.bs.120.pth', device=device)
        #paq2piq_model.blk_size = (3, 5)
        #return paq2piq_model.predict
        from metrics.paq2piq.paq2piq import InferenceModel, RoIPoolModel
        paq2piq_model = InferenceModel(RoIPoolModel(), 'metrics/data/RoIPoolModel-fit.10.bs.120.pth', device)
        paq2piq_model.blk_size = (3, 5)
        return paq2piq_model.forward
        #from metrics.paq2piq import Paq2Piq
        #Loss = Paq2Piq('metrics/data/RoIPoolModel-fit.10.bs.120.pth', device=device)

    elif loss_name == 'hyperiqa':
        from metrics import pyiqa_create_metric_wrapper
        Loss = pyiqa_create_metric_wrapper('hyperiqa', device=device, as_loss=True)

    else:
        raise NotImplementedError("Loss function name [%s] is not recognized." % loss_name)

    return Loss.to(device)