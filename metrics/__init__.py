import torch
import pyiqa


class EnsureBCHW(torch.nn.Module):
    @staticmethod
    def forward(x):
        return torch.flatten(x, start_dim=0, end_dim=1) if len(x.shape) == 5 else x


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def pyiqa_create_metric_wrapper(metric_name, as_loss=False, device=None, **kwargs):
    metric = pyiqa.create_metric(metric_name, as_loss, device, **kwargs)
    freeze(metric)

    return torch.nn.Sequential(EnsureBCHW(), metric)
