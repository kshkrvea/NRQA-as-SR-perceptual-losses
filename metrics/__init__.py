import torch
import pyiqa


class EnsureBCHW(torch.nn.Module):
    @staticmethod
    def forward(x):
        return torch.flatten(x, start_dim=0, end_dim=1) if len(x.shape) == 5 else x


class PadIfNeeded(torch.nn.Module):
    def __init__(self, min_height, min_width):
        super().__init__()
        self.min_height = min_height
        self.min_width = min_width

    def forward(self, x):
        height, width = x.shape[-2:]

        if height >= self.min_height and width >= self.min_width:
            return x

        pad_height = self.min_height - height
        pad_width = self.min_width - width

        pad = (pad_width // 2, pad_width - pad_width // 2,
               pad_height // 2, pad_height - pad_height // 2)

        return torch.nn.functional.pad(x, pad)


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def pyiqa_create_metric_wrapper(metric_name, as_loss=False, device=None, **kwargs):
    metric = pyiqa.create_metric(metric_name, as_loss, device, **kwargs)
    freeze(metric)

    return torch.nn.Sequential(EnsureBCHW(), metric)
