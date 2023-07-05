from torch import nn
import torchvision.transforms.functional as F
import torch


class SRCNN(nn.Module):
    def __init__(self, num_channels=1, upscale=2):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.upscale = upscale

    def forward(self, x):
        res_size = torch.tensor(x.size())
        res_size[-2:] *= self.upscale
        outx = []
        for i in range(x.size()[1]):
            frame = x[:, i, ...]
            frame = F.resize(frame, tuple(res_size[-2:]), antialias=True)
            frame = self.relu(self.conv1(frame))
            frame = self.relu(self.conv2(frame))
            frame = self.conv3(frame)
            outx.append(frame)
        result = torch.stack(outx, 1)

        return result
    
    def test_video(self, x, args=None):
        with torch.no_grad():
            return self.forward(x)
