import torch
import torch.nn as nn
import torch.nn.functional as F

from imetrics.subjects.mdtvsfa.src.model import CNNModel


class VQAModel(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size

        self.dimemsion_reduction = nn.Linear(input_size, reduced_size)
        self.feature_aggregation = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.regression = nn.Linear(hidden_size, 1)

        # 4 parameters
        self.nlm = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
            nn.Linear(1, 1),
        )

        torch.nn.init.constant_(self.nlm[0].weight, 2 * 3 ** 0.5)
        torch.nn.init.constant_(self.nlm[0].bias, -3 ** 0.5)
        torch.nn.init.constant_(self.nlm[2].weight, 1)
        torch.nn.init.constant_(self.nlm[2].bias, 0)

        for p in self.nlm[2].parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.dimemsion_reduction(x)
        x, _ = self.feature_aggregation(x, self._get_initial_state(x.size(0), x.device))

        # frame quality [B, T]
        q = self.regression(x).squeeze(dim=-1)

        # video overall quality [B, 1]
        relative_score = self._sitp(q)
        mapped_score = self.nlm(F.sigmoid(relative_score))

        return relative_score.squeeze(dim=-1), mapped_score.squeeze(dim=-1)

    def _sitp(self, q, tau=12, beta=0.5):
        """subjectively-inspired temporal pooling"""
        B, T = q.shape

        # [B, tau - 1]
        qm = -float("inf") * torch.ones((B, tau - 1), device=q.device)
        qp = 10000.0 * torch.ones((B, tau - 1), device=q.device)

        # [B, T]
        l = -F.max_pool1d(torch.cat((qm, -q), dim=-1), tau, stride=1)
        m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), dim=-1), tau, stride=1)
        n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), dim=-1), tau, stride=1)

        m = m / n
        q_hat = beta * m + (1 - beta) * l

        return torch.mean(q_hat, dim=-1, keepdim=True)

    def _get_initial_state(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class MDTVSFA(torch.nn.Module):
    def __init__(self, device, model_path, backbone, backbone_path):
        super().__init__()

        self.device = device

        # torch.load inside CNNModel does not specify map_location. This may cause problems with running on cpu
        self.extractor = CNNModel(model=backbone, weights_path=backbone_path).to(self.device).eval()

        state_dict = torch.load(model_path, map_location=self.device)

        self.model = VQAModel().to(self.device).eval()
        self.model.load_state_dict(state_dict, strict=False)

        # CuDNN does not support backward for RNNs in eval mode
        # train / eval affects only dropout inside RNNs, but dropout is 0 by default
        self.model.feature_aggregation.train()

        self.freeze()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = x.to(self.device)

        x = x.unsqueeze(dim=0) if len(x.shape) == 4 else x

        B, T, _, _, _ = x.shape

        # [B * T, C, H, W]
        x = torch.flatten(x, start_dim=0, end_dim=1)

        # [B * T, 4096]
        features = torch.cat(self.extractor(x), dim=1).squeeze(dim=[2, 3])

        # [B, T, 4096]
        features = torch.unflatten(features, dim=0, sizes=(B, T))

        # [B]
        _, mapped_score = self.model(features)

        return torch.mean(mapped_score)
