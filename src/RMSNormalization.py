import torch
import math


class RMSNormalization(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.g = torch.nn.Parameter(torch.ones(size=(1, dim, 1, 1),
                                               device=self.device))

        self.to(device=self.device,
                non_blocking=True)

    def forward(self, x):
        return torch.nn.functional.normalize(input=x, dim=1) * self.g * math.sqrt(x.shape[1])
