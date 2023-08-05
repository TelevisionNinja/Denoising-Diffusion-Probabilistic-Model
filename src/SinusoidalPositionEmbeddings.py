import torch
import math


class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.dimension = dimension

        self.to(device=self.device,
                non_blocking=True)


    def forward(self, time):
        half_dimension = self.dimension // 2

        embeddings = math.log(10000) / (half_dimension - 1)
        embeddings = torch.exp(torch.arange(half_dimension, device=self.device) * -embeddings)
        embeddings = time.unsqueeze(dim=1) * embeddings.unsqueeze(dim=0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings
