import torch
import einops
import math
from RMSNormalization import RMSNormalization


class LinearAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.scale = math.sqrt(dim_head)
        self.heads = heads
        hidden_dim = dim_head * heads

        self.normalize = RMSNormalization(dim)
        self.to_qkv = torch.nn.Conv2d(in_channels=dim,
                                      out_channels=hidden_dim * 3,
                                      kernel_size=(1, 1),
                                      bias=False,
                                      device=self.device)
        self.to_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_dim,
                                      out_channels=dim,
                                      kernel_size=(1, 1),
                                      device=self.device),
            RMSNormalization(dim)
        )

        self.to(device=self.device,
                non_blocking=True)


    def forward(self, x):
        batch, channel, height, width = x.shape

        x = self.normalize(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv
        )

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q / self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=height, y=width)
        out = self.to_out(out)

        return out
