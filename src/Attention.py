import torch
import einops
from RMSNormalization import RMSNormalization
from Attend import Attend


class Attention(torch.nn.Module):
    """
    https://arxiv.org/abs/2306.12929
    """

    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            flash=True,
            number_memory_key_values = 4
        ):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.heads = heads
        hidden_dim = dim_head * self.heads
        self.memory_key_values = torch.nn.Parameter(torch.randn(size=(2, heads, number_memory_key_values, dim_head),
                                                                device=self.device))
        self.to_qkv = torch.nn.Conv2d(in_channels=dim,
                                      out_channels=hidden_dim * 3,
                                      kernel_size=(1, 1),
                                      bias=False,
                                      device=self.device)
        self.to_out = torch.nn.Conv2d(in_channels=hidden_dim,
                                      out_channels=dim,
                                      kernel_size=(1, 1),
                                      device=self.device)

        self.normalize = RMSNormalization(dim=dim)

        self.attend = Attend(flash=flash)

        self.to(device=self.device,
                non_blocking=True)


    def forward(self, x):
        batch, channel, height, width = x.shape

        # normalize
        x = self.normalize(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads),
            qkv
        )

        mk, mv = map(
            lambda t: einops.repeat(t, 'h n d -> b h n d', b=batch),
            self.memory_key_values
        )
        k = torch.cat(tensors=(mk, k), dim=-2)
        v = torch.cat(tensors=(mv, v), dim=-2)

        out = self.attend(q, k, v)

        out = einops.rearrange(out, 'b h (x y) d -> b (h d) x y', x=height, y=width)
        out = self.to_out(out)

        return out
