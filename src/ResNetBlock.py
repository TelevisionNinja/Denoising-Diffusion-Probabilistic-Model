import torch


class Block(torch.nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            groups=8
        ):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.proj = torch.nn.Conv2d(in_channels=dim,
                                    out_channels=dim_out,
                                    kernel_size=(3, 3),
                                    padding=(1, 1),
                                    device=self.device)
        self.normalize = torch.nn.GroupNorm(num_groups=groups,
                                       num_channels=dim_out,
                                       device=self.device)
        self.activation_function = torch.nn.SiLU() # inplace=True

        self.to(device=self.device,
                non_blocking=True)


    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.normalize(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation_function(x)
        return x


class ResNetBlock(torch.nn.Module):
    """
    https://arxiv.org/abs/1512.03385
    """

    def __init__(
            self,
            dim,
            dim_out,
            time_embedding_dimension,
            groups=8
        ):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.mlp = torch.nn.Sequential(
            torch.nn.SiLU(), # inplace=False
            torch.nn.Linear(in_features=time_embedding_dimension,
                            out_features=dim_out * 2,
                            device=self.device)
        )

        self.block1 = Block(dim=dim,
                            dim_out=dim_out,
                            groups=groups)
        self.block2 = Block(dim=dim_out,
                            dim_out=dim_out,
                            groups=groups)
        self.res_convolution = torch.nn.Conv2d(in_channels=dim,
                                        out_channels=dim_out,
                                        kernel_size=(1, 1),
                                        device=self.device) if dim != dim_out else torch.nn.Identity()

        self.to(device=self.device,
                non_blocking=True)


    def forward(self, x, time_embedding):
        time_embedding = self.mlp(time_embedding)
        time_embedding = time_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1) # extend the last 2 dimensions
        scale_shift = time_embedding.chunk(2, dim = 1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        h = h + self.res_convolution(x)

        return h
