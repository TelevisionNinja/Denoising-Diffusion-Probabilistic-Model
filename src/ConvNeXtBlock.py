import torch


class ConvNeXtBlock1(torch.nn.Module):
    """
    https://arxiv.org/abs/2201.03545
    """

    def __init__(
            self,
            dim,
            dim_out,
            time_embedding_dimension,
            mult=2,
            norm=True
        ):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.mlp = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(in_features=time_embedding_dimension,
                            out_features=dim)
        )

        self.ds_convolution = torch.nn.Conv2d(in_channels=dim,
                                       out_channels=dim,
                                       kernel_size=(7, 7),
                                       padding=(3, 3),
                                       groups=dim,
                                       device=self.device)

        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=1,
                               num_channels=dim,
                               device=self.device) if norm else torch.nn.Identity(),
            torch.nn.Conv2d(in_channels=dim,
                            out_channels=dim_out * mult,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            device=self.device),
            torch.nn.GELU(),
            torch.nn.GroupNorm(num_groups=1,
                               num_channels=dim_out * mult,
                               device=self.device),
            torch.nn.Conv2d(in_channels=dim_out * mult,
                            out_channels=dim_out,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            device=self.device)
        )

        self.res_convolution = torch.nn.Conv2d(in_channels=dim,
                                        out_channels=dim_out,
                                        kernel_size=(1, 1),
                                        device=self.device) if dim != dim_out else torch.nn.Identity()

        self.to(device=self.device,
                non_blocking=True)


    def forward(self, x, time_embedding):
        h = self.ds_convolution(x)

        time_embedding = self.mlp(time_embedding)
        time_embedding = time_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1) # extend the last 2 dimensions
        h = h + time_embedding

        h = self.net(h)
        h = h + self.res_convolution(x)

        return h


from RMSNormalization import RMSNormalization


class ConvNeXtBlock2(torch.nn.Module):
    """
    https://arxiv.org/abs/2201.03545
    """

    def __init__(
            self,
            dim,
            dim_out,
            time_embedding_dimension,
            mult=2,
            norm=True
        ):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.mlp = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(in_features=time_embedding_dimension,
                            out_features=dim)
        )

        self.ds_convolution = torch.nn.Conv2d(in_channels=dim,
                                       out_channels=dim,
                                       kernel_size=(7, 7),
                                       padding=(3, 3),
                                       groups=dim,
                                       device=self.device)

        self.net = torch.nn.Sequential(
            RMSNormalization(dim=dim) if norm else torch.nn.Identity(),
            torch.nn.Conv2d(in_channels=dim,
                            out_channels=dim_out * mult,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            device=self.device),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=dim_out * mult,
                            out_channels=dim_out,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            device=self.device)
        )

        self.res_convolution = torch.nn.Conv2d(in_channels=dim,
                                        out_channels=dim_out,
                                        kernel_size=(1, 1),
                                        device=self.device) if dim != dim_out else torch.nn.Identity()

        self.to(device=self.device,
                non_blocking=True)


    def forward(self, x, time_embedding):
        h = self.ds_convolution(x)

        time_embedding = self.mlp(time_embedding)
        time_embedding = time_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1) # extend the last 2 dimensions
        h = h + time_embedding

        h = self.net(h)
        h = h + self.res_convolution(x)

        return h
