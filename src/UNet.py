import torch
from SinusoidalPositionEmbeddings import SinusoidalPositionEmbeddings
from ResNetBlock import ResNetBlock
from LinearAttention import LinearAttention
from Attention import Attention
import einops.layers.torch


class UNet(torch.nn.Module):
    def __init__(
            self,
            image_channels=3,
            dimensions=[2**6, 2**7, 2**8, 2**9],
            flash_attention=True,
            attention_scheme=(False, False, False, True)
        ):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.image_channels = image_channels
        self.dimensions = dimensions
        self.time_embedding_dimension = self.dimensions[0] * 4

        self.up = torch.nn.ModuleList()
        self.down = torch.nn.ModuleList()

        # determine dimensions
        self.initial_convolution = torch.nn.Conv2d(in_channels=self.image_channels,
                                                   out_channels=self.dimensions[0],
                                                   kernel_size=(7, 7),
                                                   stride=(1, 1),
                                                   padding=(3, 3),
                                                   device=self.device)

        in_out_dimensions = [self.dimensions[0]] + self.dimensions
        in_out_dimensions = list(zip(in_out_dimensions[:-1], in_out_dimensions[1:]))

        # layers
        num_of_iterations = len(in_out_dimensions)

        for index, ((dim_in, dim_out), is_full_attention) in enumerate(zip(in_out_dimensions, attention_scheme)):
            is_last = index >= num_of_iterations - 1

            self.down.append(torch.nn.ModuleList([
                ResNetBlock(dim=dim_in,
                            dim_out=dim_in,
                            time_embedding_dimension=self.time_embedding_dimension),
                ResNetBlock(dim=dim_in,
                            dim_out=dim_in,
                            time_embedding_dimension=self.time_embedding_dimension),
                Attention(dim=dim_in,
                          flash=flash_attention) if is_full_attention else LinearAttention(dim=dim_in),
                self.downsample(dim=dim_in,
                                dim_out=dim_out) if not is_last else torch.nn.Conv2d(in_channels=dim_in,
                                                                                     out_channels=dim_out,
                                                                                     kernel_size=(3, 3),
                                                                                     stride=(1, 1),
                                                                                     padding=(1, 1),
                                                                                     device=self.device)
            ]))

        bottleneck_dimension = self.dimensions[-1]
        self.bottleneck_block1 = ResNetBlock(dim=bottleneck_dimension,
                                             dim_out=bottleneck_dimension,
                                             time_embedding_dimension=self.time_embedding_dimension)
        self.bottleneck_attention = Attention(dim=bottleneck_dimension,
                                              flash=flash_attention)
        self.bottleneck_block2 = ResNetBlock(dim=bottleneck_dimension,
                                             dim_out=bottleneck_dimension,
                                             time_embedding_dimension=self.time_embedding_dimension)

        for index, ((dim_in, dim_out), is_full_attention) in enumerate(reversed(list(zip(in_out_dimensions, attention_scheme)))):
            is_last = index >= num_of_iterations - 1

            self.up.append(torch.nn.ModuleList([
                ResNetBlock(dim=dim_out + dim_in,
                            dim_out=dim_out,
                            time_embedding_dimension=self.time_embedding_dimension),
                ResNetBlock(dim=dim_out + dim_in,
                            dim_out=dim_out,
                            time_embedding_dimension=self.time_embedding_dimension),
                Attention(dim=dim_out,
                          flash=flash_attention) if is_full_attention else LinearAttention(dim=dim_out),
                self.upsample(dim=dim_out,
                              dim_out=dim_in) if not is_last else torch.nn.Conv2d(in_channels=dim_out,
                                                                                  out_channels=dim_in,
                                                                                  kernel_size=(3, 3),
                                                                                  stride=(1, 1),
                                                                                  padding=(1, 1),
                                                                                  device=self.device)
            ]))

        self.final_block = ResNetBlock(dim=self.dimensions[1],
                                       dim_out=self.dimensions[0],
                                       time_embedding_dimension=self.time_embedding_dimension)
        self.final_convolution = torch.nn.Conv2d(in_channels=self.dimensions[0],
                                                 out_channels=self.image_channels,
                                                 kernel_size=(1, 1),
                                                 device=self.device)

        self.time_layer = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(self.dimensions[0]),
            torch.nn.Linear(in_features=self.dimensions[0],
                            out_features=self.time_embedding_dimension,
                            device=self.device),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=self.time_embedding_dimension,
                            out_features=self.time_embedding_dimension,
                            device=self.device)
        )

        self.to(device=self.device,
                non_blocking=True)


    def forward(self, x, timestep):
        x = self.initial_convolution(x)
        r = x.clone()

        time = self.time_layer(timestep)

        residuals = [] # skip connections

        # down
        for block1, block2, attention, downsample in self.down:
            x = block1(x, time)
            residuals.append(x)

            x = block2(x, time)
            x = attention(x) + x
            residuals.append(x)

            x = downsample(x)

        # bottleneck
        x = self.bottleneck_block1(x, time)
        x = self.bottleneck_attention(x) + x
        x = self.bottleneck_block2(x, time)

        # up
        for block1, block2, attention, upsample in self.up:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = block1(x, time)

            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = block2(x, time)
            x = attention(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_block(x, time)
        x = self.final_convolution(x)

        return x


    def upsample(self, dim, dim_out):
        return torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2,
                              mode='nearest'),
            torch.nn.Conv2d(in_channels=dim,
                            out_channels=dim_out,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                            device=self.device)
        )


    def downsample(self, dim, dim_out):
        return torch.nn.Sequential(
            einops.layers.torch.Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w',
                                          p1=2,
                                          p2=2),
            torch.nn.Conv2d(in_channels=dim * 4,
                            out_channels=dim_out,
                            kernel_size=(1, 1),
                            device=self.device)
        )
