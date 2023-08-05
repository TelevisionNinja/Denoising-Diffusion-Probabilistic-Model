import torch
import torch.backends.cuda
import math


class Attend(torch.nn.Module):
    def __init__(
        self,
        dropout=0,
        flash=True
    ):
        super().__init__()

        cuda_available = torch.cuda.is_available()

        self.device = 'cpu'
        if cuda_available:
            self.device = 'cuda'

        self.dropout = dropout
        self.attention_dropout = torch.nn.Dropout(p=dropout) # inplace=True

        self.flash = flash

        self.cpu_config = {
            'enable_flash': True,
            'enable_math': True,
            'enable_mem_efficient': True
        }
        self.cuda_config = None

        if not cuda_available or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # A100 GPU
            # using flash attention
            self.cuda_config = {
                'enable_flash': True,
                'enable_math': False,
                'enable_mem_efficient': False
            }
        else:
            # non-A100 GPU
            # using math and memory efficient attention
            self.cuda_config = {
                'enable_flash': False,
                'enable_math': True,
                'enable_mem_efficient': True
            }

        self.to(device=self.device,
                non_blocking=True)


    def flash_attention(self, q, k, v):
        q, k, v = map(
            lambda t: t.contiguous(),
            (q, k, v)
        )

        config = self.cuda_config if q.is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(enable_flash=config['enable_flash'],
                                            enable_math=config['enable_math'],
                                            enable_mem_efficient=config['enable_mem_efficient']):

            dropout = 0
            if self.training:
                dropout = self.dropout

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout
            )

        return out


    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        if self.flash:
            return self.flash_attention(q, k, v)

        # similarity
        scale = math.sqrt(q.shape[-1])
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) / scale

        # attention
        attention = sim.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        # aggregate values
        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)

        return out
