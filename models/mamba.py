from mamba_ssm.modules.mamba_simple import Block
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaModel(nn.Module):
    def __init__(self, num_blocks: int, d_model: int,d_model_out:int, d_state: int = 16, d_conv: int = 3,
                 expand: int = 2):
        super().__init__()
        self.mamba_blocks_middle = nn.ModuleList([
            Block(
                dim=d_model,
                mixer_cls=lambda dim: Mamba(dim, dim, d_state=d_state, d_conv=d_conv, expand=expand),
                norm_cls=nn.LayerNorm,
                fused_add_norm=False,
                residual_in_fp32=False
            ) for _ in range(num_blocks - 1)

        ])
        self.final_mamba_block = Block(
                dim=d_model,
                mixer_cls=lambda dim: Mamba(dim, d_model_out, d_state=d_state, d_conv=d_conv, expand=expand),
                norm_cls=nn.LayerNorm,
                fused_add_norm=False,
                residual_in_fp32=False
            )

    def forward(self, x):
        residual = None
        for block in self.mamba_blocks_middle:
            x, residual = block(x, residual)
        out,_ = self.final_mamba_block(x,residual)
        return out

if __name__=="__main__":
    device = torch.device('cuda')
    model = MambaModel(num_blocks = 20,d_model=316,d_model_out=27377,d_state=512,d_conv=4,expand=2).to(device)
    print("total param:{}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    x= torch.randn(5,2,316)
    x = x.to(device)
    y = model(x)
    print(y.shape)
