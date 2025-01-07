import torch.nn as nn

from ssm_nn.ssm import SSM


class Block(nn.Module):
    def __init__(self, d_model, d_state, expansion_factor):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expansion_factor
        self.input_proj = nn.Linear(d_model, self.d_inner)
        self.ssm = SSM(self.d_inner, d_state)
        self.output_proj = nn.Linear(self.d_inner, d_model)
        self.act_f = nn.SELU()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.act_f(x)
        x = self.ssm(x)
        x = self.output_proj(x)
        return x
