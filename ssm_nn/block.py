import torch.nn as nn

from ssm_nn.ssm import SSM


class Block(nn.Module):
    def __init__(self, d_model, d_state, expansion_factor, conv_kernel=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expansion_factor
        self.conv = nn.Conv1d(d_model,
                              self.d_inner,
                              kernel_size=conv_kernel,
                              padding=(conv_kernel - 1) // 2)
        self.input_proj = nn.Linear(self.d_inner, self.d_inner)
        self.ssm = SSM(self.d_inner, d_state)
        self.output_proj = nn.Linear(self.d_inner, d_model)
        self.act_f = nn.SiLU()

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_proj(x)
        x = self.act_f(x)
        x = self.ssm(x)
        x = self.output_proj(x)
        return x
