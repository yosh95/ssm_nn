import torch.nn as nn

from ssm_nn.block import Block


class Model(nn.Module):
    def __init__(self,
                 d_model,
                 d_state,
                 expansion_factor,
                 num_layers,
                 input_size,
                 output_size):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            Block(d_model, d_state, expansion_factor)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.input_linear = nn.Linear(input_size, d_model)
        self.output_linear = nn.Linear(d_model, output_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, x):
        x = self.input_linear(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_linear(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
