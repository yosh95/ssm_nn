import torch
import torch.nn as nn
from ssm_nn.selection import Selection


class SSM(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.select_A = Selection(d_model, d_state * d_state)
        self.select_B = Selection(d_model, d_state * d_model)
        self.select_C = Selection(d_model, d_state * d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        A = self.select_A(x).view(batch_size,
                                  seq_len,
                                  self.d_state,
                                  self.d_state)
        B = self.select_B(x).view(batch_size,
                                  seq_len,
                                  self.d_state,
                                  d_model)
        C = self.select_C(x).view(batch_size,
                                  seq_len,
                                  self.d_state,
                                  d_model)

        h = torch.zeros(batch_size, self.d_state, 1, device=x.device)

        output = []
        for t in range(seq_len):
            h = torch.matmul(A[:, t, :, :], h) + \
                    torch.matmul(B[:, t, :, :], x[:, t, :].unsqueeze(-1))

            y = torch.matmul(C[:, t, :, :].transpose(1, 2), h)
            output.append(y.squeeze(-1))

        output = torch.stack(output, dim=1)
        return output
