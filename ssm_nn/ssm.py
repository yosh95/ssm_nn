import torch
import torch.nn as nn
import torch.nn.functional as F
from ssm_nn.selection import Selection


class SSM(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.select_A = Selection(d_model, d_state * d_state)
        self.select_B = Selection(d_model, d_state * d_model)
        self.select_C = Selection(d_model, d_state * d_model)
        self.select_log_D = Selection(d_model, d_state)
        self.select_delta = Selection(d_model, d_model)

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
        log_D = self.select_log_D(x).view(batch_size,
                                          seq_len,
                                          self.d_state)
        delta = self.select_delta(x)

        delta = F.softplus(delta)
        A = A * torch.exp(log_D).unsqueeze(-1)

        output = self.scan(x, A, B, C, delta)
        return output

    def scan(self, x, A, B, C, delta):
        batch_size, seq_len, d_model = x.shape
        h = torch.zeros(batch_size,
                        self.d_state,
                        1,
                        device=x.device)

        h_seq = []
        for t in range(seq_len):
            h = torch.matmul(A[:, t], h) + \
                torch.matmul(B[:, t], x[:, t].unsqueeze(-1))
            h_seq.append(h)

        h_seq = torch.stack(h_seq, dim=1)
        output = torch.matmul(C.transpose(-2, -1), h_seq)
        output = output.squeeze(-1)

        return output
