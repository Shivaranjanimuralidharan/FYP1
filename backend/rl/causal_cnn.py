import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution
    Ensures no future information leakage
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        padding = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (padding, 0))  # pad only on the left (causal)
        return self.conv(x)


class CausalCNN(nn.Module):
    """
    Causal CNN Encoder for time-series subsequences
    Input  : raw values (batch, seq_len, 1)
    Output : fixed-length embedding (batch, embedding_dim)
    """

    def __init__(
        self,
        input_channels=1,
        hidden_channels=32,
        kernel_size=3,
        num_layers=3,
        embedding_dim=128
    ):
        super().__init__()

        layers = []
        in_ch = input_channels

        for i in range(num_layers):
            layers.append(
                CausalConv1d(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i
                )
            )
            layers.append(nn.ReLU())
            in_ch = hidden_channels

        self.network = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, embedding_dim)

    def forward(self, x):
        """
        x shape: (batch, seq_len, 1)
        """
        x = x.transpose(1, 2)        # → (batch, 1, seq_len)
        x = self.network(x)          # → (batch, hidden, seq_len)
        x = self.global_pool(x)      # → (batch, hidden, 1)
        x = x.squeeze(-1)            # → (batch, hidden)
        x = self.fc(x)               # → (batch, embedding_dim)
        return x
