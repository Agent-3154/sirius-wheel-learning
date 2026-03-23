import torch
import torch.nn as nn

from active_adaptation.learning.modules import FlattenBatch


class MixedEncoder(nn.Module):
    def __init__(self, proprio_shape: torch.Size, terrain_shape: torch.Size):
        super().__init__()

        self.mlp_encoder = nn.Sequential(
            nn.LazyLinear(256), nn.Mish(), nn.LayerNorm(256), 
            nn.LazyLinear(256)
        )
        self.cnn_encoder = nn.Sequential(
            FlattenBatch(
                nn.Sequential(
                    nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1), 
                    nn.Mish(), # nn.GroupNorm(num_channels=2, num_groups=2),
                    nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=4, num_groups=2),
                    nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=8, num_groups=2), 
                    nn.Flatten(),
                ),
                data_dim=3,
            ),
            nn.LazyLinear(64),
            nn.Mish(),
            nn.LayerNorm(64),
            nn.LazyLinear(256)
        )
        self.out = nn.Mish()

    def forward(self, mlp_inp, cnn_inp, mask_cnn=None):
        cnn_feature = self.cnn_encoder(cnn_inp)
        mlp_feature = self.mlp_encoder(mlp_inp)
        if mask_cnn is not None:
            cnn_feature = cnn_feature * mask_cnn
        feature = mlp_feature + cnn_feature
        return self.out(feature)

