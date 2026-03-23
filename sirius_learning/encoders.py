import torch
import torch.nn as nn

from active_adaptation.learning.modules import FlattenBatch, FiLM

class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x


class MixedEncoder(nn.Module):
    """
    Empirically, the performance is not very good. Should look for better encoders.
    """
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


class CommandProprioExteroEncoder(nn.Module):
    """
    Fuses command, proprio, and extero observations into a single embedding.

    **Shapes**

    - ``command_inp``: ``(B, D_cmd)`` with ``D_cmd = command_shape[-1]`` (e.g. velocity commands).
    - ``proprio_inp``: ``(B, D_prop)`` with ``D_prop = proprio_shape[-1]``.
    - ``extero_inp``: ``(B, C, H, W)`` matching ``extero_shape`` (last three dims).

    Command passes through a compact MLP; proprio and extero use the same trunk widths as
    :class:`MixedEncoder`. Command and proprio embeddings are concatenated and projected to
    ``out_dim``, then added to the CNN vector and passed through Mish — same fusion pattern as
    ``MixedEncoder`` but with explicit command conditioning.
    """

    def __init__(
        self,
        command_shape: torch.Size,
        proprio_shape: torch.Size,
        extero_shape: torch.Size,
        cmd_hidden: int = 128,
        out_dim: int = 256,
    ):
        super().__init__()
        self.command_shape = command_shape
        self.proprio_shape = proprio_shape
        self.extero_shape = extero_shape
        self.cmd_hidden = cmd_hidden
        self.out_dim = out_dim

        self.command_encoder = nn.Sequential(
            nn.LazyLinear(cmd_hidden),
            nn.Mish(),
            nn.LayerNorm(cmd_hidden),
            nn.LazyLinear(cmd_hidden),
        )
        self.proprio_encoder = nn.Sequential(
            nn.LazyLinear(256),
            nn.Mish(),
            nn.LayerNorm(256),
            nn.LazyLinear(out_dim),
        )
        self.extero_encoder = nn.Sequential(
            FlattenBatch(
                nn.Sequential(
                    nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(),
                    nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(),
                    nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(),
                    nn.Flatten(),
                ),
                data_dim=3,
            ),
            nn.LazyLinear(64),
            nn.Mish(),
            nn.LayerNorm(64),
            nn.LazyLinear(out_dim),
        )
        self.fuse = nn.Sequential(
            nn.LazyLinear(out_dim),
            nn.Mish(),
        )

    def forward(self, command_inp, proprio_inp, extero_inp, mask_cnn=None):
        z_cmd = self.command_encoder(command_inp)
        z_prop = self.proprio_encoder(proprio_inp)
        z_ext = self.cnn_encoder(extero_inp)
        if mask_cnn is not None:
            z_ext = z_ext * mask_cnn
        z_joint = torch.cat([z_cmd, z_prop, z_ext], dim=-1)
        z_fused = self.fuse(z_joint)
        return z_fused

