import torch
import warp as wp

from active_adaptation.envs.mdp.base import Command, Reward, Observation, Termination
from active_adaptation.utils.math import (
    clamp_norm,
    normalize,
    quat_rotate,
    quat_rotate_inverse,
    quat_mul,
    sample_quat_yaw,
    quat_from_euler_xyz,
    euler_from_quat,
    wrap_to_pi,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from typing import TYPE_CHECKING
from typing_extensions import override


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor
    from isaaclab.terrains import TerrainImporter


class TerrainInterface:
    def __init__(self, terrain: "TerrainImporter"):
        self.terrain = terrain
        self.type = self.terrain.cfg.terrain_type
        self.num_rows = terrain.cfg.terrain_generator.num_rows
        self.num_cols = terrain.cfg.terrain_generator.num_cols
        self.size = terrain.cfg.terrain_generator.size
        self.terrain_origins = terrain.terrain_origins.reshape(-1, 3)

    def get_subterrain_idx(self, pos_w: torch.Tensor) -> torch.Tensor:
        row = torch.floor(pos_w[:, 0] / self.size[0]).long() + self.num_rows // 2
        row = torch.clamp(row, 0, self.num_rows - 1)
        col = torch.floor(pos_w[:, 1] / self.size[1]).long() + self.num_cols // 2
        col = torch.clamp(col, 0, self.num_cols - 1)
        idx = row * self.num_cols + col
        return idx
    
    def get_subterrain_origin(self, pos_w: torch.Tensor) -> torch.Tensor:
        idx = self.get_subterrain_idx(pos_w)
        origin = self.terrain_origins[idx]
        return origin


class ATEC(Command):
    def __init__(
        self,
        env,
        linvel_x_range: tuple[float, float],
        linvel_y_range: tuple[float, float],
        stand_prob: float = 0.05,
        teleop: bool = False
    ) -> None:
        super().__init__(env, teleop)
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.stand_prob = stand_prob

        self.asset: Articulation = self.env.scene["robot"]
        # self.terrain = TerrainInterface(self.env.scene.terrain)

        with torch.device(self.device):
            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_base_height = torch.zeros(self.num_envs, 1)
            self.command_speed = torch.zeros(self.num_envs, 1)
            self.next_command_linvel = torch.zeros(self.num_envs, 3)

            self.target_yaw_w = torch.zeros(self.num_envs, 1)
            self.target_yaw_vel_w = torch.zeros(self.num_envs, 1)

            self.ref_yaw_w = torch.zeros(self.num_envs, 1)
            self.ref_yaw_vel_w = torch.zeros(self.num_envs, 1)

            # command params
            self.yaw_stiffness = torch.zeros(self.num_envs, 1)
            self.yaw_damping = torch.zeros(self.num_envs, 1)

            self.cmd_time_left = torch.zeros(self.num_envs, 1)
            self.cum_pos_error = torch.zeros(self.num_envs, 1)

            # if self.terrain.type == "generator":
            #     subterrain_size = torch.tensor(self.terrain.size)
            #     adjacent_directions = torch.tensor([
            #         [-1., +1.], [0., +1.], [1., +1.], 
            #         [-1.,  0.],            [1.,  0.], 
            #         [-1., -1.], [0., -1.], [1., -1.]
            #     ])
            #     self.adjacent_offsets = subterrain_size * adjacent_directions
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
        self.seed = wp.rand_init(0)
    
    @property
    def command(self):
        result = torch.cat([
            self.cmd_linvel_b[:, :2],
            wrap_to_pi(self.ref_yaw_w - self.asset.data.heading_w.unsqueeze(1)),
            self.cmd_base_height,
        ], dim=-1)
        return result
    
    def symmetry_transform(self):
        return SymmetryTransform(
            perm=torch.arange(4),
            signs=torch.tensor([1, -1, -1, 1])
        )

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.target_yaw_w[env_ids] = self.asset.data.heading_w[env_ids].unsqueeze(1)
        self.ref_yaw_w[env_ids] = self.asset.data.heading_w[env_ids].unsqueeze(1)
        self.ref_yaw_vel_w[env_ids] = 0.0
        self.sample_command(env_ids)

    @override
    def update(self):
        ref_yaw_acc = (
            self.yaw_stiffness * wrap_to_pi(self.target_yaw_w - self.ref_yaw_w)
            + self.yaw_damping * (self.target_yaw_vel_w - self.ref_yaw_vel_w)
        )
        self.ref_yaw_vel_w.add_(self.env.step_dt * ref_yaw_acc)
        self.ref_yaw_w.add_(self.env.step_dt * self.ref_yaw_vel_w)

        resample_mask = (
            (self.env.episode_length_buf % 200 == 0)
            & (torch.rand(self.num_envs, device=self.device) < 0.75)
        )
        resample_ids = resample_mask.nonzero().squeeze(1)
        if len(resample_ids):
            self.sample_command(resample_ids)
        
        self.cmd_linvel_b.lerp_(self.next_command_linvel, 0.1)
        self.command_speed = self.cmd_linvel_b.norm(dim=-1, keepdim=True)
        self.cmd_linvel_w = quat_rotate(yaw_quat(self.asset.data.root_link_quat_w), self.cmd_linvel_b)
        self.is_standing_env = (self.command_speed < 0.1)
    
    def sample_command(self, env_ids: torch.Tensor):
        self.target_yaw_vel_w[env_ids, 0] = sample_uniform(len(env_ids), -2.0, 2.0, self.device)
        self.yaw_stiffness[env_ids, 0] = 0.0
        self.yaw_damping[env_ids, 0] = 2.0

        next_command_linvel = torch.zeros(len(env_ids), 3, device=self.device)
        next_command_linvel[:, 0].uniform_(*self.linvel_x_range)
        next_command_linvel[:, 1].uniform_(*self.linvel_y_range)
        speed = next_command_linvel.norm(dim=-1, keepdim=True)
        r = torch.rand(len(env_ids), 1, device=self.device) < self.stand_prob
        valid = ~((speed < 0.10) | r)
        self.next_command_linvel[env_ids] = next_command_linvel * valid

        self.cmd_base_height[env_ids] = 0.45
    
    @override
    def debug_draw(self):
        if self.env.backend == "isaac":
            yaw_vec = torch.zeros(self.num_envs, 3, device=self.device)
            yaw_vec[:, 0:1] = self.ref_yaw_w.cos()
            yaw_vec[:, 1:2] = self.ref_yaw_w.sin()

            self.env.debug_draw.vector(
                self.asset.data.root_com_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                yaw_vec,
                color=(1.0, 1.0, 1.0, 1.0),
            )
            self.env.debug_draw.vector(
                self.asset.data.root_link_pos_w
                + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                self.cmd_linvel_w,
                color=(1.0, 1.0, 1.0, 1.0),
            )


def sample_uniform(size, low: float, high: float, device: torch.device = "cuda"):
    return torch.rand(size, device=device) * (high - low) + low


class yaw_cos(Reward[ATEC], namespace="sirius"):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
    
    def compute(self) -> torch.Tensor:
        yaw_diff = wrap_to_pi(self.command_manager.ref_yaw_w - self.asset.data.heading_w.unsqueeze(1))
        yaw_cos = torch.cos(yaw_diff)
        return yaw_cos.reshape(self.num_envs, 1)


