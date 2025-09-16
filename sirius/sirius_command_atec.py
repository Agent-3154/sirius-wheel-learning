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


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor


class SiriusATECCommand(Command):
    def __init__(self, env, teleop: bool = False) -> None:
        super().__init__(env, teleop)
        
        self.asset: Articulation = env.scene["robot"]
        self.contact_sensor: ContactSensor = env.scene["contact_forces"]

        with torch.device(self.device):
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.target_yaw_w = torch.zeros(self.num_envs, 3)

            subterrain_size = torch.tensor(self.env.scene.terrain.cfg.terrain_generator.size)
            adjacent_directions = torch.tensor([
                [-1., +1.], [0., +1.], [1., +1.], 
                [-1.,  0.],            [1.,  0.], 
                [-1., -1.], [0., -1.], [1., -1.]
            ])
            self.adjacent_offsets = subterrain_size * adjacent_directions
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        init_root_state = self.init_root_state[env_ids]
        if self.terrain_type == "plane":
            origins = self.env.scene.env_origins[env_ids]
        else:
            idx = torch.randint(0, len(self._origins), (len(env_ids),), device=self.device)
            origins = self._origins[idx]
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = quat_mul(
            init_root_state[:, 3:7],
            sample_quat_yaw(len(env_ids), device=self.device)
        )
        return init_root_state
    
    @property
    def command(self):
        target_pos_b = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_quat_w),
            self.target_pos_w - self.asset.data.root_pos_w
        )
        target_yaw = wrap_to_pi(torch.pi / 2 - self.asset.data.heading_w)
        return torch.cat([target_pos_b[:, :2], target_yaw[:, None]], dim=-1)
    
    def symmetry_transforms(self):
        return SymmetryTransform(
            perm=torch.arange(3),
            signs=torch.tensor([1, -1, -1])
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.target_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids] + torch.tensor([2.0, 0.0, 0.0], device=self.device)

    def update(self):
        diff = self.target_pos_w - self.asset.data.root_pos_w
        self.target_pos_w = torch.where(
            diff.norm(dim=-1, keepdim=True) < 0.1,
            self.asset.data.root_pos_w + torch.tensor([2.0, 0.0, 0.0], device=self.device),
            self.target_pos_w
        )
    
    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w,
            self.target_pos_w - self.asset.data.root_pos_w,
            color=(1.0, 0.0, 0.0, 1.0),
        )


class sirius_atec_vel(Reward[SiriusATECCommand]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
    
    def compute(self) -> torch.Tensor:
        target_direction = normalize(self.command_manager.target_pos_w - self.asset.data.root_pos_w)
        rew = torch.sum(self.asset.data.root_lin_vel_w * target_direction, dim=-1, keepdim=True)
        rew = rew.clamp_max(1.5)
        return rew.reshape(self.num_envs, 1)


class sirius_atec_yaw(Reward[SiriusATECCommand]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
    
    def compute(self) -> torch.Tensor:
        target_yaw = 0.0
        yaw_cos = torch.cos(wrap_to_pi(target_yaw - self.asset.data.heading_w))
        return yaw_cos.reshape(self.num_envs, 1)


