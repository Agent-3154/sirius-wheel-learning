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


class SiriusATECCommand(Command):
    def __init__(self, env, teleop: bool = False) -> None:
        super().__init__(env, teleop)
        
        self.asset: Articulation = env.scene["robot"]
        self.contact_sensor: ContactSensor = env.scene["contact_forces"]
        self.terrain = TerrainInterface(self.env.scene.terrain)

        with torch.device(self.device):
            self.target_pos_w = torch.zeros(self.num_envs, 3)
            self.target_vel_w = torch.zeros(self.num_envs, 3)
            self.target_vel_b = torch.zeros(self.num_envs, 3)
            self.target_ang_w = torch.zeros(self.num_envs, 1)
            self.target_yaw_w = torch.zeros(self.num_envs, 3)
            # command params
            self.target_speed = torch.zeros(self.num_envs, 1)
            self.yaw_stiffness = torch.zeros(self.num_envs, 1)
            # self.ref_pos_w = torch.zeros(self.num_envs, 3)
            self.cmd_time_left = torch.zeros(self.num_envs, 1)
            self.cum_pos_error = torch.zeros(self.num_envs, 1)

            if self.terrain.type == "generator":
                subterrain_size = torch.tensor(self.terrain.size)
                adjacent_directions = torch.tensor([
                    [-1., +1.], [0., +1.], [1., +1.], 
                    [-1.,  0.],            [1.,  0.], 
                    [-1., -1.], [0., -1.], [1., -1.]
                ])
                self.adjacent_offsets = subterrain_size * adjacent_directions
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            self.command_angvel = torch.zeros(self.num_envs, 1)
            self.command_linvel = torch.zeros(self.num_envs, 3)
        self.seed = wp.rand_init(0)
    
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
        aux_input = torch.full((self.num_envs, 1), 0.3, device=self.device)
        return torch.cat([
            self.target_vel_b[:, :2],
            self.target_ang_w,
            aux_input
        ],dim=-1)
    
    def symmetry_transforms(self):
        return SymmetryTransform(
            perm=torch.arange(4),
            signs=torch.tensor([1, -1, -1, 1])
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.target_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids]
        # self.ref_pos_w[env_ids] = self.asset.data.root_pos_w[env_ids]

    def update(self):
        self.root_link_pos_w = self.asset.data.root_link_pos_w
        self.root_link_quat_w = self.asset.data.root_link_quat_w
        self.root_heading_w = self.asset.data.heading_w
        
        wp.launch(
            update_command,
            self.num_envs,
            inputs=[
                wp.from_torch(self.root_link_pos_w, return_ctype=True),
                wp.from_torch(self.root_link_quat_w.roll(-1, dims=1), return_ctype=True),
                wp.from_torch(self.root_heading_w, return_ctype=True),
                wp.from_torch(self.adjacent_offsets, return_ctype=True),
                self.seed,
            ],
            outputs=[
                wp.from_torch(self.target_pos_w, return_ctype=True),
                wp.from_torch(self.target_vel_w, return_ctype=True),
                wp.from_torch(self.target_vel_b, return_ctype=True),
                wp.from_torch(self.target_ang_w, return_ctype=True),
                wp.from_torch(self.target_speed, return_ctype=True),
                wp.from_torch(self.yaw_stiffness, return_ctype=True),
                wp.from_torch(self.cmd_time_left, return_ctype=True),
            ]
        )
        self.seed = wp.rand_init(self.seed)
        self.cmd_time_left.sub_(self.env.step_dt)
        self.command_linvel = self.target_vel_b
        self.command_angvel = self.target_ang_w
        
    def debug_draw(self):
        v = self.target_pos_w - self.root_link_pos_w
        v[:, 2] = 0.0
        self.env.debug_draw.vector(self.asset.data.root_pos_w, v, color=(1.0, 0.0, 0.0, 1.0))

        v = self.terrain.get_subterrain_origin(self.asset.data.root_pos_w) - self.asset.data.root_pos_w
        v[:, 2] = 0.0
        self.env.debug_draw.vector(self.asset.data.root_pos_w, v, color=(0.0, 1.0, 0.0, 1.0))


@wp.kernel(enable_backward=False)
def update_command(
    # inputs
    root_link_pos_w: wp.array(dtype=wp.vec3),
    root_link_quat_w: wp.array(dtype=wp.quat),
    root_heading_w: wp.array(dtype=wp.float32),
    adjacent_offsets: wp.array(dtype=wp.vec3),
    seed: wp.int32,
    # outputs
    target_pos_w: wp.array(dtype=wp.vec3),
    target_vel_w: wp.array(dtype=wp.vec3),
    target_vel_b: wp.array(dtype=wp.vec3),
    target_ang_w: wp.array(dtype=wp.float32),
    target_speed: wp.array(dtype=wp.float32),
    yaw_stiffness: wp.array(dtype=wp.float32),
    cmd_time_left: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    seed_ = wp.rand_init(seed, tid)

    pos_diff = target_pos_w[tid] - root_link_pos_w[tid]
    pos_diff.z = 0.0
    pos_error = wp.norm_l2(pos_diff)

    if pos_error < 0.1:
        i = wp.randi(seed_, 0, 8)
        target_pos_w[tid] = target_pos_w[tid] + adjacent_offsets[i]
        target_speed[tid] = wp.randf(seed_, 0.4, 1.8)
        yaw_stiffness[tid] = wp.randf(seed_, 0.8, 1.4)

    if cmd_time_left[tid] <= 0.0:
        cmd_time_left[tid] = wp.randf(seed_, 0.2, 0.4)
        target_vel_w[tid] = wp.normalize(pos_diff) * target_speed[tid]
    
    target_heading = wp.atan2(pos_diff[1], pos_diff[0])
    heading_error = wrap_to_pi(target_heading - root_heading_w[tid])
    target_vel_b[tid] = wp.quat_rotate_inv(root_link_quat_w[tid], target_vel_w[tid])
    target_ang_w[tid] = yaw_stiffness[tid] * heading_error


@wp.func
def wrap_to_pi(angle: wp.float32) -> wp.float32:
    wrapped_angle = (angle + wp.PI) % (2.0 * wp.PI)
    if wrapped_angle == 0.0 and angle > 0.0:
        return wp.PI
    return wrapped_angle - wp.PI


class sirius_atec_vel(Reward[SiriusATECCommand]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
    
    def update(self):
        self.root_lin_vel_w = self.asset.data.root_lin_vel_w
        self.root_lin_vel_b = self.asset.data.root_lin_vel_b

    def compute(self) -> torch.Tensor:
        error = (self.command_manager.target_vel_w - self.root_lin_vel_w)[:, :2]
        error = error.square().sum(dim=-1, keepdim=True)
        rew = torch.exp(- error / 0.25)
        return rew.reshape(self.num_envs, 1)


class sirius_atec_yaw(Reward[SiriusATECCommand]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
    
    def compute(self) -> torch.Tensor:
        target_yaw = 0.0
        yaw_cos = torch.cos(wrap_to_pi(target_yaw - self.asset.data.heading_w))
        return yaw_cos.reshape(self.num_envs, 1)


