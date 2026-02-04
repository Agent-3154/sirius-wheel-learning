import torch
from typing_extensions import override
from active_adaptation.envs.mdp.base import Command, Reward
from active_adaptation.utils.math import sample_quat_yaw, wrap_to_pi, quat_rotate_inverse, yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform


class ATECTaskDCommand(Command):
    def __init__(self, env, teleop: bool = False) -> None:
        super().__init__(env, teleop)
        self.angvel_range = (-2.0, 2.0)

        with torch.device(self.device):
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_b: torch.Tensor # derived from cmd_linvel_w
            self.command_speed: torch.Tensor # derived from cmd_linvel_w
            self.cmd_base_height = torch.zeros(self.num_envs, 1)
            self.cmd_base_height.fill_(0.3)
            self.target_yaw = torch.zeros(self.num_envs, 1)
            self.yaw_stiffness = torch.zeros(self.num_envs, 1)
            self.yaw_stiffness.fill_(1.0)

            # curriculum
            self.distance_commanded = torch.zeros(self.num_envs, 1)
            self.distance_traveled = torch.zeros(self.num_envs, 1)
            
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
        self.update()
    
    @property
    def command(self):
        return torch.cat([
            self.cmd_linvel_b[:, :2],
            self.cmd_yawvel_b.reshape(self.num_envs, 1),
            self.cmd_base_height.reshape(self.num_envs, 1),
        ], dim=-1)
    
    @override
    def symmetry_transform(self):
        # left-right symmetry: flip y velocity and yaw velocity
        transform = SymmetryTransform(perm=torch.arange(4), signs=[1, -1, -1, 1])
        return transform

    @override
    def sample_init(self, env_ids):
        idx = torch.randint(0, len(self._origins), (len(env_ids),), device=self.device)
        origins = self._origins[idx]
        # randomize the y position of the origin
        # origins[:, 1] += torch.randn(len(env_ids), device=self.device).clamp(-3., 3.)
        init_root_state = self.init_root_state[env_ids]
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = sample_quat_yaw(len(env_ids), device=self.device)
        self.env.extra["curriculum/distance_commanded"] = self.distance_commanded.mean()
        self.env.extra["curriculum/distance_traveled"] = self.distance_traveled.mean()
        self.distance_commanded[env_ids] = 0.0
        self.distance_traveled[env_ids] = 0.0
        return init_root_state

    @override
    def reset(self, env_ids):
        cmd_linvel_w = torch.zeros(len(env_ids), 3, device=self.device)
        cmd_linvel_w[:, 0].uniform_(0.3, 1.4)
        cmd_linvel_w[:, 1].uniform_(-0.5, 0.5)
        self.cmd_linvel_w[env_ids] = cmd_linvel_w

    @override
    def update(self):
        self.body_heading_w = self.asset.data.heading_w.unsqueeze(1)
        yaw_diff = wrap_to_pi(self.target_yaw - self.body_heading_w).reshape(self.num_envs, 1)
        self.cmd_yawvel_b = torch.clamp(
            self.yaw_stiffness * yaw_diff,
            min=self.angvel_range[0],
            max=self.angvel_range[1],
        )

        self.cmd_linvel_b = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_link_quat_w),
            self.cmd_linvel_w
        )
        self.command_speed = self.cmd_linvel_b.norm(dim=-1, keepdim=True)
        self.current_speed = self.asset.data.root_com_lin_vel_w.norm(dim=-1, keepdim=True)
        self.distance_commanded = self.distance_commanded + self.command_speed * self.env.step_dt
        self.distance_traveled = self.distance_traveled + self.current_speed * self.env.step_dt

        self.ref_height = self.env.get_ground_height_at(
            self.asset.data.root_link_pos_w + torch.tensor([0.7, 0.0, 0.0], device=self.device)
        ).reshape(self.num_envs, 1) + 0.4


class get_over_platform(Reward[ATECTaskDCommand]):
    
    @override
    def compute(self) -> torch.Tensor:
        root_pos_w = self.command_manager.asset.data.root_pos_w
        root_linvel_z = self.command_manager.asset.data.root_lin_vel_w[:, 2].reshape(self.num_envs, 1)
        root_height = (root_pos_w[:, 2] - self.env.get_ground_height_at(root_pos_w)).reshape(self.num_envs, 1)
        cmd_linvel_z = (self.command_manager.ref_height - root_height).clamp_min(0.0)
        return (root_linvel_z - cmd_linvel_z).square().reshape(self.num_envs, 1)
