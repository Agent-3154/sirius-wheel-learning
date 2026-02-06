import torch
from typing_extensions import override
from active_adaptation.envs.mdp.base import Command, Reward
from active_adaptation.utils.math import quat_rotate, sample_quat_yaw, wrap_to_pi, quat_rotate_inverse, yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform
from isaaclab.terrains import TerrainImporter


class ATECTaskDCommand(Command):
    def __init__(
        self,
        env,
        curriculum: bool = False,
        teleop: bool = False,
    ) -> None:
        super().__init__(env, teleop)
        self.angvel_range = (-2.0, 2.0)
        self.curriculum = curriculum and self.env.backend == "isaac"
        
        self.terrain: TerrainImporter = self.env.scene.terrain
        assert self.terrain.cfg.terrain_type == "generator", "Curriculum is only supported for generator terrain"
        assert self.terrain.cfg.terrain_generator.curriculum, "Curriculum is not enabled for the terrain"

        with torch.device(self.device):
            self.cmd_type = torch.zeros(self.num_envs, 1, dtype=torch.int32)

            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
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
            self.ref_height_offset = torch.stack([
                torch.linspace(0.6, 0.8, 10),
                torch.zeros(10),
                torch.zeros(10),
            ], dim=1)
        
        self.update()
        
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, sim_utils
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Command/ref_height",
                    markers={
                        "scandot": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.8)),
                        ),
                    }
                )
            )
            self.marker.set_visibility(True)
    
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
        if self.curriculum and self.env.episode_count > 1:
            distance_traveled = self.distance_traveled[env_ids]
            distance_commanded = self.distance_commanded[env_ids].clamp_min(1.0)
            move_up = distance_traveled > distance_commanded * 0.8
            move_down = distance_traveled < distance_commanded * 0.4
            move_up = move_up & ~move_down
            self.terrain.update_env_origins(env_ids, move_up.squeeze(-1), move_down.squeeze(-1))
            self.env.extra["curriculum/terrain_level"] = self.terrain.terrain_levels.float().mean()
        origins = self.terrain.env_origins[env_ids]
        init_root_state = self.init_root_state[env_ids]
        init_root_state[:, :3] += origins
        init_root_state[:, 1] = torch.rand(len(env_ids), device=self.device) * 1.8 - 0.9
        init_root_state[:, 3:7] = sample_quat_yaw(len(env_ids), device=self.device)
        self.env.extra["curriculum/distance_commanded"] = self.distance_commanded.mean()
        self.env.extra["curriculum/distance_traveled"] = self.distance_traveled.mean()
        self.distance_commanded[env_ids] = 0.0
        self.distance_traveled[env_ids] = 0.0
        return init_root_state

    @override
    def reset(self, env_ids):
        self.sample_command_1(env_ids)
    
    def sample_command_0(self, env_ids: torch.Tensor):
        self.cmd_type[env_ids] = 0
        cmd_linvel_b = torch.zeros(len(env_ids), 3, device=self.device)
        cmd_linvel_b[:, 0].uniform_(-1.4, 1.4)
        cmd_linvel_b[:, 1].uniform_(-1.0, 1.0)
        self.cmd_linvel_b[env_ids] = cmd_linvel_b

    def sample_command_1(self, env_ids: torch.Tensor):
        self.cmd_type[env_ids] = 1
        cmd_linvel_w = torch.zeros(len(env_ids), 3, device=self.device)
        cmd_linvel_w[:, 0].uniform_(0.6, 1.4)
        cmd_linvel_w[:, 1].uniform_(-0.1, 0.1)
        self.cmd_linvel_w[env_ids] = cmd_linvel_w

    @override
    def update(self):
        # resample_mask = (
        #     (self.env.episode_length_buf % 100 == 0)
        #     & (torch.rand(self.num_envs, device=self.device) < 0.5)
        #     & (self.cmd_type == 0).squeeze(1)
        # )
        # resample_ids = resample_mask.nonzero().squeeze(1)
        # if len(resample_ids):
        #     self.sample_command_1(resample_ids)

        self.body_heading_w = self.asset.data.heading_w.unsqueeze(1)
        yaw_diff = wrap_to_pi(self.target_yaw - self.body_heading_w).reshape(self.num_envs, 1)
        self.cmd_yawvel_b = torch.clamp(
            self.yaw_stiffness * yaw_diff,
            min=self.angvel_range[0],
            max=self.angvel_range[1],
        )

        # if cmd_type is 0, use cmd_linvel_b and derive cmd_linvel_w
        # if cmd_type is 1, use cmd_linvel_w and derive cmd_linvel_b
        quat = yaw_quat(self.asset.data.root_link_quat_w)
        self.cmd_linvel_w = torch.where(
            self.cmd_type == 0,
            quat_rotate(quat, self.cmd_linvel_b),
            self.cmd_linvel_w,
        )
        self.cmd_linvel_b = torch.where(
            self.cmd_type == 0,
            self.cmd_linvel_b,
            quat_rotate_inverse(quat, self.cmd_linvel_w),
        )
        assert self.cmd_linvel_b.shape == self.cmd_linvel_w.shape == (self.num_envs, 3)
        self.command_speed = self.cmd_linvel_b.norm(dim=-1, keepdim=True)
        self.current_speed = self.asset.data.root_com_lin_vel_w.norm(dim=-1, keepdim=True)
        self.distance_commanded = self.distance_commanded + self.command_speed * self.env.step_dt
        self.distance_traveled = self.distance_traveled + self.current_speed * self.env.step_dt

        self.ref_height_baseline = self.env.get_ground_height_at(
            self.asset.data.root_link_pos_w.unsqueeze(1) + self.ref_height_offset
        ).reshape(self.num_envs, 10)
        self.ref_height_w = self.ref_height_baseline.mean(1, keepdim=True) + 0.4
    
    @override
    def debug_draw(self):
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            dots = self.asset.data.root_link_pos_w.unsqueeze(1) + self.ref_height_offset
            dots[:, :, 2] = self.ref_height_baseline
            self.marker.visualize(dots.reshape(-1, 3))


class free_walk(Reward[ATECTaskDCommand], namespace="atec"):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
    
    @override
    def compute(self) -> torch.Tensor:
        root_linvel_w = self.asset.data.root_com_lin_vel_w
        return root_linvel_w[:, 0].reshape(self.num_envs, 1)


class get_over_platform(Reward[ATECTaskDCommand]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
        self.head_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.head_linvel_w = torch.zeros(self.num_envs, 3, device=self.device)
    
    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.head_pos_w[env_ids] = self.asset.data.root_link_pos_w[env_ids] + quat_rotate(
            self.asset.data.root_link_quat_w[env_ids],
            torch.tensor([[0.7, 0.0, 0.0]], device=self.device)
        )

    @override
    def update(self) -> None:
        head_pos_w = self.asset.data.root_link_pos_w + quat_rotate(
            self.asset.data.root_link_quat_w,
            torch.tensor([[0.7, 0.0, 0.0]], device=self.device)
        )
        self.head_linvel_w = (self.head_pos_w - head_pos_w) / self.env.step_dt
        self.head_pos_w = head_pos_w
    
    @override
    def compute(self) -> torch.Tensor:
        rew = (self.head_pos_w[:, 2:3] - self.command_manager.ref_height_w).clamp_max(0.0)
        valid = self.command_manager.ref_height_w > self.head_pos_w[:, 2:3]
        valid = valid & (self.command_manager.cmd_type == 1)
        return rew.reshape(self.num_envs, 1), valid.reshape(self.num_envs, 1)
