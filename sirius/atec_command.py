import torch
from typing_extensions import override, TYPE_CHECKING
from active_adaptation.envs.mdp.base import Command, Reward
from active_adaptation.utils.math import (
    quat_rotate,
    sample_quat_yaw,
    wrap_to_pi,
    quat_rotate_inverse,
    yaw_quat,
    clamp_norm,
)
from active_adaptation.utils.symmetry import SymmetryTransform

if TYPE_CHECKING:
    from active_adaptation.envs.terrain import BetterTerrainImporter, BetterTerrainGenerator

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
        
        self.terrain: BetterTerrainImporter = self.env.scene.terrain
        assert self.terrain.cfg.terrain_type == "generator", "Curriculum is only supported for generator terrain"
        # assert self.terrain.cfg.terrain_generator.curriculum, "Curriculum is not enabled for the terrain"
        self.terrain_generator: BetterTerrainGenerator = self.terrain.terrain_generator
        
        self.sub_terrain_types = self.terrain_generator.sub_terrain_types
        self.sub_terrain_type_mapping = self.terrain_generator.sub_terrain_type_mapping
        self.num_cols = self.terrain_generator.num_cols
        # For pit_and_platform with left_right_ratio=(1,0): platform at 0.25, box at 0.75; origin at (0.15*sx, 0.5*sy, 0)
        terrain_size = self.terrain_generator.cfg.size
        self._step_target_offset = torch.tensor(
            [0.35 * terrain_size[0], 0.25 * terrain_size[1], 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        self._pit_and_platform_idx = self.sub_terrain_type_mapping.get("pit_and_platform", -1)

        with torch.device(self.device):
            self.cmd_type = torch.zeros(self.num_envs, 1, dtype=torch.int32)
            self.pit_cmd_speed = torch.zeros(self.num_envs, 1)

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
            self.step_target_w = torch.zeros(self.num_envs, 3, device=self.device)

        if self.teleop and self.env.backend == "isaac":
            self._teleop_linvel = torch.zeros(3, device=self.device)
            self._teleop_yaw = torch.zeros(1, device=self.device)
            self._speed_scale = 0.8
            self._fast_speed_scale = 1.6
            self._slow_speed_scale = 0.4
            self.key_mappings_linvel = {
                "W": torch.tensor([1.4, 0.0, 0.0], device=self.device),
                "S": torch.tensor([-1.4, 0.0, 0.0], device=self.device),
                "A": torch.tensor([0.0, 1.0, 0.0], device=self.device),
                "D": torch.tensor([0.0, -1.0, 0.0], device=self.device),
            }
            self.key_mappings_yaw = {
                "LEFT": torch.tensor([self.angvel_range[1]], device=self.device),
                "RIGHT": torch.tensor([self.angvel_range[0]], device=self.device),
            }
            from active_adaptation.utils.isaac_keyboard import IsaacKeyboardManager
            self.keyboard_manager = IsaacKeyboardManager()

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
            self.step_target_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/step_target",
                    markers={
                        "target": sim_utils.SphereCfg(
                            radius=0.08,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.0)),
                        ),
                    }
                )
            )
            self.step_target_marker.set_visibility(True)
    
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
        # if self.curriculum and self.env.episode_count > 1:
        #     distance_traveled = self.distance_traveled[env_ids]
        #     distance_commanded = self.distance_commanded[env_ids].clamp_min(1.0)
        #     move_up = distance_traveled > distance_commanded * 0.8
        #     move_down = distance_traveled < distance_commanded * 0.4
        #     move_up = move_up & ~move_down
        #     self.terrain.update_env_origins(env_ids, move_up.squeeze(-1), move_down.squeeze(-1))
        #     self.env.extra["curriculum/terrain_level"] = self.terrain.terrain_levels.float().mean()
        # sample robot initial state
        origins = self.terrain.env_origins[env_ids]
        init_root_state = self.init_root_state[env_ids]
        init_root_state[:, :3] += origins
        # init_root_state[:, 1] = torch.rand(len(env_ids), device=self.device) * 1.8 - 0.9
        init_root_state[:, 3:7] = sample_quat_yaw(len(env_ids), device=self.device)
        self.env.extra["curriculum/distance_commanded"] = self.distance_commanded.mean()
        self.env.extra["curriculum/distance_traveled"] = self.distance_traveled.mean()
        self.distance_commanded[env_ids] = 0.0
        self.distance_traveled[env_ids] = 0.0
        # sample object initial state
        if self.env.scene.get("box", None) is not None:
            init_box_state = torch.zeros(len(env_ids), 7 + 6, device=self.device)
            init_box_state[:, :3] = origins + torch.tensor([[1.5, 0.0, 0.2]], device=self.device)
            init_box_state[:, 3:7] = sample_quat_yaw(len(env_ids), device=self.device)
            return {"robot": init_root_state, "box": init_box_state}
        else:
            return {"robot": init_root_state}

    @override
    def reset(self, env_ids):
        pit_cmd_speed = torch.empty(len(env_ids), 1, device=self.device)
        pit_cmd_speed.uniform_(0.4, 1.2)
        self.pit_cmd_speed[env_ids] = pit_cmd_speed

    @override
    def update(self):
        if self.teleop:
            self._update_teleop()
            return
        self._update_training()

    def _update_teleop(self):
        if self.env.backend != "isaac":
            self._update_training()
            return
        km = self.keyboard_manager.key_pressed
        if km.get("LEFT_SHIFT") or km.get("RIGHT_SHIFT"):
            scale = self._fast_speed_scale
        elif km.get("LEFT_CONTROL") or km.get("RIGHT_CONTROL"):
            scale = self._slow_speed_scale
        else:
            scale = self._speed_scale
        self._teleop_linvel.zero_()
        for key, vel in self.key_mappings_linvel.items():
            if km.get(key, False):
                self._teleop_linvel.add_(vel)
        self._teleop_yaw.zero_()
        for key, vel in self.key_mappings_yaw.items():
            if km.get(key, False):
                self._teleop_yaw.add_(vel)
        linvel = (self._teleop_linvel * scale).unsqueeze(0).expand(self.num_envs, -1)
        linvel[:, 2] = 0.0
        max_speed = max(0.0, 2.5 - self._teleop_yaw.abs().item())
        self.cmd_linvel_b = clamp_norm(linvel, max=max_speed)
        self.cmd_yawvel_b[:] = (self._teleop_yaw * scale).clamp(*self.angvel_range)
        self.cmd_base_height[:] = 0.3
        quat = yaw_quat(self.asset.data.root_link_quat_w)
        self.cmd_linvel_w = quat_rotate(quat, self.cmd_linvel_b)
        self.command_speed = self.cmd_linvel_b.norm(dim=-1, keepdim=True)
        self.current_speed = self.asset.data.root_com_lin_vel_w.norm(dim=-1, keepdim=True)
        self.distance_commanded = self.distance_commanded + self.command_speed * self.env.step_dt
        self.distance_traveled = self.distance_traveled + self.current_speed * self.env.step_dt
        self.ref_height_baseline = self.env.get_ground_height_at(
            self.asset.data.root_link_pos_w.unsqueeze(1) + self.ref_height_offset
        ).reshape(self.num_envs, 10)
        self.ref_height_w = self.ref_height_baseline.mean(1, keepdim=True) + 0.4

    def _update_training(self):
        # For pit_and_platform terrains: command (vx, vy, v_yaw) toward the virtual step target
        # (where the box would be) so the robot learns to cross the gap by stepping on it.
        root_pos_w = self.asset.data.root_link_pos_w
        terrain_origin_ids = self.terrain_generator.get_terrain_origin_id(root_pos_w)
        terrain_types = self.sub_terrain_types.to(self.device)[terrain_origin_ids]
        origins_flat = self.terrain.terrain_origins.reshape(-1, 3).to(self.device)
        terrain_origins = origins_flat[terrain_origin_ids]
        is_pit_env = (terrain_types == self._pit_and_platform_idx).unsqueeze(1)

        self.step_target_w = terrain_origins + self._step_target_offset
            
        delta_xy = self.step_target_w[:, :2] - root_pos_w[:, :2]
        dist_to_target = delta_xy.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        past_box = (root_pos_w[:, 0:1] > self.step_target_w[:, 0:1] + 0.2)
        # Phase 1: velocity toward step target; Phase 2: forward (+x)
        dir_xy = delta_xy / dist_to_target
        cmd_linvel_pit = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_linvel_pit[:, :2] = dir_xy * self.pit_cmd_speed
        cmd_linvel_forward = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_linvel_forward[:, 0:1] = self.pit_cmd_speed
        desired_vel = torch.where(past_box, cmd_linvel_forward, cmd_linvel_pit)
        target_yaw_pit = torch.atan2(delta_xy[:, 1:2], delta_xy[:, 0:1])
        target_yaw_forward = torch.zeros(self.num_envs, 1, device=self.device)
        desired_yaw = torch.where(past_box[:, :1], target_yaw_forward, target_yaw_pit)
        self.cmd_type = torch.where(is_pit_env, torch.ones_like(self.cmd_type), self.cmd_type)
        self.cmd_linvel_w = torch.where(is_pit_env.expand(-1, 3), desired_vel, self.cmd_linvel_w)
        self.target_yaw = torch.where(is_pit_env, desired_yaw, self.target_yaw)

        self.body_heading_w = self.asset.data.heading_w.unsqueeze(1)
        yaw_diff = wrap_to_pi(self.target_yaw - self.body_heading_w).reshape(self.num_envs, 1)
        self.cmd_yawvel_b = torch.clamp(
            self.yaw_stiffness * yaw_diff,
            min=self.angvel_range[0],
            max=self.angvel_range[1],
        )

        # Derive body-frame velocity and curriculum / ref height
        quat = yaw_quat(self.asset.data.root_link_quat_w)
        self.cmd_linvel_b = quat_rotate_inverse(quat, self.cmd_linvel_w)
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
            self.env.debug_draw.vector(
                self.asset.data.root_link_pos_w,
                self.step_target_w - self.asset.data.root_link_pos_w,
                color=(1.0, 0.4, 0.0, 1.0),
            )
            self.step_target_marker.visualize(self.step_target_w)


class free_walk(Reward[ATECTaskDCommand]):
    namespace = "atec"
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
    
    @override
    def compute(self) -> torch.Tensor:
        root_linvel_w = self.asset.data.root_com_lin_vel_w
        rew = root_linvel_w[:, 0:1].clamp_max(self.command_manager.command_speed)
        return rew.reshape(self.num_envs, 1)


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
