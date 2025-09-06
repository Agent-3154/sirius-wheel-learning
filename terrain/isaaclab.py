from isaaclab.terrains import TerrainImporterCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import numpy as np

from dataclasses import MISSING
from .functional import platform_with_slope


@configclass
class PlatformWithSlopeCfg(SubTerrainBaseCfg):
    function = lambda difficulty, cfg: platform_with_slope(
        cfg.size,
        cfg.height_range[0] + np.random.uniform() * (cfg.height_range[1] - cfg.height_range[0])
    )
    height_range: tuple[float, float] = (0.1, 0.2)



ROUGH_TERRAIN_BASE_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=MISSING,
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #     project_uvw=True,
    # ),
    debug_vis=False,
)
