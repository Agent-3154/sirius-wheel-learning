import active_adaptation
from active_adaptation.registry import Registry

from isaaclab.terrains import TerrainImporterCfg, SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import numpy as np

from dataclasses import MISSING
from .functional import platform_with_slope

registry = Registry.instance()

@configclass
class PlatformWithSlopeCfg(SubTerrainBaseCfg):
    function = lambda difficulty, cfg: platform_with_slope(
        cfg.size,
        cfg.height_range[0] + np.random.uniform() * (cfg.height_range[1] - cfg.height_range[0])
    )
    height_range: tuple[float, float] = (0.1, 0.2)


SIRIUS_DEMO = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "platform_with_slope": PlatformWithSlopeCfg(
            proportion=0.5,
            height_range=(0.1, 0.3),
        ),
    },
)


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

registry.register("terrain", "sirius_demo", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=SIRIUS_DEMO))