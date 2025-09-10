import active_adaptation
from active_adaptation.registry import Registry

from isaaclab.terrains import (
    TerrainImporterCfg,
    SubTerrainBaseCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
)
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import numpy as np
import random

from dataclasses import MISSING
from .functional import platform_with_slope, pallet_with_platform

registry = Registry.instance()


@configclass
class PlatformWithSlopeCfg(SubTerrainBaseCfg):
    function = lambda difficulty, cfg: platform_with_slope(
        cfg.size,
        np.random.uniform(cfg.height_range[0], cfg.height_range[1])
    )
    height_range: tuple[float, float] = (0.1, 0.2)


@configclass
class PalletWithPlatformCfg(SubTerrainBaseCfg):
    function = lambda difficulty, cfg: pallet_with_platform(
        cfg.size,
        platform_width=cfg.platform_width,
        board_width=random.uniform(cfg.board_width_range[0], cfg.board_width_range[1]),
        num_stringers=cfg.num_stringers,
        interval=random.uniform(cfg.interval_range[0], cfg.interval_range[1]),
        rotate=random.random() < 0.5,
    )
    platform_width: float = 1.0
    board_width_range: tuple[float, float] = (0.1, 0.2)
    interval_range: tuple[float, float] = (0.05, 0.1)
    num_stringers: int = 4


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


SIRIUS_ATEC = TerrainGeneratorCfg(
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
        "plane": MeshPlaneTerrainCfg(
            proportion=0.5,
        ),
        "pallet_with_platform": PalletWithPlatformCfg(
            proportion=0.5,
            platform_width=2.0,
            board_width_range=(0.1, 0.2),
            interval_range=(0.05, 0.1),
            num_stringers=4,
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
registry.register("terrain", "sirius_atec", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=SIRIUS_ATEC))