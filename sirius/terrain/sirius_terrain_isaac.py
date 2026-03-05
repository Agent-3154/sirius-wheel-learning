import active_adaptation
import trimesh
from active_adaptation.registry import Registry

from isaaclab.terrains import (
    TerrainImporterCfg,
    SubTerrainBaseCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
    HfRandomUniformTerrainCfg,
    MeshPitTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshGapTerrainCfg,
    HfSteppingStonesTerrainCfg,
)
from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.mesh_terrains import flat_terrain
import isaaclab.sim as sim_utils
import numpy as np
import random
import copy

from dataclasses import MISSING
from .functional import platform_with_slope, pallet_with_platform, platform_with_stairs

registry = Registry.instance()


@configclass
class PlatformWithSlopeCfg(SubTerrainBaseCfg):
    function = lambda difficulty, cfg: platform_with_slope(
        cfg.size,
        np.random.uniform(cfg.height_range[0], cfg.height_range[1])
    )
    height_range: tuple[float, float] = (0.1, 0.2)


@configclass
class PlatformWithStairsCfg(SubTerrainBaseCfg):
    function = lambda difficulty, cfg: platform_with_stairs(
        cfg.size,
        num_steps=random.randint(1, 4),
        step_width=random.uniform(*cfg.step_width_range),
        step_height=random.uniform(*cfg.step_height_range)
    )
    step_width_range: tuple[float, float] = (0.20, 0.30)
    step_height_range: tuple[float, float] = (0.05, 0.15)


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


def ramp_terrain(difficulty: float, cfg: "RampTerrainCfg"):
    height = cfg.height_range[0] + (cfg.height_range[1] - cfg.height_range[0]) * difficulty
    
    mesh = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], height))

    up = np.random.rand() > 0.5
    if up:
        # remove the bottom face
        bottom_faces = np.where(mesh.triangles_center[:, 2] <= -height / 2)
        mesh.faces = np.delete(mesh.faces, bottom_faces, axis=0)
        # bevel top edges
        top_vertices = mesh.vertices[mesh.vertices[:, 2] >= height / 2]
        top_vertices[:, 0] *= 0.5
        mesh.vertices[mesh.vertices[:, 2] >= height / 2] = top_vertices
        mesh.vertices[:, 2] += height / 2
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, height])
    else:
        # remove the top face
        top_faces = np.where(mesh.triangles_center[:, 2] >= height / 2)
        mesh.faces = np.delete(mesh.faces, top_faces, axis=0)
        # bevel bottom edges
        bottom_vertices = mesh.vertices[mesh.vertices[:, 2] <= -height / 2]
        bottom_vertices[:, 0] *= 0.5
        mesh.vertices[mesh.vertices[:, 2] <= -height / 2] = bottom_vertices
        mesh.vertices[:, 2] -= height / 2
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, -height])
        # flip the normals for correct collision
        mesh.faces = np.fliplr(mesh.faces)
    # center the mesh
    mesh.vertices += np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    return [mesh], origin


@configclass
class RampTerrainCfg(SubTerrainBaseCfg):
    function = ramp_terrain
    height_range: tuple[float, float] = MISSING


def room_terrain(difficulty: float, cfg: "RoomTerrainCfg"):
    mesh_list, origin = flat_terrain(difficulty, cfg)
    wall_0 = trimesh.creation.box(extents=(4.0, 0.4, 1.0))
    wall_0.apply_translation(np.array([2.0, 0.2, 0.5]))
   
    wall_1 = wall_0.copy()
    wall_1.apply_transform(
        trimesh.transformations.transform_around(
            matrix=trimesh.transformations.rotation_matrix(
                angle=np.pi / 2,
                direction=np.array([0.0, 0.0, 1.0]),
            ),
            point=np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0]),
        )
    )
    mesh_list.append(wall_0)
    mesh_list.append(wall_1)
    return mesh_list, origin


@configclass
class RoomTerrainCfg(SubTerrainBaseCfg):
    function = room_terrain



def double_prism(difficulty: float, cfg):
    height_scale = cfg.height_range[0] + (cfg.height_range[1] - cfg.height_range[0]) * difficulty
    sink = np.random.uniform(cfg.sink_range[0], cfg.sink_range[1])
    sink = np.clip(sink, 0.0, height_scale)

    meshes = []
    prism_0 = trimesh.creation.extrude_triangulation(
        vertices=np.array([[0., 0.], [0., 1.], [2., 0.]]),
        faces=np.array([[0, 1, 2]]),
        height=1,
    )
    # prism_0.apply_translation(np.array([-1., 0., 0.]))
    prism_0.apply_transform(
        trimesh.transformations.rotation_matrix(
            angle=np.pi / 2,
            direction=np.array([1, 0, 0])
        )
    )
    prism_0.apply_translation([-0.5, -0.5, 0.])
    meshes.append(prism_0)

    prism_1 = trimesh.creation.extrude_triangulation(
        vertices=np.array([[0., 0.], [0., -1.], [-2., 0.]]),
        faces=np.array([[0, 1, 2]]),
        height=1
    )
    # prism_1.apply_translation(np.array([1., 0., 0.]))
    prism_1.apply_transform(
        trimesh.transformations.rotation_matrix(
            angle=-np.pi / 2,
            direction=np.array([1, 0, 0])
        )
    )
    prism_1.apply_translation([0.5, 0.5, 0.])
    meshes.append(prism_1)

    box_0 = trimesh.creation.box(
        extents=(3.0, 1.0, 1.0),
        transform=trimesh.transformations.translation_matrix([0.0, 0.0, 0.5]),
    )
    meshes.append(box_0)
    box_1 = trimesh.creation.box(
        extents=(1.0, 1.0, 1.0),
        transform=trimesh.transformations.translation_matrix([1.0, 1.0, 0.5]),
    )
    meshes.append(box_1)
    box_2 = trimesh.creation.box(
        extents=(1.0, 1.0, 1.0),
        transform=trimesh.transformations.translation_matrix([-1.0, -1.0, 0.5]),
    )
    meshes.append(box_2)

    # Combine meshes
    mesh: trimesh.Trimesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices()
    mesh.apply_translation([1.5, 1.5, -sink])
    mesh.apply_scale(np.array([cfg.size[0] / 3, cfg.size[1] / 3, height_scale]))
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, height_scale - sink])
    return [mesh], origin


@configclass
class DoublePrismCfg(SubTerrainBaseCfg):
    function = double_prism
    height_range = (1.0, 2.0)
    sink_range = (0.1, 0.2)


SIRIUS_DEMO = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 10.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "platform_with_slope": PlatformWithSlopeCfg(
        #     proportion=0.5,
        #     height_range=(0.1, 0.3),
        # ),
        # "flat": MeshPlaneTerrainCfg(
        #     proportion=0.5,
        # ),
        "platform_with_stairs": PlatformWithStairsCfg(
            proportion=0.5,
            step_width_range=(0.20, 0.30),
            step_height_range=(0.04, 0.10),
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
            interval_range=(0.10, 0.20),
            num_stringers=4,
        ),
        "gap": MeshGapTerrainCfg(
            proportion=0.5,
            gap_width_range=(0.1, 0.40),
            platform_width=4.0,
        ),
        "pit": MeshPitTerrainCfg(
            proportion=0.5,
            pit_depth_range=(0.05, 0.20),
            platform_width=4.0,
        ),
        # "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.5,
        #     step_height_range=(0.05, 0.15),
        #     step_width=0.35,
        #     platform_width=2.0,
        #     border_width=1.0,
        # ),
    },
)


SIRIUS_STEPPING_STONE = TerrainGeneratorCfg(
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
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=0.5,
            stone_distance_range=(0.05, 0.1),
            stone_width_range=(0.25, 0.5),
            stone_height_max=0.4,
            platform_width=2.0,
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
registry.register("terrain", "sirius_stepping_stone", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=SIRIUS_STEPPING_STONE))

try:
    from atec_rl_lab.tasks.task_d import TASK_D_TERRAIN_CFG
    from atec_rl_lab.tasks.task_d.terrain import PlatformTerrainCfg, PitAndPlatformTerrainCfg
    TASK_D_TERRAIN_CFG.terrain_generator.num_rows = 10
    TASK_D_TERRAIN_CFG.terrain_generator.num_cols = 20
    TASK_D_TERRAIN_CFG.terrain_generator.border_width = 50.0
    TASK_D_TERRAIN_CFG.max_init_terrain_level = None

    TASK_D_TERRAIN_CFG_1 = copy.deepcopy(TASK_D_TERRAIN_CFG)
    TASK_D_TERRAIN_CFG_2 = copy.deepcopy(TASK_D_TERRAIN_CFG)
    TASK_D_TERRAIN_CFG_1.terrain_generator.sub_terrains = {
        "flat": MeshPlaneTerrainCfg(proportion=0.1),
        "platform_0": PlatformTerrainCfg(proportion=0.4, platform_height_range=(0.1, 0.6)),
        "pit": MeshPitTerrainCfg(proportion=0.2, pit_depth_range=(0.05, 0.20), platform_width=4.0),
        "platform_1": PlatformTerrainCfg(proportion=0.3, platform_height_range=(0.1, 0.6)),
    }
    TASK_D_TERRAIN_CFG_2.terrain_generator.sub_terrains = {
        # "platform": PlatformTerrainCfg(proportion=0.3, platform_height_range=(0.1, 0.4)),
        "pit_and_platform": PitAndPlatformTerrainCfg(proportion=0.7, border_width=0.5),
    }
    TASK_D_TERRAIN_CFG_1.terrain_generator.curriculum = True
    TASK_D_TERRAIN_CFG_2.terrain_generator.curriculum = True
    registry.register("terrain", "atec_task_d_1", TASK_D_TERRAIN_CFG_1)
    registry.register("terrain", "atec_task_d_2", TASK_D_TERRAIN_CFG_2)
except ImportError:
    pass