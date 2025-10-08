import os
import isaaclab.sim as sim_utils
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from active_adaptation.registry import Registry
from active_adaptation.assets.base import ArticulationCfg
from active_adaptation.utils.symmetry import mirrored

ASSET_PATH = Path(__file__).parent / "assets"
registry = Registry.instance()

SIRIUS_WHEEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/sirius_wheel_sphere.usd",
        # usd_path=f"{ASSET_PATH}/sirius_wheel/sirius_wheel_mesh.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*_HAA": 0.,
            "[L,R]F_HFE":  0.4,
            "[L,R]H_HFE": -0.4,
            "[L,R]F_KFE": -1.2,
            "[L,R]H_KFE":  1.2,
            ".*_WHEEL": 0.
        },
        joint_vel={".*": 0.}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                ".*_HAA": 50.,
                ".*_HFE": 50.,
                ".*_KFE": 80.,
                ".*_WHEEL": 80.
            },
            velocity_limit_sim=80.,
            stiffness={
                ".*_HAA": 40.,
                ".*_HFE": 40.,
                ".*_KFE": 40.,
                ".*_WHEEL": 0.
            },
            damping={
                ".*_HAA": 1.,
                ".*_HFE": 1.,
                ".*_KFE": 1.,
                ".*_WHEEL": 10.
            },
            armature=0.01,
            friction=0.01
        )
    },
    joint_symmetry_mapping=mirrored({
        "LF_HAA": (-1, "RF_HAA"),
        "LH_HAA": (-1, "RH_HAA"),
        "LF_HFE": (1, "RF_HFE"),
        "LH_HFE": (1, "RH_HFE"),
        "LF_KFE": (1, "RF_KFE"),
        "LH_KFE": (1, "RH_KFE"),
        "LF_WHEEL": (1, "RF_WHEEL"),
        "LH_WHEEL": (1, "RH_WHEEL"),
    }),
    spatial_symmetry_mapping=mirrored({
        "trunk": "trunk",
        "front": "front",
        "back": "back",
        "LF_hip": "RF_hip",
        "LH_hip": "RH_hip",
        "LF_calf": "RF_calf",
        "LH_calf": "RH_calf",
        "LF_thigh": "RF_thigh",
        "LH_thigh": "RH_thigh",
        "LF_FOOT": "RF_FOOT",
        "LH_FOOT": "RH_FOOT",
    }),
)


SIRIUS_WHEEL2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/ly-mid-w-0915B/ly-mid-w-0915B.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            "L[F,H]_HAA": 0.1,
            "R[F,H]_HAA": -0.1,
            "[L,R]F_HFE":  0.95,
            "[L,R]H_HFE": -0.95,
            "[L,R]F_KFE": -1.6,
            "[L,R]H_KFE":  1.6,
            ".*_WHEEL": 0.
        },
        joint_vel={".*": 0.}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                ".*_HAA": 50.,
                ".*_HFE": 50.,
                ".*_KFE": 100.,
                ".*_WHEEL": 80.
            },
            velocity_limit_sim=80.,
            stiffness={
                ".*_HAA": 50.,
                ".*_HFE": 50.,
                ".*_KFE": 50.,
                ".*_WHEEL": 0.
            },
            damping={
                ".*_HAA": 2.,
                ".*_HFE": 2.,
                ".*_KFE": 2.,
                ".*_WHEEL": 10.
            },
            armature=0.01,
            friction=0.01
        )
    },
    joint_symmetry_mapping=mirrored({
        "LF_HAA": (-1, "RF_HAA"),
        "LH_HAA": (-1, "RH_HAA"),
        "LF_HFE": (1, "RF_HFE"),
        "LH_HFE": (1, "RH_HFE"),
        "LF_KFE": (1, "RF_KFE"),
        "LH_KFE": (1, "RH_KFE"),
        "LF_WHEEL": (1, "RF_WHEEL"),
        "LH_WHEEL": (1, "RH_WHEEL"),
    }),
    spatial_symmetry_mapping=mirrored({
        "base": "base",
        "LF_hip": "RF_hip",
        "LH_hip": "RH_hip",
        "LF_calf": "RF_calf",
        "LH_calf": "RH_calf",
        "LF_thigh": "RF_thigh",
        "LH_thigh": "RH_thigh",
        "LF_FOOT": "RF_FOOT",
        "LH_FOOT": "RH_FOOT",
    }),
)


SIRIUS_DIFF = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/ly-mid-p-0916/ly-mid-p-0916.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*_HAA": 0.,
            "[L,R]F_HFE":  0.4,
            "[L,R]H_HFE": -0.4,
            "[L,R]F_KNEE": -1.2,
            "[L,R]H_KNEE":  1.2,
        },
        joint_vel={".*": 0.}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={".*": 50.},
            velocity_limit_sim=80.,
            stiffness={
                ".*_HAA": 40.,
                ".*_HFE": 40.,
                ".*_KNEE": 40.,
            },
            damping={
                ".*_HAA": 2.,
                ".*_HFE": 2.,
                ".*_KNEE": 2.,
            },
            armature=0.01,
            friction=0.01
        )
    },
    joint_symmetry_mapping=mirrored({
        "LF_HAA": (-1, "RF_HAA"),
        "LH_HAA": (-1, "RH_HAA"),
        "LF_HFE": (1, "RF_HFE"),
        "LH_HFE": (1, "RH_HFE"),
        "LF_KNEE": (1, "RF_KNEE"),
        "LH_KNEE": (1, "RH_KNEE")
    }),
    spatial_symmetry_mapping=mirrored({
        "trunk": "trunk",
        "LF_hip": "RF_hip",
        "LH_hip": "RH_hip",
        "LF_calf": "RF_calf",
        "LH_calf": "RH_calf",
        "LF_thigh": "RF_thigh",
        "LH_thigh": "RH_thigh",
        "LF_FOOT": "RF_FOOT",
        "LH_FOOT": "RH_FOOT",
    }),
)


SIRIUS_DIFF_OLD = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/sirius_diff_new/sirius_diff_new.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*_HAA": 0.,
            "[L,R]F_HFE":  0.4,
            "[L,R]H_HFE": -0.4,
            "[L,R]F_KFE": -1.2,
            "[L,R]H_KFE":  1.2,
        },
        joint_vel={".*": 0.}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={".*": 50.},
            velocity_limit_sim=80.,
            stiffness={
                ".*_HAA": 40.,
                ".*_HFE": 40.,
                ".*_KFE": 40.,
            },
            damping={
                ".*_HAA": 2.,
                ".*_HFE": 2.,
                ".*_KFE": 2.,
            },
            armature=0.01,
            friction=0.01
        )
    },
    joint_symmetry_mapping=mirrored({
        "LF_HAA": (-1, "RF_HAA"),
        "LH_HAA": (-1, "RH_HAA"),
        "LF_HFE": (1, "RF_HFE"),
        "LH_HFE": (1, "RH_HFE"),
        "LF_KFE": (1, "RF_KFE"),
        "LH_KFE": (1, "RH_KFE"),
    }),
    spatial_symmetry_mapping=mirrored({
        "trunk": "trunk",
        "LF_hip": "RF_hip",
        "LH_hip": "RH_hip",
        "LF_calf": "RF_calf",
        "LH_calf": "RH_calf",
        "LF_thigh": "RF_thigh",
        "LH_thigh": "RH_thigh",
        "LF_FOOT": "RF_FOOT",
        "LH_FOOT": "RH_FOOT",
    }),
)
    
registry.register("asset", "sirius_wheel", SIRIUS_WHEEL_CFG)
registry.register("asset", "sirius_wheel2", SIRIUS_WHEEL2_CFG)
registry.register("asset", "sirius_diff", SIRIUS_DIFF)
registry.register("asset", "sirius_diff_old", SIRIUS_DIFF_OLD)