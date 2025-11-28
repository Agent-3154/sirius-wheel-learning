from pathlib import Path

from active_adaptation.registry import Registry
from active_adaptation.utils.symmetry import mirrored

from active_adaptation.assets import AssetCfg, InitialStateCfg, ActuatorCfg, ContactSensorCfg

ASSET_PATH = Path(__file__).parent / "assets"
registry = Registry.instance()


SIRIUS_WHEEL2_CFG = AssetCfg(
    usd_path=ASSET_PATH / "ly-mid-w-0915B/ly-mid-w-0915B.usd",
    mjcf_path=ASSET_PATH / "sirius_wheel_new/ly-mid-w-0915B.xml",
    self_collisions=False,
    init_state=InitialStateCfg(
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
    actuators={
        "base_legs": ActuatorCfg(
            joint_names_expr=".*",
            effort_limit={
                ".*_HAA": 50.,
                ".*_HFE": 50.,
                ".*_KFE": 100.,
                ".*_WHEEL": 80.
            },
            velocity_limit=80.,
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


SIRIUS_DIFF = AssetCfg(
    usd_path=ASSET_PATH / "ly-mid-p-0916/ly-mid-p-0916.usd",
    mjcf_path=ASSET_PATH / "ly-mid-p-0916/ly-mid-p-0916.xml",
    self_collisions=False,
    sensors_isaaclab=[
        ContactSensorCfg(
            name="contact_forces",
            primary=".*",
            secondary=[],
            history_length=3,
            track_air_time=True
        ),
    ],
    sensors_mjlab=[
        ContactSensorCfg(
            name="contact_forces",
            primary=".*_FOOT",
            secondary={"mode": "body", "pattern": "terrain"},
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True
        )
    ],
    init_state=InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            "L[F,H]_HAA": 0.1,
            "R[F,H]_HAA": -0.1,
            "[L,R]F_HFE":  0.5,
            "[L,R]H_HFE": -0.5,
            "[L,R]F_KNEE": -1.2,
            "[L,R]H_KNEE":  1.2,
        },
        joint_vel={".*": 0.}
    ),
    actuators={
        "(HAA|HFE)": ActuatorCfg(
            joint_names_expr=".*_(HAA|HFE)",
            effort_limit=80.,
            velocity_limit=40.,
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
            friction=0.01
        ),
        "(KNEE)": ActuatorCfg(
            joint_names_expr=".*_(KNEE)",
            effort_limit=80.,
            velocity_limit=20.,
            stiffness=50.0,
            damping=2.0,
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
    joint_names_isaac=[
        "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
        "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
        "LF_KNEE", "LH_KNEE", "RF_KNEE", "RH_KNEE"
    ],
    joint_names_mjlab=[
        "LF_HAA", "LF_HFE", "LF_KNEE",
        "RF_HAA", "RF_HFE", "RF_KNEE",
        "LH_HAA", "LH_HFE", "LH_KNEE",
        "RH_HAA", "RH_HFE", "RH_KNEE"
    ],
    body_names_isaac=[
        "trunk",
        "LF_hip", "LH_hip", "RF_hip", "RH_hip",
        "LF_thigh", "LH_thigh", "RF_thigh", "RH_thigh",
        "LF_calf", "LH_calf", "RF_calf", "RH_calf",
        "LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"
    ]
)


registry.register("asset", "sirius_wheel2", SIRIUS_WHEEL2_CFG)
registry.register("asset", "sirius_diff", SIRIUS_DIFF)
