import os
import json
from pathlib import Path

from active_adaptation.envs.mujoco import MJArticulationCfg
from active_adaptation.registry import Registry
import active_adaptation.utils.symmetry as symmetry_utils

ASSET_PATH = Path(__file__).parent / "assets"
registry = Registry.instance()

SIRIUS_WHEEL_CFG = MJArticulationCfg(
    mjcf_path=str(ASSET_PATH/ "sirius_wheel" / "sirius_wheel.xml"),
    **json.load(open(ASSET_PATH/ "sirius_wheel" / "sirius_wheel.json")),
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "LF_HAA": (-1, "RF_HAA"),
        "LH_HAA": (-1, "RH_HAA"),
        "LF_HFE": (1, "RF_HFE"),
        "LH_HFE": (1, "RH_HFE"),
        "LF_KFE": (1, "RF_KFE"),
        "LH_KFE": (1, "RH_KFE"),
        "LF_WHEEL": (1, "RF_WHEEL"),
        "LH_WHEEL": (1, "RH_WHEEL"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
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
    })
)

registry.register("asset", "sirius_wheel", SIRIUS_WHEEL_CFG)