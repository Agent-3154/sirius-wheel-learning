import active_adaptation

if active_adaptation.get_backend() == "isaac":
    from . import sirius_terrain_isaac
else:
    from . import sirius_terrain_mujoco