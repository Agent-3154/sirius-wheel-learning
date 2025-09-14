import active_adaptation

from . import terrain
from . import sirius_command_atec
from . import sirius_demo

if active_adaptation.get_backend() == "isaac":
    from . import sirius_assets_isaac
else:
    from . import sirius_assets_mujoco

