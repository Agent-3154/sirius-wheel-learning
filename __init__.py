import active_adaptation

from . import terrain

if active_adaptation.get_backend() == "isaac":
    from . import sirius_assets
else:
    pass