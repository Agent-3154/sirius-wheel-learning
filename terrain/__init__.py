import active_adaptation

if active_adaptation.get_backend() == "isaac":
    from . import isaaclab
else:
    pass