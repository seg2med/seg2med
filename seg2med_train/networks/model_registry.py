"""Class-level registry filled by the @register_model decorator on runner classes.

Training does not look models up here: `python -m seg2med_train.train` selects
`model_name` through seg2med_train.model_selection.MODEL_RUNNERS. To add a model
from another package, use seg2med_train.model_selection.register_model (see
"Extending seg2med" in the main README). MODEL_REGISTRY stays because extension
packages read it to find the built-in runner classes.
"""
MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator
