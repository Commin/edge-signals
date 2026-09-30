"""Switch path - DOCUMENTED STUB, NOT IMPLEMENTED.

A switch would activate another registered model version on the edge device (for example a cloud-retrained version, or a previously
registered one) without retraining. In this delivery the registry (register stage) records versions; activating one is done by pointing
`model.weights` and `model.version` in configs/inference.yaml at its registry entry. Automatic selection between versions (which version
to switch to, when, and how to validate it before activation) is not implemented.
"""


def switch(*args, **kwargs):
    raise NotImplementedError("the switch path is a documented stub: activate a registered version by editing configs/inference.yaml (see README)")
