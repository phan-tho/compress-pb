"""Model constructors used by the original experiments.

Some historical constructors depend on :mod:`timm`.  Keeping that optional
means the projector module remains usable with torchvision-only experiments.
"""

try:  # timm is needed only for the historical model zoo.
    from .resnet import resnet18k, resnet20
    from .fcnet import fc_784_10
    from .lenet import LeNet, LeNet5
    from .utils import create_model
    from .small_cnn import layer13s
    from .squeezenet import SqueezeNet
    from .surgery_efficientnet_b0 import surgery_efficientnet_b0, EfficientNet_b1
except ModuleNotFoundError as exc:
    if exc.name != "timm":
        raise

if "resnet18k" in globals():
    __all__ = [
        'create_model', 'resnet18k', 'resnet20', 'surgery_efficientnet_b0',
        'SqueezeNet', 'EfficientNet_b1', 'fc_784_10', 'LeNet', 'LeNet5',
        'layer13s',
    ]
else:
    __all__ = []
