import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetV1Bridge(nn.Module):
    def __init__(
        self,
        arch: str = "resnet34",
    ):
        super().__init__()

        base = {
            "resnet18": tvm.resnet18(weights=None),
            "resnet34": tvm.resnet34(weights=None),
            "resnet50": tvm.resnet50(weights=None),
            "resnet101": tvm.resnet101(weights=None),
            "resnet152": tvm.resnet152(weights=None),
        }[arch]
        # keep stem and layers; drop avgpool/fc
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )

        self._in_ch = None
        self.num_output_features = None
        
    def adapt_to_input_channels(self, x: torch.Tensor):
        """Adapt conv1 if input channels do not match the incoming tensor.

        This modifies the first convolution to accept arbitrary input channels
        by replicating the average across original RGB weights.
        """
        in_ch = int(x.shape[1])
        conv1: nn.Conv2d = self.stem[0]

        if isinstance(conv1, nn.Conv2d) and conv1.in_channels != in_ch:
            new = nn.Conv2d(
                in_ch,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )
            with torch.no_grad():
                w = conv1.weight
                mean_w = w.mean(dim=1, keepdim=True)  # average across channels
                new_w = mean_w.repeat(1, in_ch, 1, 1)
                new.weight.copy_(new_w)
            self.stem[0] = new.to(conv1.weight.device)
            self._in_ch = in_ch
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input channel count is compatible with stem[0]
        self.adapt_to_input_channels(x)
        y = self.stem(x)
        # Lazily set feature dimension for downstream MLP construction
        # Flatten over all non-batch dimensions
        try:
            self._feat_dim = int(y.flatten(1).shape[1])
        except Exception:
            # Best-effort; leave unset if shape probing fails
            pass
        return y


def build_encoder(name: str, **kwargs) -> nn.Module:
    # TODO: clean this later, just for compatibility with eval.py
    """Factory to build visual encoders.

    Supported names (case-insensitive):
    - "resnetv1-18-bridge", "resnetv1-34-bridge", "resnetv1-50-bridge",
      "resnetv1-101-bridge", "resnetv1-152-bridge"
    - "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"

    Extra kwargs are accepted for forward-compatibility and ignored here.
    """
    n = (name or "").lower()
    arch = kwargs.get("arch")
    if arch is None:
        if "resnetv1-18" in n or n == "resnet18":
            arch = "resnet18"
        elif "resnetv1-34" in n or n == "resnet34" or n == "resnetv1-bridge":
            arch = "resnet34"
        elif "resnetv1-50" in n or n == "resnet50":
            arch = "resnet50"
        elif "resnetv1-101" in n or n == "resnet101":
            arch = "resnet101"
        elif "resnetv1-152" in n or n == "resnet152":
            arch = "resnet152"
        else:
            arch = "resnet34"  # sensible default
    return ResNetV1Bridge(arch=arch)