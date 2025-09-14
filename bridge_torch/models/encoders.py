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
        """Call this at the start of forward, to adapt conv1 if input channels != expected"""
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

        self.num_output_features = self.forward(x).flatten(1).shape[1] # in resnet101, it's 131072
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)