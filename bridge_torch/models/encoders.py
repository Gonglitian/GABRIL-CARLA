import torch
import torch.nn as nn
import torchvision.models as tvm


class AddSpatialCoords(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (N, C, H, W)
        n, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(n, -1, -1, -1)
        return torch.cat([x, coords], dim=1)


class ResNetV1Bridge(nn.Module):
    def __init__(
        self,
        arch: str = "resnet34",
        pooling_method: str = "avg",  # "avg" | "none"
        add_spatial_coordinates: bool = True,
        out_dim: int | None = None,
    ):
        super().__init__()
        assert arch in {"resnet18", "resnet34", "resnet50"}
        self.pooling_method = pooling_method
        self.add_spatial = add_spatial_coordinates

        base = {
            "resnet18": tvm.resnet18(weights=None),
            "resnet34": tvm.resnet34(weights=None),
            "resnet50": tvm.resnet50(weights=None),
        }[arch]

        # keep stem and layers; drop avgpool/fc
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.add_coords = AddSpatialCoords() if self.add_spatial else nn.Identity()
        self.pool = (
            nn.AdaptiveAvgPool2d((1, 1)) if pooling_method == "avg" else nn.Identity()
        )

        # compute feature dim lazily at first forward pass
        self._feat_dim = None
        self._in_ch = None  # track input channels used to adapt conv1 dynamically

        self.proj = None
        self.out_dim = out_dim

    def _adapt_input_conv(self, in_ch: int):
        # Replace first conv to accept in_ch while preserving out_channels, kernel, stride, padding
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
                # average across input channel then repeat to match in_ch
                mean_w = w.mean(dim=1, keepdim=True)
                new_w = mean_w.repeat(1, in_ch, 1, 1)
                new.weight.copy_(new_w)
            self.stem[0] = new.to(conv1.weight.device)
        self._in_ch = in_ch

    def _infer_feat_dim(self, x: torch.Tensor):
        with torch.no_grad():
            # adapt conv1 if channel count differs
            self._adapt_input_conv(int(x.shape[1]))
            y = self.stem(x)
            y = self.add_coords(y)
            y = self.pool(y)
            if self.pooling_method == "avg":
                y = torch.flatten(y, 1)
            else:
                y = torch.flatten(y, 1)
            self._feat_dim = y.shape[1]
            if self.out_dim is not None and self.proj is None:
                self.proj = nn.Linear(self._feat_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._feat_dim is None:
            self._infer_feat_dim(x)
        # ensure input conv matches the current input channels
        if self._in_ch is None or self._in_ch != int(x.shape[1]):
            self._adapt_input_conv(int(x.shape[1]))
        y = self.stem(x)
        y = self.add_coords(y)
        y = self.pool(y)
        y = torch.flatten(y, 1)
        if self.proj is not None:
            y = self.proj(y)
        return y


def build_encoder(name: str, **kwargs) -> nn.Module:
    """Factory for encoders used in the torch port.

    Supported names:
      - "resnetv1-bridge" with kwarg "arch" in {"resnet18", "resnet34", "resnet50"}
      - "resnetv1-18-bridge" (alias for arch="resnet18")
      - "resnetv1-34-bridge" (alias for arch="resnet34")
      - "resnetv1-50-bridge" (alias for arch="resnet50")
    """

    # Drop kwargs not supported in torch implementation but may appear from JAX configs
    kw = dict(kwargs)
    kw.pop("act", None)

    if name == "resnetv1-bridge":
        arch = kw.pop("arch", "resnet34")
        return ResNetV1Bridge(arch=arch, **kw)

    if name in {"resnetv1-18-bridge", "resnetv1-34-bridge", "resnetv1-50-bridge"}:
        arch = {
            "resnetv1-18-bridge": "resnet18",
            "resnetv1-34-bridge": "resnet34",
            "resnetv1-50-bridge": "resnet50",
        }[name]
        kw.pop("arch", None)
        return ResNetV1Bridge(arch=arch, **kw)

    # placeholder for film variant (not implemented)
    if name.endswith("-film"):
        raise NotImplementedError("FILM variant not implemented in torch port yet")

    raise ValueError(f"Unknown encoder name: {name}")
