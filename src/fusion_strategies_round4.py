"""Validation-only fusion candidates for the round-4 shoreline model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_multidate_ablation_pilot import AblationDFANet, FixedEdgeMode
from run_quick_model_pilot import DSConv, Encoder, masked_loss


class WeightedParallelEdges(FixedEdgeMode):
    """Learn one normalized weight for each fixed edge operator."""

    def __init__(self) -> None:
        super().__init__("parallel")
        self.operator_logits = nn.Parameter(torch.zeros(2))  # Prewitt, LoG

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = super().forward(x)  # Pgreen,Lgreen,Pnir,Lnir
        weights = 2.0 * torch.softmax(self.operator_logits, dim=0)
        scale = torch.stack([weights[0], weights[1], weights[0], weights[1]])
        return edge * scale[None, :, None, None]


class WeightedParallelDFANet(AblationDFANet):
    def __init__(self) -> None:
        super().__init__(prior=True, edge_mode="parallel")
        self.edges = WeightedParallelEdges()


class GatedParallelDFANet(nn.Module):
    """Gate edge features with semantic context at every encoder scale."""

    def __init__(self) -> None:
        super().__init__()
        semantic_widths = (24, 48, 96, 128)
        edge_widths = (12, 20, 32, 48)
        self.semantic = Encoder(8, semantic_widths)
        self.edges = FixedEdgeMode("parallel")
        self.edge_encoder = Encoder(4, edge_widths)
        self.gates = nn.ModuleList([
            nn.Conv2d(sw + ew, ew, 1) for sw, ew in zip(semantic_widths, edge_widths)
        ])
        self.d2 = DSConv(128 + 48 + 96 + 32, 96)
        self.d1 = DSConv(96 + 48 + 20, 64)
        self.d0 = DSConv(64 + 24 + 12, 48)
        self.head = nn.Conv2d(48, 1, 1)

    def forward(self, x: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        semantic = self.semantic(torch.cat([x, priors], dim=1))
        edge = self.edge_encoder(self.edges(x))
        gated = [
            e * torch.sigmoid(g(torch.cat([s, e], dim=1)))
            for s, e, g in zip(semantic, edge, self.gates)
        ]
        s0, s1, s2, s3 = semantic
        e0, e1, e2, e3 = gated
        z = torch.cat([s3, e3], dim=1)
        z = F.interpolate(z, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        z = self.d2(torch.cat([z, s2, e2], dim=1))
        z = F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        z = self.d1(torch.cat([z, s1, e1], dim=1))
        z = F.interpolate(z, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        z = self.d0(torch.cat([z, s0, e0], dim=1))
        return F.interpolate(self.head(z), size=x.shape[-2:], mode="bilinear", align_corners=False)


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    kernel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 4
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(image, kernel_x, padding=1)
    gy = F.conv2d(image, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-6).clamp(0, 1)


def segmentation_edge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    edge_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, segmentation and masked boundary-consistency losses."""
    segmentation = masked_loss(logits, target, mask)
    valid = (mask > 0).to(logits.dtype)
    predicted_edge = gradient_magnitude(torch.sigmoid(logits))
    target_edge = gradient_magnitude(target)
    boundary = (torch.abs(predicted_edge - target_edge) * valid).sum() / valid.sum().clamp_min(1)
    total = segmentation + float(edge_weight) * boundary
    return total, segmentation.detach(), boundary.detach()


def factories() -> dict[str, callable]:
    return {
        "parallel_concat": lambda: AblationDFANet(True, "parallel"),
        "parallel_weighted": WeightedParallelDFANet,
        "parallel_gated": GatedParallelDFANet,
    }


if __name__ == "__main__":
    for name, factory in factories().items():
        model = factory()
        x = torch.randn(2, 6, 96, 96)
        priors = torch.randn(2, 2, 96, 96)
        output = model(x, priors)
        loss, _, _ = segmentation_edge_loss(output, torch.rand_like(output).round(), torch.ones_like(output), 0.3)
        loss.backward()
        print(name, tuple(output.shape), sum(p.numel() for p in model.parameters()))
