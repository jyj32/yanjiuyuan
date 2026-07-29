#!/usr/bin/env python3
"""Dual-YOLO-backbone RGB-D candidate ranking network."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from ultralytics import YOLO


class ConvBNAct(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, kernel: int, stride: int = 1
    ):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FuseBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.input_projection = ConvBNAct(input_channels, output_channels, 1)
        self.residual = nn.Sequential(
            ConvBNAct(output_channels, output_channels, 3),
            ConvBNAct(output_channels, output_channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        return projected + self.residual(projected)


class YoloFeatureBackbone(nn.Module):
    """Official YOLO11s layers 0..10, returning P3/P4/P5."""

    output_indices = (4, 6, 10)

    def __init__(self, layers: nn.ModuleList, input_channels: int):
        super().__init__()
        self.layers = layers
        for index, layer in enumerate(self.layers):
            if getattr(layer, "f", -1) != -1:
                raise ValueError(
                    f"YOLO backbone layer {index} unexpectedly reads layer {layer.f}"
                )
        if input_channels != 3:
            first = self.layers[0]
            old = first.conv
            replacement = nn.Conv2d(
                input_channels,
                old.out_channels,
                old.kernel_size,
                old.stride,
                old.padding,
                old.dilation,
                old.groups,
                bias=False,
                padding_mode=old.padding_mode,
            )
            with torch.no_grad():
                mean_weight = old.weight.mean(dim=1, keepdim=True)
                replacement.weight.zero_()
                replacement.weight[:, :1].copy_(mean_weight)
                # Channel 1 is the validity mask; start neutral and learn it.
            first.conv = replacement

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index in self.output_indices:
                outputs.append(x)
        return tuple(outputs)


class RGBDFusionNeck(nn.Module):
    def __init__(self, channels: tuple[int, int, int]):
        super().__init__()
        c3, c4, c5 = channels
        self.scale3 = FuseBlock(c3 * 2, c3)
        self.scale4 = FuseBlock(c4 * 2, c4)
        self.scale5 = FuseBlock(c5 * 2, c5)
        self.top4 = FuseBlock(c5 + c4, c4)
        self.top3 = FuseBlock(c4 + c3, c3)
        self.down3 = ConvBNAct(c3, c4, 3, 2)
        self.bottom4 = FuseBlock(c4 + c4, c4)
        self.down4 = ConvBNAct(c4, c5, 3, 2)
        self.bottom5 = FuseBlock(c5 + c5, c5)

    def forward(
        self,
        rgb_features: tuple[torch.Tensor, ...],
        depth_features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        p3 = self.scale3(torch.cat((rgb_features[0], depth_features[0]), dim=1))
        p4 = self.scale4(torch.cat((rgb_features[1], depth_features[1]), dim=1))
        p5 = self.scale5(torch.cat((rgb_features[2], depth_features[2]), dim=1))
        td4 = self.top4(
            torch.cat(
                (F.interpolate(p5, size=p4.shape[-2:], mode="nearest"), p4), dim=1
            )
        )
        td3 = self.top3(
            torch.cat(
                (F.interpolate(td4, size=p3.shape[-2:], mode="nearest"), p3), dim=1
            )
        )
        out4 = self.bottom4(torch.cat((self.down3(td3), td4), dim=1))
        out5 = self.bottom5(torch.cat((self.down4(out4), p5), dim=1))
        return td3, out4, out5


def sample_rotated_quads(
    feature: torch.Tensor,
    quads: torch.Tensor,
    candidate_batch_idx: torch.Tensor,
    image_size: int,
    output_size: int = 7,
) -> torch.Tensor:
    candidates = feature.index_select(0, candidate_batch_idx)
    dtype = quads.dtype
    device = quads.device
    u = torch.linspace(0.0, 1.0, output_size, device=device, dtype=dtype)
    v = torch.linspace(0.0, 1.0, output_size, device=device, dtype=dtype)
    u = u.view(1, 1, output_size, 1)
    v = v.view(1, output_size, 1, 1)
    p0 = quads[:, 0].view(-1, 1, 1, 2)
    p1 = quads[:, 1].view(-1, 1, 1, 2)
    p2 = quads[:, 2].view(-1, 1, 1, 2)
    p3 = quads[:, 3].view(-1, 1, 1, 2)
    grid_px = (
        (1 - u) * (1 - v) * p0
        + u * (1 - v) * p1
        + u * v * p2
        + (1 - u) * v * p3
    )
    grid = grid_px.clone()
    grid[..., 0] = grid[..., 0] * (2.0 / max(image_size - 1, 1)) - 1.0
    grid[..., 1] = grid[..., 1] * (2.0 / max(image_size - 1, 1)) - 1.0
    return F.grid_sample(
        candidates,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class PriorityNetwork(nn.Module):
    def __init__(self, pretrained_weights: str, image_size: int):
        super().__init__()
        base_model = YOLO(pretrained_weights).model
        backbone_layers = nn.ModuleList(list(base_model.model[:11]))
        self.rgb_backbone = YoloFeatureBackbone(deepcopy(backbone_layers), 3)
        self.depth_backbone = YoloFeatureBackbone(deepcopy(backbone_layers), 2)
        self.image_size = int(image_size)

        rgb_training = self.rgb_backbone.training
        self.rgb_backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            dummy_features = self.rgb_backbone(dummy)
        self.rgb_backbone.train(rgb_training)
        channels = tuple(int(feature.shape[1]) for feature in dummy_features)

        self.neck = RGBDFusionNeck(channels)
        self.roi_projections = nn.ModuleList(
            [ConvBNAct(channel, 128, 1) for channel in channels]
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(18, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(384 + 128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.priority_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.top1_projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def set_rgb_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.rgb_backbone.parameters():
            parameter.requires_grad = trainable

    def candidate_embeddings(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        quads: torch.Tensor,
        geometry: torch.Tensor,
        candidate_batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        fused = self.neck(rgb_features, depth_features)
        pooled = []
        for feature, projection in zip(fused, self.roi_projections):
            roi = sample_rotated_quads(
                projection(feature),
                quads,
                candidate_batch_idx,
                self.image_size,
            )
            pooled.append(roi.mean(dim=(-1, -2)))
        visual = torch.cat(pooled, dim=1)
        encoded_geometry = self.geometry_encoder(geometry)
        return self.candidate_projection(
            torch.cat((visual, encoded_geometry), dim=1)
        )

    def scene_scores(
        self, embeddings: torch.Tensor, candidate_batch_idx: torch.Tensor
    ) -> torch.Tensor:
        scene_count = int(candidate_batch_idx.max().item()) + 1
        context = embeddings.new_zeros((scene_count, embeddings.shape[1]))
        context.index_add_(0, candidate_batch_idx, embeddings)
        counts = torch.bincount(
            candidate_batch_idx, minlength=scene_count
        ).to(embeddings.dtype)
        context = context / counts.clamp_min(1).unsqueeze(1)
        candidate_context = context.index_select(0, candidate_batch_idx)
        head_input = torch.cat(
            (embeddings, candidate_context, embeddings - candidate_context), dim=1
        )
        return self.priority_head(head_input).squeeze(1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings = self.candidate_embeddings(
            batch["rgb"],
            batch["depth"],
            batch["quads"],
            batch["geometry"],
            batch["candidate_batch_idx"],
        )
        return {
            "embeddings": embeddings,
            "scores": self.scene_scores(
                embeddings, batch["candidate_batch_idx"]
            ),
            "projected": self.top1_projector(embeddings),
        }
