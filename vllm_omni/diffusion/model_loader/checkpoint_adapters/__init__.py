# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn

from .modelopt import (
    ModelOptFp8CheckpointAdapter,
    ModelOptMixedPrecisionCheckpointAdapter,
    ModelOptNvFp4CheckpointAdapter,
)
from .modelopt_native import ModelOptNativeFp8CheckpointAdapter


def _model_dtype(model: nn.Module) -> torch.dtype:
    param = next(model.parameters(), None)
    return param.dtype if param is not None else torch.bfloat16


def get_checkpoint_adapter(
    model: nn.Module,
    source: object,
    quant_config: object | None,
    use_safetensors: bool,
) -> (
    ModelOptFp8CheckpointAdapter
    | ModelOptNvFp4CheckpointAdapter
    | ModelOptMixedPrecisionCheckpointAdapter
    | ModelOptNativeFp8CheckpointAdapter
    | None
):
    if use_safetensors:
        # Checkpoint-driven (sidecar) detection; independent of quant_config.
        # Raises CheckpointIntegrityError on a present-but-unsupported sidecar
        # (fail fast), returns None for unquantized checkpoints.
        native_adapter = ModelOptNativeFp8CheckpointAdapter.detect(
            source, target_dtype=_model_dtype(model)
        )
        if native_adapter is not None:
            return native_adapter
    if ModelOptFp8CheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptFp8CheckpointAdapter(model, source)
    if ModelOptNvFp4CheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptNvFp4CheckpointAdapter(model, source)
    if ModelOptMixedPrecisionCheckpointAdapter.is_compatible(source, quant_config, use_safetensors):
        return ModelOptMixedPrecisionCheckpointAdapter(model, source)
    return None


__all__ = [
    "ModelOptFp8CheckpointAdapter",
    "ModelOptMixedPrecisionCheckpointAdapter",
    "ModelOptNativeFp8CheckpointAdapter",
    "ModelOptNvFp4CheckpointAdapter",
    "get_checkpoint_adapter",
]
