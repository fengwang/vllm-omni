# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
from torch import nn

from .modelopt import (
    ModelOptFp8CheckpointAdapter,
    ModelOptMixedPrecisionCheckpointAdapter,
    ModelOptNvFp4CheckpointAdapter,
)
from .modelopt_native import ModelOptNativeFp8CheckpointAdapter
from .modelopt_native_fp8_w8a16 import ModelOptNativeFp8W8A16CheckpointAdapter
from .modelopt_native_nvfp4 import ModelOptNativeNvfp4CheckpointAdapter

# Opt-in flag (P6-S2): when set, the FP8-blockwise checkpoint is served
# W8A16-**resident** (MLP targets FP8-resident, per-op dequant) instead of the
# default dequant-on-load path. Necessary because FP8-dist carries no
# ``quant_recipe`` hook and the checkpoint is immutable. Unset ⇒ dequant path
# (INV-6 fallback + GATE-S2-W8A16 NO-GO escape).
FP8_W8A16_FLAG = "COSMOS3_FP8_W8A16"


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
    | ModelOptNativeFp8W8A16CheckpointAdapter
    | ModelOptNativeNvfp4CheckpointAdapter
    | None
):
    if use_safetensors:
        # Checkpoint-driven (sidecar) detection; independent of quant_config.
        # Raises CheckpointIntegrityError on a present-but-unsupported sidecar
        # (fail fast), returns None for unquantized checkpoints. NVFP4 and FP8
        # native adapters key off distinct sidecar filenames, so order is safe;
        # both must precede the generic quant_config-driven adapters below.
        #
        # P6-S2: when COSMOS3_FP8_W8A16 is set, the FP8-blockwise sidecar is served
        # W8A16-resident. This must precede the dequant FP8 adapter (both key off the
        # same root quantization_config.json); it returns None for the NVFP4 sidecar,
        # so NVFP4 is unaffected either way.
        if os.environ.get(FP8_W8A16_FLAG) == "1":
            w8a16_adapter = ModelOptNativeFp8W8A16CheckpointAdapter.detect(
                source, target_dtype=_model_dtype(model)
            )
            if w8a16_adapter is not None:
                return w8a16_adapter
        native_adapter = ModelOptNativeFp8CheckpointAdapter.detect(
            source, target_dtype=_model_dtype(model)
        )
        if native_adapter is not None:
            return native_adapter
        nvfp4_native_adapter = ModelOptNativeNvfp4CheckpointAdapter.detect(
            source, target_dtype=_model_dtype(model)
        )
        if nvfp4_native_adapter is not None:
            return nvfp4_native_adapter
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
    "ModelOptNativeFp8W8A16CheckpointAdapter",
    "ModelOptNvFp4CheckpointAdapter",
    "get_checkpoint_adapter",
]
