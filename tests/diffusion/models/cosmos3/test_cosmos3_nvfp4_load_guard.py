# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: the pipeline fp8 bypass guard must admit NVFP4 W4A16 scale params.

`Cosmos3OmniDiffusersPipeline.load_weights` runs `assert_not_fp8` on every
streamed tensor as a last-line guard against an FP8 checkpoint that bypassed its
dequant adapter. The NVFP4 W4A16 (nvfp4_blockwise) path is different: it keeps
target weights FP4 and loads the fp8 block-scale (`.weight_scale`) + fp32 global
scale (`.weight_scale_2`) straight into the ModelOpt linear method. Those scale
params MUST pass the guard, while an un-adapted fp8 `.weight` MUST still be
rejected. (Caught by the S4 sharded review: before the fix the guard aborted the
NVFP4 load on the first fp8 block scale.)
"""

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _StubTransformer(nn.Module):
    sound_gen = False
    action_gen = False

    def post_load_weights(self) -> None:  # called by load_weights
        pass


def _make_pipeline():
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    pipe = object.__new__(Cosmos3OmniDiffusersPipeline)
    nn.Module.__init__(pipe)
    pipe.transformer = _StubTransformer()
    return pipe


def _integrity_error():
    from vllm_omni.diffusion.model_loader.checkpoint_adapters.modelopt_native import (
        CheckpointIntegrityError,
    )

    return CheckpointIntegrityError


def test_load_weights_admits_nvfp4_fp8_block_scale():
    """An fp8 `.weight_scale` streamed to load_weights must NOT trip the guard."""
    pipe = _make_pipeline()
    # Not a param of the stub, so it is filtered downstream; the point is the
    # guard at the top of _remapped_weights must let this fp8 scale through.
    stream = [
        ("transformer.layers.0.mlp.gate_proj.weight_scale", torch.zeros(4, 1, dtype=torch.float8_e4m3fn)),
        ("transformer.layers.0.mlp.gate_proj.weight_scale_2", torch.zeros(1, dtype=torch.float32)),
        ("transformer.layers.0.mlp.gate_proj.weight", torch.zeros(4, 2, dtype=torch.uint8)),  # packed, uint8
    ]
    # Must not raise CheckpointIntegrityError (would have, before the fix).
    pipe.load_weights(iter(stream))


def test_load_weights_still_rejects_unadapted_fp8_weight():
    """An fp8 `.weight` (not a scale) still means a bypassed adapter -> abort."""
    pipe = _make_pipeline()
    stream = [
        ("transformer.layers.0.mlp.gate_proj.weight", torch.zeros(4, 4, dtype=torch.float8_e4m3fn)),
    ]
    with pytest.raises(_integrity_error()):
        pipe.load_weights(iter(stream))


class _StubTransformerFp8Dest(nn.Module):
    """Stub whose state_dict exposes an fp8-resident destination param, mimicking a W8A16
    MLP target after create_weights. ``proj_in`` has an identity remap, so streaming a
    matching fp8 weight exercises the guard's fp8->fp8 exemption through load_weights."""

    sound_gen = False
    action_gen = False

    def __init__(self) -> None:
        super().__init__()
        proj = nn.Module()
        proj.register_buffer("weight", torch.zeros(4, 4, dtype=torch.float8_e4m3fn))
        self.proj_in = proj

    def post_load_weights(self) -> None:
        pass


def test_load_weights_admits_fp8_weight_when_dest_is_fp8_resident():
    """P6-S3 (task 3.3, S2 eval-seed gap): the fp8->fp8 guard exemption, through the real
    load_weights wiring. An incoming fp8 ``.weight`` whose destination param is itself
    fp8-resident (the W8A16-resident MLP targets) must be ADMITTED, not aborted — the
    complement of the reject test above (bf16/absent dest). A regression that dropped the
    ``_dest_param_is_fp8`` call would spuriously abort every W8A16 load; this pins it."""
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    pipe = object.__new__(Cosmos3OmniDiffusersPipeline)
    nn.Module.__init__(pipe)
    pipe.transformer = _StubTransformerFp8Dest()
    # dest transformer.proj_in.weight is fp8-resident -> the guard must NOT abort.
    stream = [("transformer.proj_in.weight", torch.zeros(4, 4, dtype=torch.float8_e4m3fn))]
    pipe.load_weights(iter(stream))  # must not raise CheckpointIntegrityError
