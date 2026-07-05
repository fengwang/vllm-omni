# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A16 FP8-blockwise config + weight-only linear method for Cosmos3 (P6-S2 spike).

The FP8-dist deliverable (recipe ``fp8_blockwise_mixed``, blockwise-128x128) stores
each quantized module as an FP8 (e4m3) ``<m>.weight`` plus a 2D per-block
``<m>.weight_quantizer._scale`` grid (bf16, ``[ceil(rows/128), ceil(cols/128)]``) and
an ``._amax`` twin. The default vllm-omni FP8 path *dequantizes* every such weight to
BF16 on load (:mod:`...checkpoint_adapters.modelopt_native`), so weights are
BF16-resident and FP8 buys no VRAM relief.

This module is the *weight-resident* alternative: the 216 ``mlp.*``/``mlp_moe_gen.*``
targets stay FP8-resident (1 byte/elem) and are dequantized to BF16 **per operation**
for a normal BF16 GEMM (W8A16 — weights FP8, activations BF16), reusing the tested
:func:`...modelopt_native.dequantize_weight` calculation. ``lm_head`` and every
non-target stay/become BF16 at compute (INV-7). It mirrors the NVFP4 W4A16 structure
(:mod:`vllm_omni.quantization.nvfp4_blockwise`).

Selection is gated: FP8-dist's ``transformer/config.json`` carries no ``quant_recipe``
(unlike NVFP4) and the checkpoint is immutable, so the path is opt-in via the
``COSMOS3_FP8_W8A16`` env flag, resolved at transformer construction. Flag unset ⇒ the
dequant-on-load path is served unchanged (INV-6 fallback + the GATE-S2-W8A16 NO-GO
escape).

Pure calculation (no I/O): :func:`is_target_prefix`. Compute (Action at the GEMM
boundary): :class:`Fp8BlockwiseW8A16LinearMethod`. Config factories:
:func:`build_fp8_blockwise_w8a16_config`, :func:`maybe_build_fp8_blockwise_w8a16_config`.
"""

import re

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase

logger = init_logger(__name__)

RECIPE = "fp8_blockwise_mixed"
BLOCK = (128, 128)  # blockwise-128x128 (declared in quantization_config.json)

# Deploy-side marker (mirrors the native adapters): its presence + this string in the
# serve log proves the W8A16-resident path (not dequant-on-load) is engaged.
W8A16_MARKER = "cosmos3-fp8-blockwise-w8a16/p6s2"

# The 216 quantized MLP projections (both UND ``mlp.*`` and GEN ``mlp_moe_gen.*``).
# ``lm_head`` is deliberately EXCLUDED (INV-7: ``lm_head`` stays BF16 at compute; the
# FP8 ``lm_head`` is dequantized to BF16 by the adapter). Anchored on the module
# prefix (e.g. ``language_model.layers.0.mlp.gate_proj``), not a parameter name.
_TARGET_RE = re.compile(r"\.(mlp|mlp_moe_gen)\.(gate_proj|up_proj|down_proj)$")


def is_target_prefix(prefix: str) -> bool:
    """True iff *prefix* is one of the quantized MLP projections (pure).

    Matches the module prefix, not a parameter name; ``lm_head``, attention, and the
    small projections never match.
    """
    return bool(_TARGET_RE.search(prefix))


class Fp8BlockwiseW8A16LinearMethod(LinearMethodBase):
    """Weight-only FP8 blockwise (W8A16) linear method: resident FP8 + JIT dequant.

    Keeps the target weight resident as ``float8_e4m3fn`` (1 byte/elem) plus its 2D
    128x128 block scale, and computes each op by dequantizing the weight to the
    activation dtype (BF16) per forward, then a plain GEMM. No activation
    quantization (weight-only): no ``input_scale`` is created. Mirrors
    ``ModelOptNvFp4W4A16LinearMethod`` but with a JIT dequant + ``F.linear`` compute
    path (no fused kernel), which is the guaranteed-correct route on sm_120.
    """

    def __init__(self, quant_config) -> None:
        self.quant_config = quant_config
        self.block = tuple(getattr(quant_config, "weight_block_size", BLOCK))

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        block_n, block_k = self.block

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = [block_n, block_k]

        from vllm.model_executor.parameter import (
            BlockQuantScaleParameter,
            ModelWeightParameter,
        )

        # Resident FP8 weight (1 byte/elem) — never dequantized on load.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # 2D per-block scale grid (bf16 on disk), blocked 128x128 along both dims.
        scale_rows = (output_size_per_partition + block_n - 1) // block_n
        scale_cols = (input_size_per_partition + block_k - 1) // block_k
        weight_scale = BlockQuantScaleParameter(
            data=torch.empty(scale_rows, scale_cols, dtype=torch.bfloat16),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Assert real FP8 residency (kills silent dequant-on-load) and log proof."""
        weight = layer.weight
        if weight.dtype != torch.float8_e4m3fn or weight.element_size() != 1:
            raise ValueError(
                f"W8A16 residency violated: weight dtype={weight.dtype} "
                f"element_size={weight.element_size()} (expected float8_e4m3fn, 1 "
                "byte). A silent dequant-on-load would defeat the VRAM goal."
            )
        logger.info(
            "W8A16 resident target: dtype=%s elem=%d shape=%s block=%s (marker: %s)",
            weight.dtype,
            weight.element_size(),
            tuple(weight.shape),
            layer.weight_block_size,
            W8A16_MARKER,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dequantize the resident FP8 weight to ``x.dtype`` per op, then GEMM.

        Reuses the same block-scale dequant calculation as the load-time dequant
        path, so W8A16 is numerically a *deferred* form of it (near-bit-exact parity).
        """
        from vllm_omni.diffusion.model_loader.checkpoint_adapters.modelopt_native import (  # noqa: E501
            dequantize_weight,
        )

        weight = dequantize_weight(
            layer.weight, layer.weight_scale, x.dtype, tuple(layer.weight_block_size)
        )
        return F.linear(x, weight, bias)


def build_fp8_blockwise_w8a16_config():
    """Build the weight-only W8A16 FP8-blockwise config (target-inclusion).

    Subclasses vLLM's ``ModelOptFp8Config`` (mirroring how the NVFP4 W4A16 config
    subclasses ``ModelOptNvFp4Config``): inverts layer selection to quantize ONLY the
    ``.mlp``/``.mlp_moe_gen.{gate,up,down}_proj`` targets and pins
    :class:`Fp8BlockwiseW8A16LinearMethod` as the linear method; every other Linear
    (attention, ``lm_head``, projections, norms) resolves to
    ``UnquantizedLinearMethod`` (BF16).
    """
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config

    class Fp8BlockwiseW8A16Config(ModelOptFp8Config):
        """W8A16 FP8-blockwise with target-inclusion selection (only MLP projs)."""

        def is_target_module(self, prefix: str) -> bool:
            return is_target_prefix(prefix)

        def is_layer_excluded(self, prefix: str, *args, **kwargs) -> bool:
            # Invert vLLM's exclusion default: quantize ONLY the MLP targets; every
            # other Linear is excluded -> UnquantizedLinearMethod (BF16). *args/**kwargs
            # tolerate base-signature drift across vLLM builds.
            return not self.is_target_module(prefix)

    cfg = Fp8BlockwiseW8A16Config(
        quant_method="FP8",
        is_checkpoint_fp8_serialized=True,
        kv_cache_quant_method=None,
        exclude_modules=[],
    )
    # The method reads the declared block size; pin it and the weight-only method.
    cfg.weight_block_size = list(BLOCK)
    cfg.LinearMethodCls = Fp8BlockwiseW8A16LinearMethod
    logger.info(
        "Built FP8 blockwise W8A16 config (recipe=%s): target-inclusion on "
        "'.mlp|.mlp_moe_gen.{gate,up,down}_proj', method=%s, block=%s",
        RECIPE,
        cfg.LinearMethodCls.__name__,
        tuple(BLOCK),
    )
    return cfg


def maybe_build_fp8_blockwise_w8a16_config(enabled, active_quant_config=None):
    """Return the W8A16 config when *enabled*, else *active_quant_config* unchanged.

    *enabled* is the resolved ``COSMOS3_FP8_W8A16`` opt-in. Mirrors the intent of
    :func:`vllm_omni.quantization.nvfp4_blockwise.maybe_build_nvfp4_blockwise_config`,
    but gated on an explicit flag rather than a checkpoint ``quant_recipe`` (which the
    FP8-dist checkpoint lacks). Never overrides an already-active config (e.g. the
    NVFP4 W4A16 config): W8A16 only fills the FP8-dist case where no ``quant_recipe``
    hook produced one.
    """
    if not enabled:
        return active_quant_config
    if active_quant_config is not None:
        logger.info(
            "COSMOS3_FP8_W8A16 set but a quant_config (%s) is already active; "
            "keeping it (W8A16 does not override NVFP4/explicit configs).",
            getattr(active_quant_config, "get_name", lambda: active_quant_config)(),
        )
        return active_quant_config
    return build_fp8_blockwise_w8a16_config()
