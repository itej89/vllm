# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.logger import init_logger
from vllm.utils.torch_utils import _resolve_layer_name

logger = init_logger(__name__)

# Track first-call to avoid log spam
_first_wrapper_call: dict[str, bool] = {}


def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # Import at runtime to avoid circular dependency
    from vllm.model_executor.layers.attention.attention import get_attention_context

    # Inspect the signature ONCE when the decorator is applied.
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find the index of 'layer_name' parameter.
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e

    @wraps(func)
    def wrapper(*args, **kwargs):
        has_group = has_kv_transfer_group()
        is_v1 = is_v1_kv_transfer_group() if has_group else False

        # Log once per func to confirm decorator fires (before any early exit)
        key = f"{func.__name__}_{id(wrapper)}"
        if key not in _first_wrapper_call:
            _first_wrapper_call[key] = True
            logger.debug(
                "KVTRANSFER_DECORATOR_ENTRY func=%s has_kv_transfer_group=%s "
                "is_v1=%s",
                func.__name__, has_group, is_v1,
            )

        if not has_group or not is_v1:
            return func(*args, **kwargs)

        layer_name = _resolve_layer_name(args[layer_name_index])

        # Extract attention context (metadata, layer, kv_cache, layer_slot_mapping)
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        connector = get_kv_transfer_group()
        has_meta = connector.has_connector_metadata()

        # Log once per layer
        lkey = f"{layer_name}_{id(connector)}"
        if lkey not in _first_wrapper_call:
            _first_wrapper_call[lkey] = True
            logger.debug(
                "KVTRANSFER_DECORATOR layer=%s attn_metadata_is_none=%s "
                "has_connector_metadata=%s",
                layer_name,
                attn_metadata is None,
                has_meta,
            )

        if attn_metadata is None or not has_meta:
            return func(*args, **kwargs)

        # Wait for KV layer on entry
        connector.wait_for_layer_load(layer_name)

        # Execute the function
        result = func(*args, **kwargs)

        # Save KV cache layer on exit
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)

        return result

    return wrapper
