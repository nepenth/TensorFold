"""Execute forward dispatch contracts without importing CUDA or PyTorch."""

from __future__ import annotations

import ast
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src/tensorfold/families/glm5_next/cuda"


def _definitions(file, names, namespace):
    tree = ast.parse((ROOT / file).read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), file, "exec"), namespace)


@pytest.fixture
def dispatch():
    ns = {"__name__": __name__, "dataclass": dataclass}
    _definitions("nvfp4_table.py", {"RoutedTable"}, ns)
    for name in ("glue", "grouped", "exl3_generic", "torch", "mm", "out_proj", "gather",
                 "dense_eager", "routed_eager", "routed_decode", "kda_block", "dsa_block"):
        ns[name] = MagicMock(name=name)
    ns["prof"] = SimpleNamespace(timed=lambda label: nullcontext())
    _definitions("forward.py", {"moe_block", "mlp_block", "layer_forward"}, ns)
    f = SimpleNamespace(**ns)
    owners = [SimpleNamespace(act=act) for act in (0.375, 0.375, 0.625) * 4]
    table = f.RoutedTable(owners, None, None, None, None, None, {}, experts=4)
    w = MagicMock()
    w.cfg.top_k, w.cfg.experts, w.cfg.hidden, w.cfg.limit = 2, 4, 128, 7.0
    b = MagicMock()
    layer = MagicMock()
    layer.moe = SimpleNamespace(experts=table, shared=None, router=object(), bias=object())
    return f, layer, w, b


@pytest.mark.parametrize("rows,prefill", [(1, False), (1, True), (3, False), (3, True)])
def test_packed_dispatch_excludes_grouped_and_shared_slot(dispatch, rows, prefill):
    f, layer, w, b = dispatch
    b.prefill = prefill
    assert f.moe_block(layer, w, b, rows) is f.gather.return_value
    f.grouped.route.assert_not_called()
    f.grouped.gate_up.assert_not_called()
    f.grouped.down.assert_not_called()
    b.pick.__getitem__.assert_any_call((slice(None, rows), slice(None, 2)))
    b.pick.__getitem__.return_value.to.assert_called_once_with(f.torch.int64)
    b.wts.__getitem__.assert_any_call((slice(None, rows), slice(None, 2)))
    if rows == 1:
        f.routed_eager.assert_not_called()
        f.routed_decode.assert_called_once()
        assert f.routed_decode.call_args.args[1] is layer.moe.experts
        assert f.routed_decode.call_args.kwargs == {"residual_act": 0.375, "intermediate_act": 0.625, "limit": 7.0}
    else:
        f.routed_decode.assert_not_called()
        f.routed_eager.assert_called_once()
        assert f.routed_eager.call_args.args[1] == [tuple(layer.moe.experts.linears[i:i + 3])
                                                  for i in range(0, 12, 3)]
        assert f.routed_eager.call_args.kwargs == {"backend": "prompt" if prefill else "lane", "limit": 7.0}
    f.gather.assert_called_once_with(w, b, rows)


@pytest.mark.parametrize("slot,act", [(0, None), (4, 0.5), (5, 0.5)])
def test_decode_refuses_missing_or_disagreeing_static_factors(dispatch, slot, act):
    f, layer, w, b = dispatch
    layer.moe.experts.linears[slot].act = act
    with pytest.raises(ValueError, match="common static"):
        f.moe_block(layer, w, b, 1)
    f.routed_decode.assert_not_called()
    f.routed_eager.assert_not_called()
    f.grouped.gate_up.assert_not_called()


def test_dense_packed_and_bf16_keep_their_arms(dispatch):
    f, layer, w, b = dispatch
    layer.dense_nvfp4 = SimpleNamespace(linears=layer.moe.experts.linears[:3])
    f.mlp_block(layer, w, b, 1)
    f.dense_eager.assert_called_once()
    assert f.dense_eager.call_args.args[1:4] == tuple(layer.dense_nvfp4.linears)
    f.mm.assert_not_called()
    layer.dense_nvfp4 = None
    f.mlp_block(layer, w, b, 1)
    f.mm.assert_called_once()
    f.glue.swiglu.assert_called_once()
    f.out_proj.assert_called_once()
    assert f.dense_eager.call_count == 1


def test_exl3_and_grouped_keep_their_arms(dispatch):
    f, layer, w, b = dispatch
    layer.moe.experts = object()
    layer.moe.shared = MagicMock()
    f.moe_block(layer, w, b, 1)
    f.exl3_generic.routed.assert_called_once()
    assert f.mm.call_count == 2
    f.grouped.route.assert_not_called()
    f.routed_decode.assert_not_called()
    layer.moe.shared = None
    f.moe_block(layer, w, b, 1)
    f.grouped.route.assert_called_once()
    f.grouped.gate_up.assert_called_once()
    f.grouped.down.assert_called_once()


def test_layer_with_only_dense_nvfp4_reaches_mlp_block(dispatch):
    f, layer, w, b = dispatch
    layer.kind, layer.mlp, layer.moe = "kda", None, None
    layer.dense_nvfp4 = SimpleNamespace(linears=[object(), object(), object()])
    f.layer_forward(layer, w, MagicMock(), b, 1)
    f.dense_eager.assert_called_once()
    f.glue.router.assert_not_called()


def test_nvfp4_engine_does_not_attempt_graph_capture():
    ns = {"Buffers": MagicMock(), "State": MagicMock(), "PREFILL_ROWS": 2048}
    _definitions("decode.py", {"Engine"}, ns)
    w = SimpleNamespace(cfg=SimpleNamespace(quant="modelopt"), meta={}, mtp=None, head=SimpleNamespace(n=128))
    # Graph capture imports .graphs: executing without that module MUST stay eager.
    engine = ns["Engine"](w, graphs=True)
    assert engine.graphs is None
