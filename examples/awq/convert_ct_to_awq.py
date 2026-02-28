#!/usr/bin/env python3
"""
Convert a compressed-tensors AWQ model (from llmcompressor) to standard
AutoAWQ format so vLLM auto-detects it as awq_marlin at runtime.

Weight layout differences
-------------------------
compressed-tensors (CT):
    weight_packed : [out, in//8]   packed along input  dim (dim=1)
    weight_scale  : [out, groups]
    weight_zero_point : [out//8, groups] packed along output dim (dim=0)

AutoAWQ:
    qweight : [in, out//8]  packed along output dim (dim=1), AWQ interleave
    scales  : [groups, out]
    qzeros  : [groups, out//8] packed along output dim (dim=1), AWQ interleave

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
Within every group of 8 output values the packed int32 stores them as
    packed = v[0] | v[2]<<4 | v[4]<<8 | v[6]<<12
           | v[1]<<16 | v[3]<<20 | v[5]<<24 | v[7]<<28

Symmetric quantization note
----------------------------
compressed-tensors uses uint4b8 for symmetric: stored values 0-15 with an
implicit bias of -8.  AWQ Marlin only supports uint4 (with explicit zero
points), so we set zero_point=true and fill qzeros with 8 to achieve the
same dequantization: float = (val - 8) * scale.

Usage
-----
    python convert_ct_to_awq.py <compressed-tensors-model-dir> <output-dir>

Then:
    vllm serve <output-dir>          # auto-detects awq_marlin
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
BITS = 4
PACK_FACTOR = 32 // BITS  # 8
SYMMETRIC_ZP_VALUE = 8  # uint4b8 midpoint bias


# ── helpers ──────────────────────────────────────────────────────────────────


def _unpack_int4(packed: torch.Tensor, pack_dim: int) -> torch.Tensor:
    """Sequential (non-interleaved) int4 unpack along *pack_dim*."""
    packed = packed.movedim(pack_dim, -1)
    shifts = torch.arange(0, 32, BITS, dtype=torch.int32)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32)
    unpacked = unpacked.reshape(*packed.shape[:-1], -1)
    return unpacked.movedim(-1, pack_dim)


def _pack_int4(vals: torch.Tensor, pack_dim: int) -> torch.Tensor:
    """Sequential int4 pack along *pack_dim*."""
    vals = vals.movedim(pack_dim, -1)
    vals = vals.reshape(*vals.shape[:-1], -1, PACK_FACTOR)
    shifts = torch.arange(0, 32, BITS, dtype=torch.int32)
    packed = ((vals & 0xF) << shifts).sum(dim=-1).to(torch.int32)
    return packed.movedim(-1, pack_dim)


def _apply_awq_interleave(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Reorder within every group of 8 along *dim* using AWQ_ORDER."""
    order = torch.tensor(AWQ_ORDER, dtype=torch.long)
    idx = torch.arange(t.shape[dim]).reshape(-1, PACK_FACTOR)
    idx = idx[:, order].reshape(-1)
    return t.index_select(dim, idx)


def _make_symmetric_qzeros(num_groups: int, out_features: int) -> torch.Tensor:
    """Create qzeros filled with SYMMETRIC_ZP_VALUE=8 in AWQ packed format.

    All 8 nibbles in each int32 are the same value (8), so the AWQ
    interleave is a no-op.  0x88888888 as signed int32 = -2004318072.
    """
    # Build one packed int32 with 8 nibbles each set to 8, then broadcast.
    single = torch.zeros(1, dtype=torch.int32)
    for i in range(PACK_FACTOR):
        single |= SYMMETRIC_ZP_VALUE << (i * BITS)
    return single.expand(num_groups, out_features // PACK_FACTOR).contiguous()


# ── per-layer conversion ────────────────────────────────────────────────────


def _convert_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    """CT weight_packed [out, in//8] → AWQ qweight [in, out//8]."""
    w = _unpack_int4(weight_packed, pack_dim=1)  # [out, in]
    w = w.t().contiguous()                        # [in, out]
    w = _apply_awq_interleave(w, dim=1)
    return _pack_int4(w, pack_dim=1)              # [in, out//8]


def _convert_scales(weight_scale: torch.Tensor) -> torch.Tensor:
    """CT weight_scale [out, groups] → AWQ scales [groups, out]."""
    return weight_scale.t().contiguous()


def _convert_zeros(weight_zp: torch.Tensor) -> torch.Tensor:
    """CT weight_zero_point [out//8, groups] → AWQ qzeros [groups, out//8]."""
    zp = _unpack_int4(weight_zp, pack_dim=0)  # [out, groups]
    zp = zp.t().contiguous()                    # [groups, out]
    zp = _apply_awq_interleave(zp, dim=1)
    return _pack_int4(zp, pack_dim=1)           # [groups, out//8]


# ── config helpers ───────────────────────────────────────────────────────────


def _find_ct_qcfg(config: dict) -> tuple[dict, str]:
    """Locate the compressed-tensors quantization_config.

    For VLMs (e.g. Qwen3_5ForConditionalGeneration) it lives inside
    text_config; for pure text models it's at the top level.
    Returns (ct_qcfg_dict, location) where location is 'top' or 'text_config'.
    """
    top = config.get("quantization_config", {})
    if top.get("quant_method") == "compressed-tensors":
        return top, "top"

    tc = config.get("text_config", {})
    nested = tc.get("quantization_config", {})
    if nested.get("quant_method") == "compressed-tensors":
        return nested, "text_config"

    return top if top else nested, "top"


def _regex_to_literal(pattern: str) -> str | None:
    """Extract a usable literal substring from a compressed-tensors regex.

    Handles common patterns produced by llmcompressor's ignore list:
      ``model\\.layers\\.0\\.``  →  ``model.layers.0.``
      ``.*self_attn\\.q_proj$``  →  ``self_attn.q_proj``
      ``.*mtp.*``                →  ``mtp``
    """
    pattern = pattern.rstrip("$")

    # Strip leading/trailing greedy wildcards (.*  .+)
    pattern = re.sub(r"^(\.\*|\.\+)+", "", pattern)
    pattern = re.sub(r"(\.\*|\.\+)+$", "", pattern)

    literal: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            if nxt in r"\.[](){}*+?|^$":
                literal.append(nxt)
                i += 2
                continue
            break  # \d, \w, etc. – not a plain literal
        if c in r".*+?[](){}|^$":
            break  # unescaped metacharacter
        literal.append(c)
        i += 1

    s = "".join(literal)
    return s if s else None


def _build_modules_to_not_convert(ignore_list: list[str]) -> list[str]:
    """Convert compressed-tensors ignore list to AWQ modules_to_not_convert.

    AWQ uses substring matching (skip_with_substr=True), so we convert
    regex patterns like ``re:visual.*`` to plain substrings like ``visual``.
    """
    result = []
    for entry in ignore_list:
        if entry.startswith("re:"):
            lit = _regex_to_literal(entry[3:])
            if lit:
                result.append(lit)
        else:
            result.append(entry)
    return result


# ── main ─────────────────────────────────────────────────────────────────────

CT_SUFFIXES = {
    ".weight_packed",
    ".weight_scale",
    ".weight_zero_point",
    ".weight_shape",
    ".weight_g_idx",
}


def _strip_ct_suffix(key: str):
    for s in CT_SUFFIXES:
        if key.endswith(s):
            return key[: -len(s)], s
    return None, None


def convert_model(src: str, dst: str):
    src_path, dst_path = Path(src), Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    # ── parse compressed-tensors quantization config ────────────────────
    with open(src_path / "config.json") as f:
        config = json.load(f)

    ct_qcfg, ct_location = _find_ct_qcfg(config)
    config_groups = ct_qcfg.get("config_groups", {})

    group_size, symmetric, bits = 128, True, 4
    for _, grp in config_groups.items():
        wcfg = grp.get("weights", {})
        if wcfg:
            group_size = wcfg.get("group_size", 128)
            symmetric = wcfg.get("symmetric", True)
            bits = wcfg.get("num_bits", 4)
            break

    # awq_marlin only supports uint4 (has_zp=True).  For symmetric
    # quantization we set zero_point=True with qzeros = 8 (see docstring).
    zero_point = True

    ignore_list = ct_qcfg.get("ignore", [])
    modules_to_not_convert = _build_modules_to_not_convert(ignore_list)

    print(f"Source config location: {ct_location}")
    print(f"  bits={bits}  group_size={group_size}  symmetric={symmetric}")
    print(f"  modules_to_not_convert ({len(modules_to_not_convert)} entries):")
    for m in modules_to_not_convert:
        print(f"    - {m!r}")

    # ── process safetensors shards ──────────────────────────────────────
    st_files = sorted(src_path.glob("*.safetensors"))
    weight_map: dict[str, str] = {}

    for st_file in st_files:
        new_tensors: dict[str, torch.Tensor] = {}
        layers: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        plain_keys: list[str] = []

        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                prefix, suffix = _strip_ct_suffix(key)
                if prefix is not None:
                    layers[prefix][suffix] = f.get_tensor(key)
                else:
                    plain_keys.append(key)

            for key in plain_keys:
                new_tensors[key] = f.get_tensor(key)

        for prefix, parts in layers.items():
            wp = parts.get(".weight_packed")
            ws = parts.get(".weight_scale")
            wz = parts.get(".weight_zero_point")

            if wp is None or ws is None:
                for suffix, tensor in parts.items():
                    new_tensors[f"{prefix}{suffix}"] = tensor
                continue

            qweight = _convert_weight(wp)
            scales = _convert_scales(ws)

            if wz is not None:
                qzeros = _convert_zeros(wz)
            elif symmetric:
                num_groups, out_features = scales.shape
                qzeros = _make_symmetric_qzeros(num_groups, out_features)
            else:
                num_groups, out_features = scales.shape
                qzeros = torch.zeros(
                    num_groups, out_features // PACK_FACTOR, dtype=torch.int32
                )

            new_tensors[f"{prefix}.qweight"] = qweight
            new_tensors[f"{prefix}.scales"] = scales
            new_tensors[f"{prefix}.qzeros"] = qzeros

        save_file(new_tensors, dst_path / st_file.name)
        for k in new_tensors:
            weight_map[k] = st_file.name

        print(f"  converted {st_file.name}  ({len(new_tensors)} tensors)")

    # ── auto-detect unquantized linear layers ──────────────────────────
    quantized_prefixes = set()
    unquantized_weight_keys = set()
    for k in weight_map:
        if k.endswith(".qweight"):
            quantized_prefixes.add(k[: -len(".qweight")])
        elif k.endswith(".weight"):
            unquantized_weight_keys.add(k)

    auto_added = []
    for wkey in sorted(unquantized_weight_keys):
        prefix = wkey[: -len(".weight")]
        if prefix in quantized_prefixes:
            continue
        # Skip non-linear parameters (embeddings, norms, biases, etc.)
        if any(
            seg in prefix.rsplit(".", 1)[-1]
            for seg in ("embed", "norm", "layernorm", "head")
        ):
            continue
        # Check if already covered by existing modules_to_not_convert
        if any(m in prefix for m in modules_to_not_convert):
            continue
        modules_to_not_convert.append(prefix)
        auto_added.append(prefix)

    if auto_added:
        print(f"  auto-detected {len(auto_added)} unquantized linear layers:")
        for m in auto_added:
            print(f"    + {m!r}")

    # ── safetensors index ───────────────────────────────────────────────
    idx_file = src_path / "model.safetensors.index.json"
    if idx_file.exists():
        with open(idx_file) as f:
            idx = json.load(f)
        idx["weight_map"] = weight_map
        with open(dst_path / "model.safetensors.index.json", "w") as f:
            json.dump(idx, f, indent=2)

    # ── AWQ quantization config ─────────────────────────────────────────
    awq_qcfg = {
        "quant_method": "awq",
        "bits": bits,
        "group_size": group_size,
        "zero_point": zero_point,
        "version": "gemm",
        "modules_to_not_convert": modules_to_not_convert,
    }

    # ── config.json ─────────────────────────────────────────────────────
    # Remove CT quantization_config from wherever it was, put AWQ at top level.
    new_config = dict(config)
    new_config.pop("quantization_config", None)

    if "text_config" in new_config:
        tc = dict(new_config["text_config"])
        tc.pop("quantization_config", None)
        new_config["text_config"] = tc

    new_config["quantization_config"] = awq_qcfg

    with open(dst_path / "config.json", "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)

    # ── quantize_config.json (vLLM also reads this) ─────────────────────
    with open(dst_path / "quantize_config.json", "w") as f:
        json.dump(awq_qcfg, f, indent=2)

    # ── copy remaining files (tokenizer, etc.) ──────────────────────────
    skip = {p.name for p in st_files} | {
        "config.json",
        "quantize_config.json",
        "model.safetensors.index.json",
    }
    for p in src_path.iterdir():
        if p.is_file() and p.name not in skip:
            shutil.copy2(p, dst_path / p.name)

    # ── tokenizer sanity check ───────────────────────────────────────────
    tok_cfg_path = dst_path / "tokenizer_config.json"
    if tok_cfg_path.exists():
        with open(tok_cfg_path) as f:
            tok_cfg = json.load(f)
        tok_cls = tok_cfg.get("tokenizer_class", "")
        if tok_cls == "TokenizersBackend":
            print(
                "\n⚠ WARNING: tokenizer_config.json has "
                'tokenizer_class="TokenizersBackend" (transformers 5.x default).'
                "\n  vLLM may produce garbled output.  Replace the tokenizer "
                "files with the originals from the source model:\n"
                "    tokenizer_config.json, tokenizer.json, vocab.json, merges.txt"
            )
        if not (dst_path / "vocab.json").exists():
            print(
                "\n⚠ WARNING: vocab.json not found.  Some inference engines "
                "need it for the slow tokenizer fallback."
            )

    print(f"\nDone.  AWQ model saved to {dst_path}")
    print(f"quantize_config: {json.dumps(awq_qcfg, indent=2)}")
    print(
        "\nvLLM should now log:\n"
        '  "The model is convertible to awq_marlin during runtime.'
        ' Using awq_marlin kernel."'
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert compressed-tensors AWQ → AutoAWQ format"
    )
    p.add_argument("input_dir", help="compressed-tensors model directory")
    p.add_argument("output_dir", help="output directory for AWQ model")
    args = p.parse_args()
    convert_model(args.input_dir, args.output_dir)
