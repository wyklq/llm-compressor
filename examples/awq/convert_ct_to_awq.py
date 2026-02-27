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

Usage
-----
    python convert_ct_to_awq.py <compressed-tensors-model-dir> <output-dir>

Then:
    vllm serve <output-dir>          # auto-detects awq_marlin
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
BITS = 4
PACK_FACTOR = 32 // BITS  # 8


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

    ct_qcfg = config.get("quantization_config", {})
    config_groups = ct_qcfg.get("config_groups", {})
    group_size, zero_point, bits = 128, True, 4
    for _, grp in config_groups.items():
        wcfg = grp.get("weights", {})
        if wcfg:
            group_size = wcfg.get("group_size", 128)
            zero_point = not wcfg.get("symmetric", True)
            bits = wcfg.get("num_bits", 4)
            break

    ignore_list = ct_qcfg.get("ignore", [])
    modules_to_not_convert = [m for m in ignore_list if not m.startswith("re:")]

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

    # ── safetensors index ───────────────────────────────────────────────
    idx_file = src_path / "model.safetensors.index.json"
    if idx_file.exists():
        with open(idx_file) as f:
            idx = json.load(f)
        idx["weight_map"] = weight_map
        with open(dst_path / "model.safetensors.index.json", "w") as f:
            json.dump(idx, f, indent=2)

    # ── config.json (replace quantization_config) ───────────────────────
    awq_qcfg = {
        "quant_method": "awq",
        "bits": bits,
        "group_size": group_size,
        "zero_point": zero_point,
        "version": "gemm",
    }
    if modules_to_not_convert:
        awq_qcfg["modules_to_not_convert"] = modules_to_not_convert

    new_config = {k: v for k, v in config.items() if k != "quantization_config"}
    new_config["quantization_config"] = awq_qcfg
    with open(dst_path / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

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
