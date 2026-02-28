import os
import shutil

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# Select model and load it.
# IMPORTANT: Load with the VLM class (Qwen3_5ForConditionalGeneration), NOT
# AutoModelForCausalLM. This preserves the multimodal config (model_type: "qwen3_5")
# and the correct weight prefix (model.language_model.layers.*) so that the
# quantized model can be loaded again as a VLM without empty output issues.
MODEL_ID = "Qwen/Qwen3.5-27B"

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map=None
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "neuralmagic/calibration"
DATASET_SPLIT = "train"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, name="LLM", split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    messages = []
    for message in example["messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }


# Configure the quantization algorithm to run.
# Qwen3.5-27B is dense with hybrid attention; AWQ mappings use
# full_attention_interval=4 (full_attention at layers 3,7,11,...).
#
# Ignore list (refers QuantTrio/Qwen3.5-27B-AWQ for vLLM compat):
#   - lm_head: output head, not quantized
#   - layers.0: first decoder layer (quantization-sensitive).
#     NOTE: use "re:.*layers\\.0\\." (not "re:model\\.layers\\.0\\.")
#     because the VLM wrapper adds a "language_model" prefix.
#   - model.visual / visual: vision encoder (not quantized)
#   - linear_attn.in_proj_b/a: out_features=48, not divisible by group_size=128
#   - self_attn q/k/v: vLLM merges these into a packed QKV tensor via
#     QKVParallelLinear; quantized q/k/v are incompatible with the AWQ
#     Marlin kernel's merge path.  Keep fp16.
#   - mtp: multi-token prediction heads (not quantized)
recipe = [
    AWQModifier(
        ignore=[
            "lm_head",
            "re:.*layers\\.0\\.",
            "re:.*linear_attn\\.in_proj_b$",
            "re:.*linear_attn\\.in_proj_a$",
            "re:.*self_attn\\.q_proj$",
            "re:.*self_attn\\.k_proj$",
            "re:.*self_attn\\.v_proj$",
            "re:.*mtp.*",
            "re:model[.]visual.*",
            "re:visual.*",
        ],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    processor=processor,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

# ── Fix tokenizer for inference compatibility ──
# processor.save_pretrained() with transformers 5.x writes
# tokenizer_class="TokenizersBackend" and omits added_tokens_decoder,
# vocab.json, and merges.txt.  vLLM and older transformers expect the
# original Qwen2Tokenizer class and the full set of tokenizer files.
# Copy the originals from the HF hub cache to ensure correct byte-level
# BPE decoding at inference time.
_TOKENIZER_FILES = [
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
]
for _fname in _TOKENIZER_FILES:
    try:
        _cached = hf_hub_download(repo_id=MODEL_ID, filename=_fname)
        shutil.copy2(_cached, os.path.join(SAVE_DIR, _fname))
    except Exception:
        pass

# ── Post-processing for awq-marlin kernel compatibility ──
# The model is saved in compressed-tensors format (quant_method: "compressed-tensors").
# For vLLM to auto-detect awq_marlin kernel, convert to standard AWQ format:
#
#   python convert_ct_to_awq.py <SAVE_DIR> <SAVE_DIR>-awq-marlin
#
# This converts:
#   - weight_packed/weight_scale/weight_zero_point → qweight/scales/qzeros
#   - quant_method: "compressed-tensors" → quant_method: "awq"
#   - Applies AWQ interleave packing order

