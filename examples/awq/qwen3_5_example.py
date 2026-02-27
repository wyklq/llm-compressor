from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# Select model and load it.
MODEL_ID = "Qwen/Qwen3.5-27B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Configure the quantization algorithm to run.
# Qwen3.5-27B is dense with hybrid attention; AWQ mappings use default
# full_attention_interval=4 (full_attention at layers 3,7,11,...).
# Exclude: first layer (quantization-sensitive), self_attn q/k/v_proj (GQA dimensions),
# linear_attn in_proj_b/a (out_features=48, not divisible by group_size=128),
# and MTP heads.
recipe = [
    AWQModifier(
        ignore=[
            "lm_head",
            "re:model\\.layers\\.0\\.",
            "re:.*self_attn\\.q_proj$",
            "re:.*self_attn\\.k_proj$",
            "re:.*self_attn\\.v_proj$",
            "re:.*linear_attn\\.in_proj_b$",
            "re:.*linear_attn\\.in_proj_a$",
            "re:.*mtp.*",
        ],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
