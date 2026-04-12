# Training (SFT)

ProCAD training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). This folder contains only the ProCAD-specific deltas; the framework itself is not vendored, and the training-ready SFT JSON files are **not** shipped — you regenerate them from the released dataset (see below).

## 1. Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

## 2. Download the ProCAD dataset

From the ProCAD repo root:

```bash
hf download BBexist/ProCAD --repo-type dataset --local-dir ./dataset
```

You should now have:

```
dataset/
├── train_val_test.json
├── improved.json
├── selected_misleading_samples_train
└── selected_misleading_samples_test
```

## 3. Build SFT training files

Create the two SFT JSONs LLaMA-Factory will read:

```bash
mkdir -p training/data
```

### Clarifier SFT (`misleading_sft_data.json`)

Generated from `selected_misleading_samples_train` via the provided converter:

```bash
python to_sft.py \
  --input dataset/selected_misleading_samples_train \
  --output training/data/misleading_sft_data.json
```

### Coder SFT (`cadquery_sft.json`)

Generated from `improved.json` (the 10k clean prompts + final CadQuery code). One-liner:

```bash
python - <<'PY'
import json
with open("dataset/improved.json") as f:
    samples = json.load(f)["samples"]

out = [
    {
        "uid": s["uid"],
        "instruction": "Generate CadQuery Python code for the following 3D shape:\n\n"
                       + s["final_modified_prompt"],
        "input": "",
        "output": s["final_generated_code"],
        "system": "",
    }
    for s in samples
    if s.get("final_modified_prompt") and s.get("final_generated_code")
]

with open("training/data/cadquery_sft.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote {len(out)} samples")
PY
```

## 4. Wire into LLaMA-Factory

```bash
# Copy generated SFT files into LLaMA-Factory's data dir
cp training/data/*.json LLaMA-Factory/data/

# Merge ProCAD dataset entries into LLaMA-Factory's dataset_info.json
# (append the objects in training/dataset_info.json to LLaMA-Factory/data/dataset_info.json)

# Copy training configs
cp training/configs/qwen2-5_full_sft.yaml            LLaMA-Factory/examples/train_full/
cp training/configs/qwen2-5_full_sft_misleading.yaml LLaMA-Factory/examples/train_full/
```

## 5. Run training

```bash
cd LLaMA-Factory

# Coder (CadQuery code generation)
llamafactory-cli train examples/train_full/qwen2-5_full_sft.yaml

# Clarifier (misleading-prompt detection + clarifying questions)
llamafactory-cli train examples/train_full/qwen2-5_full_sft_misleading.yaml
```

## Released checkpoints

If you'd rather skip training and use our checkpoints:

- **Coder** — [`BBexist/ProCAD-coder`](https://huggingface.co/BBexist/ProCAD-coder)
- **Clarifier** — [`BBexist/ProCAD-clarifier`](https://huggingface.co/BBexist/ProCAD-clarifier)
