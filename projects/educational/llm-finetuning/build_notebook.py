#!/usr/bin/env python3
"""Build script that generates llm_finetuning_cookbook.ipynb (nbformat 4)."""
import sys as _sys
import os as _os


def _find_repo_root(start_dir):
    current = _os.path.abspath(start_dir)
    while True:
        if _os.path.exists(_os.path.join(current, "manage.sh")) and _os.path.isdir(_os.path.join(current, "kaggle_portfolio")):
            return current
        parent = _os.path.dirname(current)
        if parent == current:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        current = parent


_sys.path.insert(0, _find_repo_root(_os.path.dirname(_os.path.abspath(__file__))))
from kaggle_portfolio.shared.build_utils import md, code, write_notebook

cells = []

# ── 1. Title Banner ───────────────────────────────────────────────────────
cells.append(md(
    "# LLM Fine-Tuning Cookbook: LoRA & QLoRA\n"
    "\n"
    "![Python](https://img.shields.io/badge/Python-3.10%2B-blue)\n"
    "![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)\n"
    "![Transformers](https://img.shields.io/badge/Transformers-4.40%2B-orange)\n"
    "![PEFT](https://img.shields.io/badge/PEFT-0.11%2B-green)\n"
    "![License](https://img.shields.io/badge/License-MIT-lightgrey)\n"
    "\n"
    "---"
))

# ── 2. TL;DR ─────────────────────────────────────────────────────────────
cells.append(md(
    "## TL;DR\n"
    "\n"
    "Fine-tuning large language models used to require **hundreds of GBs** of VRAM.\n"
    "With **LoRA** (Low-Rank Adaptation) and **QLoRA** (Quantized LoRA) you can\n"
    "fine-tune a 7 B-parameter model on a **single 16 GB GPU** while retaining\n"
    "> 97 % of full fine-tuning quality.\n"
    "\n"
    "This notebook walks through **everything** from first principles to a\n"
    "production-ready training pipeline you can submit to Kaggle competitions."
))

# ── 3. Table of Contents ─────────────────────────────────────────────────
cells.append(md(
    "## Table of Contents\n"
    "\n"
    "1. [Setup & Imports](#1-setup--imports)\n"
    "2. [Understanding LLM Architecture](#2-understanding-llm-architecture)\n"
    "3. [Full Fine-Tuning vs Parameter-Efficient](#3-full-fine-tuning-vs-parameter-efficient)\n"
    "4. [LoRA Deep Dive](#4-lora-deep-dive)\n"
    "5. [QLoRA Explained](#5-qlora-explained)\n"
    "6. [Practical Fine-Tuning Pipeline](#6-practical-fine-tuning-pipeline)\n"
    "7. [Advanced Techniques](#7-advanced-techniques)\n"
    "8. [Evaluation & Inference](#8-evaluation--inference)\n"
    "9. [Deployment](#9-deployment)\n"
    "10. [Key Takeaways & Further Reading](#10-key-takeaways--further-reading)"
))

# ── 4. Section 1 Header ──────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 1. Setup & Imports\n"
    "\n"
    "We install and import the core libraries.\n"
    "On Kaggle the GPU runtime already has most of these, but we pin versions\n"
    "for reproducibility."
))

# ── 5. Install / Import code ─────────────────────────────────────────────
cells.append(code(
    "# ── Install Kaggle-missing deps (trl ships outside the base image) ──\n"
    "!pip install -q trl bitsandbytes\n"
    "\n"
    "import os\n"
    "import random\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "from transformers import (\n"
    "    AutoModelForCausalLM,\n"
    "    AutoTokenizer,\n"
    "    BitsAndBytesConfig,\n"
    "    TrainingArguments,\n"
    ")\n"
    "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel\n"
    "from datasets import load_dataset\n"
    "from trl import SFTTrainer\n"
    "\n"
    "SEED = 42\n"
    "os.environ[\"PYTHONHASHSEED\"] = str(SEED)\n"
    "random.seed(SEED)\n"
    "np.random.seed(SEED)\n"
    "torch.manual_seed(SEED)\n"
    "if torch.cuda.is_available():\n"
    "    torch.cuda.manual_seed_all(SEED)\n"
    "    torch.backends.cudnn.deterministic = True\n"
    "    torch.backends.cudnn.benchmark = False\n"
    "\n"
    "print(f\"PyTorch  : {torch.__version__}\")\n"
    "print(f\"CUDA     : {torch.cuda.is_available()}\")\n"
    "print(f\"Seed     : {SEED}\")\n"
    "if torch.cuda.is_available():\n"
    "    print(f\"GPU      : {torch.cuda.get_device_name(0)}\")\n"
    "    print(f\"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
))

# ── 6. Key Takeaway box ──────────────────────────────────────────────────
cells.append(md(
    "> **Key Takeaway -- Setup**\n"
    ">\n"
    "> The stack is **torch + transformers + peft + bitsandbytes + trl**.\n"
    "> `trl.SFTTrainer` wraps HuggingFace `Trainer` with LoRA/QLoRA best practices\n"
    "> baked in."
))

# ── 7. Section 2 Header ──────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 2. Understanding LLM Architecture\n"
    "\n"
    "Before fine-tuning, we need to know **what** we are tuning.\n"
    "A decoder-only transformer (GPT-style) stacks *N* identical blocks, each\n"
    "containing a multi-head self-attention layer and a feed-forward network (FFN)."
))

# ── 8. Parameter counting ────────────────────────────────────────────────
cells.append(code(
    "def count_parameters(model):\n"
    "    \"\"\"Return total and trainable parameter counts.\"\"\"\n"
    "    total  = sum(p.numel() for p in model.parameters())\n"
    "    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
    "    return total, train\n"
    "\n"
    "# Example: a small GPT-2 style model\n"
    "from transformers import GPT2LMHeadModel, GPT2Config\n"
    "\n"
    "cfg = GPT2Config(n_layer=12, n_head=12, n_embd=768)\n"
    "demo_model = GPT2LMHeadModel(cfg)\n"
    "total, trainable = count_parameters(demo_model)\n"
    "print(f\"Total params     : {total / 1e6:.1f} M\")\n"
    "print(f\"Trainable params : {trainable / 1e6:.1f} M\")"
))

# ── 9. Memory estimation ─────────────────────────────────────────────────
cells.append(code(
    "def estimate_memory_gb(num_params, dtype_bytes=2, optimizer_factor=2):\n"
    "    \"\"\"Rough GPU memory estimate for training.\n"
    "\n"
    "    Parameters\n"
    "    ----------\n"
    "    num_params : int\n"
    "    dtype_bytes : int  (2 = fp16, 4 = fp32)\n"
    "    optimizer_factor : int  (2 for AdamW states: m + v)\n"
    "    \"\"\"\n"
    "    model_mem   = num_params * dtype_bytes\n"
    "    grad_mem    = num_params * dtype_bytes\n"
    "    optim_mem   = num_params * 4 * optimizer_factor  # optimizer states in fp32\n"
    "    total_bytes = model_mem + grad_mem + optim_mem\n"
    "    return total_bytes / (1024 ** 3)\n"
    "\n"
    "for size_b in [1, 3, 7, 13, 70]:\n"
    "    mem = estimate_memory_gb(size_b * 1e9)\n"
    "    print(f\"{size_b:>3d}B params -> ~{mem:6.1f} GB (fp16 + AdamW)\")"
))

# ── 10. Takeaway ──────────────────────────────────────────────────────────
cells.append(md(
    "> **Key Takeaway -- Architecture**\n"
    ">\n"
    "> A 7 B model needs **~112 GB** just for weights + gradients + optimizer.\n"
    "> That is far beyond a single consumer GPU.  We need a smarter approach."
))

# ── 11. Section 3 Header ─────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 3. Full Fine-Tuning vs Parameter-Efficient\n"
    "\n"
    "| Aspect | Full Fine-Tuning | LoRA / QLoRA |\n"
    "|--------|------------------|--------------|\n"
    "| Trainable params | 100 % | 0.1 -- 2 % |\n"
    "| GPU VRAM (7 B) | ~112 GB | ~6 -- 16 GB |\n"
    "| Training speed | Baseline | 1.5 -- 3x faster |\n"
    "| Risk of catastrophic forgetting | Higher | Lower |\n"
    "| Checkpoint size | Full model | Adapter only (few MB) |"
))

# ── 12. Memory comparison code ────────────────────────────────────────────
cells.append(code(
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "rows = []\n"
    "for size_b in [1, 3, 7, 13]:\n"
    "    n = size_b * 1e9\n"
    "    full_mem  = estimate_memory_gb(n)\n"
    "    # LoRA: only ~1% params are trainable but we still load the model in fp16\n"
    "    lora_mem  = (n * 2) / (1024**3) + estimate_memory_gb(n * 0.01)\n"
    "    # QLoRA: model in 4-bit, trainable adapters in fp16\n"
    "    qlora_mem = (n * 0.5) / (1024**3) + estimate_memory_gb(n * 0.01)\n"
    "    rows.append({\"Model\": f\"{size_b}B\", \"Full FT (GB)\": f\"{full_mem:.1f}\",\n"
    "                 \"LoRA (GB)\": f\"{lora_mem:.1f}\", \"QLoRA (GB)\": f\"{qlora_mem:.1f}\"})\n"
    "\n"
    "df = pd.DataFrame(rows)\n"
    "print(df.to_string(index=False))\n"
    "\n"
    "# Convert formatted strings back to numeric for charting\n"
    "plot_df = df.copy()\n"
    "for col in [\"Full FT (GB)\", \"LoRA (GB)\", \"QLoRA (GB)\"]:\n"
    "    plot_df[col] = plot_df[col].astype(float)\n"
    "\n"
    "ax = plot_df.set_index(\"Model\")[[\"Full FT (GB)\", \"LoRA (GB)\", \"QLoRA (GB)\"]].plot(\n"
    "    kind=\"bar\",\n"
    "    figsize=(10, 5),\n"
    ")\n"
    "ax.set_title(\"VRAM Comparison: Full FT vs LoRA vs QLoRA\")\n"
    "ax.set_ylabel(\"Estimated Memory (GB)\")\n"
    "ax.grid(axis=\"y\", alpha=0.25)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── 13. Takeaway ──────────────────────────────────────────────────────────
cells.append(md(
    "> **Key Takeaway -- Efficiency**\n"
    ">\n"
    "> QLoRA lets you fine-tune a **7 B model in ~6 GB VRAM** -- that fits on a\n"
    "> free Kaggle T4 GPU!"
))

# ── 13b. Insight note ──────────────────────────────────────────────────────
cells.append(md(
    "### Insight: Memory vs Quality Trade-off\n"
    "\n"
    "- **Observation:** the memory gap between full fine-tuning and QLoRA grows as model size increases.\n"
    "- **Because** QLoRA keeps the frozen base in 4-bit, large models remain trainable on commodity GPUs.\n"
    "- **Therefore**, you can iterate faster on prompts, data curation, and evaluation instead of waiting on infra.\n"
    "- **Limitation:** extremely low-rank adapters may underfit niche domain style unless you tune rank and alpha."
))

# ── 14. Section 4 Header ─────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 4. LoRA Deep Dive\n"
    "\n"
    "### The Math\n"
    "\n"
    "Instead of updating the full weight matrix $W \\in \\mathbb{R}^{d \\times d}$,\n"
    "LoRA decomposes the update into two low-rank matrices:\n"
    "\n"
    "$$W' = W + \\Delta W = W + BA$$\n"
    "\n"
    "where $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times d}$,\n"
    "with rank $r \\ll d$.\n"
    "\n"
    "**Parameter savings:** $d^2 \\to 2dr$. For $d = 4096, r = 16$ this is\n"
    "$16.7\\text{M} \\to 131\\text{K}$ -- a **128x** reduction."
))

# ── 15. LoRA from scratch ────────────────────────────────────────────────
cells.append(code(
    "class LoRALayer(nn.Module):\n"
    "    \"\"\"Minimal LoRA layer implemented from scratch.\"\"\"\n"
    "\n"
    "    def __init__(self, in_features, out_features, rank=8, alpha=16):\n"
    "        super().__init__()\n"
    "        self.linear = nn.Linear(in_features, out_features, bias=False)\n"
    "        self.linear.weight.requires_grad = False  # freeze original\n"
    "\n"
    "        # Low-rank adapter matrices\n"
    "        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)\n"
    "        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))\n"
    "        self.scaling = alpha / rank\n"
    "\n"
    "    def forward(self, x):\n"
    "        base = self.linear(x)\n"
    "        # Low-rank path:  x @ A^T @ B^T  (scaled)\n"
    "        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling\n"
    "        return base + lora\n"
    "\n"
    "\n"
    "# Quick sanity check\n"
    "layer = LoRALayer(768, 768, rank=16, alpha=32)\n"
    "total, trainable = count_parameters(layer)\n"
    "print(f\"Frozen   : {(total - trainable):,}\")\n"
    "print(f\"Trainable: {trainable:,}  ({100*trainable/total:.2f} %)\")\n"
    "\n"
    "x = torch.randn(2, 10, 768)\n"
    "y = layer(x)\n"
    "print(f\"Output shape: {y.shape}\")"
))

# ── 16. Rank ablation ────────────────────────────────────────────────────
cells.append(code(
    "# Effect of rank on parameter count\n"
    "d = 4096\n"
    "print(f\"{'Rank':>6s}  {'LoRA Params':>14s}  {'% of Full':>10s}\")\n"
    "print(\"-\" * 35)\n"
    "for r in [4, 8, 16, 32, 64, 128, 256]:\n"
    "    lora_params = 2 * d * r\n"
    "    full_params = d * d\n"
    "    print(f\"{r:>6d}  {lora_params:>14,}  {100*lora_params/full_params:>9.2f}%\")"
))

# ── 17. Takeaway ──────────────────────────────────────────────────────────
cells.append(md(
    "> **Key Takeaway -- LoRA**\n"
    ">\n"
    "> Rank 16 is the sweet spot for most 7 B models -- it gives ~0.2 % trainable\n"
    "> parameters and matches full FT quality on most benchmarks."
))

# ── 18. Section 5 Header ─────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 5. QLoRA Explained\n"
    "\n"
    "QLoRA adds **4-bit NormalFloat (NF4)** quantization on top of LoRA:\n"
    "\n"
    "1. Load the base model in 4-bit precision (0.5 bytes / param)\n"
    "2. Attach LoRA adapters in fp16 / bf16\n"
    "3. Train only the adapters; back-prop through the quantized base via\n"
    "   *double quantization* and *paged optimizers*.\n"
    "\n"
    "This cuts VRAM by an additional **~3x** compared to fp16 LoRA."
))

# ── 19. BitsAndBytesConfig ────────────────────────────────────────────────
cells.append(code(
    "bnb_config = BitsAndBytesConfig(\n"
    "    load_in_4bit=True,\n"
    "    bnb_4bit_quant_type=\"nf4\",            # NormalFloat 4-bit\n"
    "    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16\n"
    "    bnb_4bit_use_double_quant=True,          # double quantization\n"
    ")\n"
    "\n"
    "print(\"BitsAndBytesConfig ready:\")\n"
    "print(f\"  load_in_4bit        = {bnb_config.load_in_4bit}\")\n"
    "print(f\"  quant_type          = {bnb_config.bnb_4bit_quant_type}\")\n"
    "print(f\"  compute_dtype       = {bnb_config.bnb_4bit_compute_dtype}\")\n"
    "print(f\"  double_quant        = {bnb_config.bnb_4bit_use_double_quant}\")"
))

# ── 20. Takeaway ──────────────────────────────────────────────────────────
cells.append(md(
    "> **Key Takeaway -- QLoRA**\n"
    ">\n"
    "> The magic is `load_in_4bit=True` + `bnb_4bit_quant_type=\"nf4\"` +\n"
    "> `bnb_4bit_use_double_quant=True`. Three flags, 3x VRAM savings."
))

# ── 21. Section 6 Header ─────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 6. Practical Fine-Tuning Pipeline\n"
    "\n"
    "Let's put it all together with a **real** model and dataset.\n"
    "We'll fine-tune a small model as a demonstration -- swap in any HF model ID\n"
    "for your competition."
))

# ── 22. Load model + tokenizer ────────────────────────────────────────────
cells.append(code(
    "MODEL_ID = \"TinyLlama/TinyLlama-1.1B-Chat-v1.0\"  # swap for your target model\n"
    "\n"
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)\n"
    "tokenizer.pad_token = tokenizer.eos_token\n"
    "tokenizer.padding_side = \"right\"\n"
    "\n"
    "# Keep trust_remote_code disabled by default; only enable for vetted models.\n"
    "model = AutoModelForCausalLM.from_pretrained(\n"
    "    MODEL_ID,\n"
    "    quantization_config=bnb_config,\n"
    "    device_map=\"auto\",\n"
    ")\n"
    "model = prepare_model_for_kbit_training(model)\n"
    "\n"
    "total, trainable = count_parameters(model)\n"
    "print(f\"Base model loaded  : {total / 1e6:.1f} M params\")\n"
    "print(f\"Trainable (before) : {trainable / 1e6:.1f} M params\")"
))

# ── 23. LoRA config & attach ──────────────────────────────────────────────
cells.append(code(
    "lora_config = LoraConfig(\n"
    "    r=16,\n"
    "    lora_alpha=32,\n"
    "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n"
    "                    \"gate_proj\", \"up_proj\", \"down_proj\"],\n"
    "    lora_dropout=0.05,\n"
    "    bias=\"none\",\n"
    "    task_type=\"CAUSAL_LM\",\n"
    ")\n"
    "\n"
    "model = get_peft_model(model, lora_config)\n"
    "model.print_trainable_parameters()"
))

# ── 24. Dataset ───────────────────────────────────────────────────────────
cells.append(code(
    "# Using a small instruction-tuning dataset for demonstration\n"
    "dataset = load_dataset(\"yahma/alpaca-cleaned\", split=\"train[:2000]\")\n"
    "\n"
    "def format_instruction(sample):\n"
    "    if sample[\"input\"]:\n"
    "        text = (\n"
    "            f\"### Instruction:\\n{sample['instruction']}\\n\\n\"\n"
    "            f\"### Input:\\n{sample['input']}\\n\\n\"\n"
    "            f\"### Response:\\n{sample['output']}\"\n"
    "        )\n"
    "    else:\n"
    "        text = (\n"
    "            f\"### Instruction:\\n{sample['instruction']}\\n\\n\"\n"
    "            f\"### Response:\\n{sample['output']}\"\n"
    "        )\n"
    "    return {\"text\": text}\n"
    "\n"
    "dataset = dataset.map(format_instruction)\n"
    "print(f\"Dataset size: {len(dataset)}\")\n"
    "print(f\"Sample:\\n{dataset[0]['text'][:300]}...\")"
))

# ── 25. SFTTrainer ────────────────────────────────────────────────────────
cells.append(code(
    "training_args = TrainingArguments(\n"
    "    output_dir=\"./lora-checkpoints\",\n"
    "    num_train_epochs=1,\n"
    "    per_device_train_batch_size=4,\n"
    "    gradient_accumulation_steps=4,\n"
    "    learning_rate=2e-4,\n"
    "    weight_decay=0.01,\n"
    "    warmup_ratio=0.03,\n"
    "    lr_scheduler_type=\"cosine\",\n"
    "    logging_steps=10,\n"
    "    save_strategy=\"steps\",\n"
    "    save_steps=50,\n"
    "    bf16=True,\n"
    "    optim=\"paged_adamw_8bit\",\n"
    "    gradient_checkpointing=True,\n"
    "    max_grad_norm=0.3,\n"
    "    report_to=\"none\",\n"
    ")\n"
    "\n"
    "trainer = SFTTrainer(\n"
    "    model=model,\n"
    "    train_dataset=dataset,\n"
    "    args=training_args,\n"
    "    tokenizer=tokenizer,\n"
    "    max_seq_length=512,\n"
    "    dataset_text_field=\"text\",\n"
    "    packing=False,\n"
    ")\n"
    "\n"
    "print(\"Trainer ready. Call trainer.train() to start.\")"
))

# ── 26. Train (commented) ────────────────────────────────────────────────
cells.append(code(
    "# Uncomment to actually train (takes ~15-20 min on T4):\n"
    "# trainer.train()\n"
    "# trainer.save_model(\"./lora-final\")\n"
    "print(\"Training cell ready -- uncomment to run.\")"
))

# ── 27. Takeaway ──────────────────────────────────────────────────────────
cells.append(md(
    "> **Key Takeaway -- Pipeline**\n"
    ">\n"
    "> The full recipe is: `BitsAndBytesConfig` -> `AutoModelForCausalLM` ->\n"
    "> `prepare_model_for_kbit_training` -> `LoraConfig` -> `get_peft_model` ->\n"
    "> `SFTTrainer`. Six steps."
))

# ── 28. Section 7 Header ─────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 7. Advanced Techniques\n"
    "\n"
    "Push quality even further with these proven tricks."
))

# ── 29. Gradient checkpointing + Flash Attention ─────────────────────────
cells.append(code(
    "# ── Gradient Checkpointing ─────────────────────────────────────────\n"
    "# Already enabled via TrainingArguments above.  Trades ~30% speed\n"
    "# for ~60% less activation memory.\n"
    "\n"
    "# ── Flash Attention 2 ──────────────────────────────────────────────\n"
    "# Supported on Ampere+ GPUs (A100, H100, RTX 30xx/40xx).\n"
    "# Just add attn_implementation when loading:\n"
    "#\n"
    "# model = AutoModelForCausalLM.from_pretrained(\n"
    "#     MODEL_ID,\n"
    "#     quantization_config=bnb_config,\n"
    "#     device_map=\"auto\",\n"
    "#     attn_implementation=\"flash_attention_2\",\n"
    "# )\n"
    "\n"
    "print(\"Flash Attention 2: add attn_implementation='flash_attention_2'\")"
))

# ── 30. NEFTune + LR scheduling ──────────────────────────────────────────
cells.append(code(
    "# ── NEFTune (Noisy Embeddings) ─────────────────────────────────────\n"
    "# Adds uniform noise to embedding vectors during training.\n"
    "# Shown to improve chat-style fine-tuning by ~2-5 pts on MT-Bench.\n"
    "#\n"
    "# trainer = SFTTrainer(\n"
    "#     ...,\n"
    "#     neftune_noise_alpha=5,   # recommended: 5-15\n"
    "# )\n"
    "\n"
    "# ── Learning Rate Schedule Comparison ─────────────────────────────\n"
    "import math\n"
    "\n"
    "steps = 500\n"
    "warmup = int(steps * 0.03)\n"
    "lr_max = 2e-4\n"
    "\n"
    "def cosine_lr(step):\n"
    "    if step < warmup:\n"
    "        return lr_max * step / warmup\n"
    "    progress = (step - warmup) / (steps - warmup)\n"
    "    return lr_max * 0.5 * (1 + math.cos(math.pi * progress))\n"
    "\n"
    "def linear_lr(step):\n"
    "    if step < warmup:\n"
    "        return lr_max * step / warmup\n"
    "    return lr_max * (1 - (step - warmup) / (steps - warmup))\n"
    "\n"
    "# Print a few sample values\n"
    "print(f\"{'Step':>6s}  {'Cosine':>10s}  {'Linear':>10s}\")\n"
    "for s in [0, 15, 50, 100, 250, 400, 499]:\n"
    "    print(f\"{s:>6d}  {cosine_lr(s):>10.6f}  {linear_lr(s):>10.6f}\")"
))

# ── 31. Takeaway ──────────────────────────────────────────────────────────
cells.append(md(
    "> **Key Takeaway -- Advanced**\n"
    ">\n"
    "> Enable **gradient checkpointing** (free VRAM win), try **Flash Attention 2**\n"
    "> on Ampere GPUs, and experiment with **NEFTune** for chat models."
))

# ── 32. Section 8 Header ─────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 8. Evaluation & Inference\n"
    "\n"
    "After training we need to:\n"
    "1. **Merge** LoRA weights back into the base model\n"
    "2. **Generate** text to sanity-check\n"
    "3. **Benchmark** perplexity on a held-out set"
))

# ── 33. Merge LoRA weights ────────────────────────────────────────────────
cells.append(code(
    "# ── Merge LoRA into base model ─────────────────────────────────────\n"
    "# After training, you can merge for faster inference:\n"
    "#\n"
    "# from peft import PeftModel\n"
    "#\n"
    "# base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)\n"
    "# merged = PeftModel.from_pretrained(base, \"./lora-final\")\n"
    "# merged = merged.merge_and_unload()\n"
    "# merged.save_pretrained(\"./merged-model\")\n"
    "# tokenizer.save_pretrained(\"./merged-model\")\n"
    "\n"
    "print(\"Merge recipe: PeftModel.from_pretrained -> merge_and_unload -> save\")"
))

# ── 34. Generation pipeline ──────────────────────────────────────────────
cells.append(code(
    "# ── Quick generation test ─────────────────────────────────────────\n"
    "from transformers import pipeline\n"
    "\n"
    "# For demo we use the (un-trained) model already in memory\n"
    "pipe = pipeline(\n"
    "    \"text-generation\",\n"
    "    model=model,\n"
    "    tokenizer=tokenizer,\n"
    "    max_new_tokens=128,\n"
    "    do_sample=True,\n"
    "    temperature=0.7,\n"
    "    top_p=0.9,\n"
    ")\n"
    "\n"
    "prompt = \"### Instruction:\\nExplain LoRA in one sentence.\\n\\n### Response:\\n\"\n"
    "result = pipe(prompt)\n"
    "print(result[0][\"generated_text\"])"
))

# ── 35. Perplexity benchmark ─────────────────────────────────────────────
cells.append(code(
    "# ── Perplexity on a held-out split ───────────────────────────────\n"
    "import math\n"
    "\n"
    "def compute_perplexity(model, tokenizer, texts, max_length=512):\n"
    "    \"\"\"Compute perplexity over a list of texts.\"\"\"\n"
    "    model.eval()\n"
    "    total_loss = 0.0\n"
    "    total_tokens = 0\n"
    "\n"
    "    with torch.no_grad():\n"
    "        for text in texts:\n"
    "            enc = tokenizer(text, return_tensors=\"pt\",\n"
    "                            truncation=True, max_length=max_length).to(model.device)\n"
    "            outputs = model(**enc, labels=enc[\"input_ids\"])\n"
    "            total_loss += outputs.loss.item() * enc[\"input_ids\"].size(1)\n"
    "            total_tokens += enc[\"input_ids\"].size(1)\n"
    "\n"
    "    avg_loss = total_loss / total_tokens\n"
    "    return math.exp(avg_loss)\n"
    "\n"
    "# Usage (uncomment when model is trained):\n"
    "# eval_texts = load_dataset(\"yahma/alpaca-cleaned\", split=\"train[2000:2100]\")[\"output\"]\n"
    "# ppl = compute_perplexity(model, tokenizer, eval_texts)\n"
    "# print(f\"Perplexity: {ppl:.2f}\")\n"
    "\n"
    "print(\"Perplexity benchmark ready -- uncomment after training.\")"
))

# ── 36. Takeaway ──────────────────────────────────────────────────────────
cells.append(md(
    "> **Key Takeaway -- Evaluation**\n"
    ">\n"
    "> Always merge before deployment (`merge_and_unload`) and measure\n"
    "> **perplexity** as a quick sanity metric alongside task-specific evals."
))

# ── 37. Section 9 Header ─────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 9. Deployment\n"
    "\n"
    "Once trained and merged, you have several deployment paths."
))

# ── 38. GGUF export ──────────────────────────────────────────────────────
cells.append(code(
    "# ── GGUF Export (for llama.cpp / Ollama) ──────────────────────────\n"
    "# 1. Install llama.cpp:  git clone https://github.com/ggerganov/llama.cpp\n"
    "# 2. Convert:\n"
    "#    python llama.cpp/convert_hf_to_gguf.py ./merged-model \\\\\n"
    "#        --outfile model.gguf --outtype q4_k_m\n"
    "# 3. Quantize further (optional):\n"
    "#    ./llama.cpp/build/bin/llama-quantize model.gguf model-q4.gguf Q4_K_M\n"
    "\n"
    "print(\"GGUF export: convert_hf_to_gguf.py -> quantize -> serve with Ollama\")"
))

# ── 39. vLLM serving ─────────────────────────────────────────────────────
cells.append(code(
    "# ── vLLM Serving ─────────────────────────────────────────────────\n"
    "# vLLM gives you OpenAI-compatible API with continuous batching.\n"
    "#\n"
    "# pip install vllm\n"
    "# python -m vllm.entrypoints.openai.api_server \\\\\n"
    "#     --model ./merged-model \\\\\n"
    "#     --dtype auto \\\\\n"
    "#     --max-model-len 4096 \\\\\n"
    "#     --gpu-memory-utilization 0.90 \\\\\n"
    "#     --port 8000\n"
    "#\n"
    "# Then query:\n"
    "# curl http://localhost:8000/v1/completions \\\\\n"
    "#   -H 'Content-Type: application/json' \\\\\n"
    "#   -d '{\"model\": \"./merged-model\", \"prompt\": \"Hello!\", \"max_tokens\": 64}'\n"
    "\n"
    "print(\"vLLM: OpenAI-compatible API with PagedAttention & continuous batching\")"
))

# ── 40. Section 10 Header ────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 10. Key Takeaways & Further Reading\n"
    "\n"
    "### Summary\n"
    "\n"
    "| What | Why |\n"
    "|------|-----|\n"
    "| **LoRA** | Train <2 % of params, keep >97 % quality |\n"
    "| **QLoRA** | Add 4-bit quantization for 3x more VRAM savings |\n"
    "| **Rank 16** | Sweet spot for 7 B models |\n"
    "| **Cosine LR + warmup** | Stable convergence |\n"
    "| **Gradient checkpointing** | Free VRAM, slight speed cost |\n"
    "| **NEFTune** | +2-5 pts on chat benchmarks |\n"
    "| **Merge + GGUF** | Ship anywhere |"
))

# ── 41. Competition references ────────────────────────────────────────────
cells.append(md(
    "### Kaggle Competition References\n"
    "\n"
    "These competitions benefit directly from LoRA / QLoRA fine-tuning:\n"
    "\n"
    "- **Med-Gemma** -- Medical question answering with Google's Med-Gemma.\n"
    "  Fine-tune with domain-specific medical QA pairs using QLoRA on a T4.\n"
    "- **Akkadian Translation** -- Translate cuneiform tablets. Low-resource\n"
    "  language tasks are *ideal* for LoRA because the base model already\n"
    "  understands language structure; you just teach it a new mapping.\n"
    "- **AIMO 3 (AI Math Olympiad)** -- Mathematical reasoning. Fine-tune on\n"
    "  chain-of-thought math traces to boost step-by-step problem solving."
))

# ── 42. Further Reading ──────────────────────────────────────────────────
cells.append(md(
    "### Further Reading\n"
    "\n"
    "- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)\n"
    "- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)\n"
    "- [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914) (Jain et al., 2023)\n"
    "- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)\n"
    "- [TRL -- Transformer Reinforcement Learning](https://huggingface.co/docs/trl)\n"
    "- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)\n"
    "- [vLLM](https://docs.vllm.ai/)"
))

# ── 43. CTA ───────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "\n"
    "**Ready to fine-tune?** Fork this notebook, pick a competition model,\n"
    "swap in `MODEL_ID`, point to your dataset, and hit **Run All**.\n"
    "\n"
    "If this notebook helped you, please **upvote** and leave a comment!\n"
    "\n"
    "Happy fine-tuning!"
))

# ---------------------------------------------------------------------------
# Assemble notebook & write
# ---------------------------------------------------------------------------

# Convert each cell's source from a list of strings to the nbformat convention:
# each line (except the last) should end with "\n"
for cell in cells:
    lines = cell["source"]
    if lines:
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]


write_notebook(cells, __file__, "llm_finetuning_cookbook.ipynb")
