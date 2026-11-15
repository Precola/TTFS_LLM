<div align="center">

# Large Language Models with At Most One Spike per Neuron

### TTFS-based spiking language modeling and downstream evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2609.05151-b31b1b.svg)](https://arxiv.org/abs/2609.05151)
[![Paper](https://img.shields.io/badge/Paper-Read%20PDF-0d9488.svg)](https://arxiv.org/abs/2609.05151)

**Zhuoya Zhao** · **Parsa Omidi** · **Aref Jafari** · **Richard Naud**

</div>

This repository contains the implementation used for the paper "Large Language Models with At Most One Spike per Neuron". It is organized around a GPT-2 pretraining pipeline and a BERT pretraining/evaluation pipeline, with the goal of integrating TTFS-based spiking computation into modern large-language-model architectures without abandoning the underlying transformer structure.

## TL;DR

The core idea is to bring spiking neural network (SNN) dynamics into modern language models through time-to-first-spike (TTFS) coding, while keeping the overall transformer structure recognizable. In this setting, each neuron emits at most one spike within a time window, which makes the representation extremely sparse and creates a direct path toward energy-efficient execution. The main engineering challenge is not simply to swap in a different activation pattern, but to preserve the behavior of the key LLM components — embeddings, layer normalization, attention-related operations, and dropout — under a TTFS formulation that remains compatible with the model structure.

The purpose of this work is  to build a TTFS-based LLM that retains competitive performance while exploiting the sparse, event-driven nature of SNNs. The technical contribution is therefore grounded in architecture-aware design: the model is made compatible with the transformer stack and the reference-based strategy is used to preserve the essential behavior of large-model components while maintaining the sparsity and energy-saving properties that motivate the approach.

## Repository structure

```text
TTFS_LLM/
├── README.md
├── .gitignore
├── requirements.txt
├── GPT2_pretrain/
│   ├── train.py
│   ├── model.py
│   ├── perplexity.py
│   ├── sample.py
│   ├── base_TTFS.py
│   ├── basic_layer_TTFS.py
│   ├── config/
│   └── ...
├── BERT_pretrain/
│   ├── dev_vanilla.py
│   ├── dev_TTFS.py
│   ├── run_glue.py
│   ├── model/
│   ├── Tasks/
│   ├── utils/
│   └── ...
└── ...
```

### GPT2_pretrain

This directory contains the GPT-2 pretraining and language-modeling code from the original TTFS research pipeline. It includes the model definition, TTFS layer primitives, training loop, perplexity evaluation, and sampling utilities.

Key entry points:

- `GPT2_pretrain/train.py`
- `GPT2_pretrain/perplexity.py`
- `GPT2_pretrain/sample.py`
- `GPT2_pretrain/base_TTFS.py`
- `GPT2_pretrain/basic_layer_TTFS.py`

### BERT_pretrain

This directory contains the BERT training and evaluation code used for the TTFS study. It includes the original pretraining pipeline, downstream task training, vanilla vs TTFS comparisons, and the task-specific modules needed for GLUE-style benchmark evaluation.

Key entry points:

- `BERT_pretrain/Tasks/TaskForPretraining_TTFS.py`
- `BERT_pretrain/dev_vanilla.py`
- `BERT_pretrain/dev_TTFS.py`
- `BERT_pretrain/run_glue.py`
- `BERT_pretrain/model/`
- `BERT_pretrain/Tasks/`

## Why the engineering matters here

This project is not just a toy TTFS experiment. It reflects a deliberate engineering effort to integrate spike-based computation into the LLM stack without losing the behavior and optimization structure that make large language models work in the first place:

- custom TTFS building blocks for transformer-like computations
- end-to-end training setup for GPT-2 style language modeling
- BERT-side pretraining and downstream task training/evaluation
- explicit comparison between vanilla and TTFS behavior under the same research pipeline

The implementation is designed to make the architectural choices explicit: the value of the work lies in the TTFS formulation itself, the reference-based strategy used to preserve model behavior, and the fact that the design is built around the actual needs of large-model training and evaluation rather than around a generic reproduction workflow.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

The retained code paths depend on:

- Python 3.10+
- PyTorch
- transformers
- datasets
- evaluate
- tiktoken
- wandb
- tqdm

GPU hardware is strongly recommended for GPT-2 pretraining and the larger BERT/GLUE runs. The code is designed for a research workstation or cluster environment rather than a minimal CPU-only toy setup.

## Data preparation

This repository does not include the large training corpora or full pretrained checkpoints used in the original experiments.

### GPT2_pretrain

The GPT-2 side expects a local corpus directory such as:

- `data/openwebtext`
- or another tokenized corpus in the binary format expected by `GPT2_pretrain/train.py`

The training loop reads tokenized binary data and runs the GPT-2 style pretraining setup from `GPT2_pretrain/train.py`.

### BERT_pretrain

The BERT side includes both pretraining-oriented TTFS code and downstream task training/evaluation. It uses Hugging Face `datasets` for tasks such as SST-2, MRPC, QNLI, QQP, RTE, MNLI, STS-B, and CoLA, and it includes the task-specific model wrappers and pretraining scripts needed to study the same TTFS design across both stages. No benchmark data is bundled here; it is fetched from the standard Hugging Face dataset sources when the scripts run.

## Usage

### GPT2_pretrain training

```bash
cd GPT2_pretrain
python train.py --help
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

For a larger-scale run, use the config files under `GPT2_pretrain/config/` and the original GPT-2 training pipeline in `GPT2_pretrain/train.py`.

### GPT2_pretrain perplexity and sampling

```bash
cd GPT2_pretrain
python perplexity.py
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

### BERT_pretrain training and evaluation

```bash
cd BERT_pretrain
python dev_vanilla.py
python dev_TTFS.py
python run_glue.py
```

The BERT code path includes both pretraining-oriented model setup and downstream task training/evaluation, so the same directory supports the TTFS experiments as well as the GLUE benchmark comparisons.

## Pretraining and evaluation flow

The code preserves the actual research flow rather than forcing a single wrapper around everything:

- `GPT2_pretrain` handles model initialization, tokenized corpus loading, optimizer setup, evaluation, checkpointing, and language-model training
- `GPT2_pretrain/base_TTFS.py` and `GPT2_pretrain/basic_layer_TTFS.py` implement the TTFS-specific computational blocks
- `BERT_pretrain` covers both the TTFS pretraining-oriented setup and the downstream benchmark pipeline, comparing vanilla and TTFS behavior under the same task setup

This is a research implementation, so some local paths and cache directories may still need small adjustments based on the host machine and the available dataset layout.

## Reproducibility notes

This repo is a faithful source snapshot of the original implementation. It is not a polished benchmark runner or a complete package of datasets and checkpoints. Exact reproduction still depends on the original preparation steps, the local data layout, and the environment used during the original experiments.

## Citation

```bibtex
@article{zhao2026one,
  title={Large Language Models with At Most One Spike per Neuron},
  author={Zhao, Zhuoya and Omidi, Parsa and Jafari, Aref and Naud, Richard},
  journal={arXiv preprint arXiv:2609.05151},
  year={2026}
}
```

## License

The repository keeps the original project licenses from the source trees. The GPT side includes `GPT2_pretrain/LICENSE`, and this repository does not currently impose a separate unified license on top of the original code.
