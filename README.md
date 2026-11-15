<div align="center">

# Large Language Models with At Most One Spike per Neuron

### TTFS-based language model pretraining and downstream evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2609.05151-b31b1b.svg)](https://arxiv.org/abs/2609.05151)
[![Paper](https://img.shields.io/badge/Paper-Read%20PDF-0d9488.svg)](https://arxiv.org/abs/2609.05151)

**Zhuoya Zhao** · **Parsa Omidi** · **Aref Jafari** · **Richard Naud**

</div>

This repo contains the code used for the paper "Large Language Models with At Most One Spike per Neuron". It keeps the original research workflow in a clean local layout: a GPT-2-style pretraining path for TTFS language modeling, and a BERT/GLUE path for downstream evaluation.

## TL;DR

The project is built around a time-to-first-spike (TTFS) formulation for transformer-style language models. The GPT branch is the pretraining side, and the BERT branch is the evaluation side used to test the same idea on downstream language tasks.

## Project structure

- `GPT2_pretrain/`: GPT-2 model code, TTFS primitives, training scripts, and perplexity/sampling utilities
- `BERT_pretrain/`: BERT and GLUE experiments, vanilla vs TTFS comparisons, and task-specific model code
- `requirements.txt`: dependency list for the retained code paths

This is a source snapshot of the original implementation, not a refactor into a new benchmark framework.

## Why it is organized like this

The code matches the actual research flow:

1. Pretraining a GPT-2-style language model with TTFS layers and custom reference-based dynamics.
2. Evaluating the learned representation on BERT/GLUE-style downstream tasks.

The relevant entry points are:

- `GPT2_pretrain/train.py` and `GPT2_pretrain/perplexity.py`
- `GPT2_pretrain/base_TTFS.py` and `GPT2_pretrain/basic_layer_TTFS.py`
- `BERT_pretrain/dev_vanilla.py` and `BERT_pretrain/dev_TTFS.py`
- `BERT_pretrain/run_glue.py`

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Dependencies used by the code paths in this repo:

- Python 3.10+
- PyTorch
- transformers
- datasets
- evaluate
- tiktoken
- wandb
- tqdm

GPU hardware is strongly recommended for GPT pretraining. The BERT/GLUE path is also meant for GPU usage in practice.

## Data preparation

This repo does not include the large corpora or checkpoint bundles from the original experiments.

### GPT2 pretraining

The GPT side expects a local corpus directory such as:

- `data/openwebtext`
- or another corpus in the binary format expected by `GPT2_pretrain/train.py`

The training loop reads tokenized binary data and runs the GPT-2-style pretraining setup from `GPT2_pretrain/train.py`.

### BERT / GLUE

The BERT scripts use the Hugging Face `datasets` loader for GLUE tasks such as SST-2, MRPC, QNLI, QQP, RTE, MNLI, STS-B, and CoLA. No benchmark data is bundled in this repo; it is fetched from the standard Hugging Face dataset sources during execution.

## Quick start

### GPT2 pretraining

```bash
cd GPT2_pretrain
python train.py --help
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

For a larger run, use the config files in `GPT2_pretrain/config/` and the training loop in `GPT2_pretrain/train.py`.

### GPT2 perplexity / sampling

```bash
cd GPT2_pretrain
python perplexity.py
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

### BERT / GLUE evaluation

```bash
cd BERT_pretrain
python dev_vanilla.py
python dev_TTFS.py
python run_glue.py
```

## Pretraining and evaluation flow

This project keeps the original research workflow instead of trying to package everything into a single benchmark wrapper.

- GPT2 side: model init, tokenized corpus loading, optimizer setup, evaluation, and checkpointing
- TTFS side: custom reference-based operations in `base_TTFS.py` and `basic_layer_TTFS.py`
- downstream validation: BERT/GLUE comparison code that checks the TTFS design against standard task performance

This is a research implementation, so some local paths and cache names may still need small adjustments depending on the machine and dataset layout.

## Notes on reproducibility

This repo is a faithful source snapshot of the original implementation. It does not ship a complete dataset bundle or a full pretrained model archive, and it is not a polished standalone benchmark runner.

If you want to reproduce the experiments, you still need the original corpora, tokenizer assets, and environment-specific setup used by the source repositories.

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

The repo keeps the original project licenses from the source trees. The GPT side includes `GPT2_pretrain/LICENSE`, and the repository does not currently impose a separate unified license on top of the original code.
