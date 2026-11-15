# Large Language Models with At Most One Spike per Neuron

Paper link: https://arxiv.org/abs/2609.05151

## Overview

This repository consolidates the original research code used for the paper "Large Language Models with At Most One Spike per Neuron". It preserves two complementary experimental codebases built around a time-to-first-spike (TTFS) formulation for transformer-style language models:

- `GPT2/` contains the GPT-2 language-modeling experiments used for TTFS training, evaluation, and perplexity analysis.
- `BERT/` contains the BERT/GLUE benchmarks used to evaluate downstream language understanding under the same TTFS design.

The key idea is to apply the sparse, event-driven TTFS coding principle to large language models while keeping the overall transformer structure familiar. In standard TTFS coding, each neuron emits at most one spike within a temporal window; the paper extends this idea to reference-based representations for the main LLM components, including embeddings, normalization, attention-related transforms, and dropout. This repository preserves the original implementation paths rather than redesigning the experiments.

## Repository structure

```text
LLM-One-Spike-per-Neuron/
├── README.md
├── .gitignore
├── requirements.txt
├── GPT2/
│   ├── LICENSE
│   ├── train.py
│   ├── model.py
│   ├── sample.py
│   ├── perplexity.py
│   ├── config/
│   ├── base_TTFS.py
│   ├── basic_layer_TTFS.py
│   └── ...
├── BERT/
│   ├── dev_TTFS.py
│   ├── dev_vanilla.py
│   ├── run_glue.py
│   ├── model/
│   ├── Tasks/
│   ├── utils/
│   └── ...
└── ...
```

### `GPT2/`

This directory contains the GPT-2 training and evaluation code from the original `TTFS_GPT2` project. It includes the model definition, TTFS layer logic, and language-model training or perplexity scripts used in the paper experiments.

Key entry points:

- `GPT2/train.py`
- `GPT2/train_fineweb.py`
- `GPT2/perplexity.py`
- `GPT2/sample.py`

### `BERT/`

This directory contains the BERT/GLUE code derived from the `test_on_GLUE` project, treated here as the downstream evaluation path for the paper. The implementation is not a generic word-level benchmark; it is the GLUE-style classification and comparison pipeline used to test TTFS and vanilla BERT behavior.

Key entry points:

- `BERT/dev_TTFS.py`
- `BERT/dev_vanilla.py`
- `BERT/run_glue.py`
- `BERT/model/DownstreamTasks/`
- `BERT/Tasks/`

## Installation

The repository is intended to run in a Python environment with the source tree checked out locally.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Dependencies used by the retained code paths include:

- Python 3.10+
- PyTorch
- Hugging Face `transformers`
- Hugging Face `datasets`
- `evaluate`
- `tiktoken`
- `wandb`
- `tqdm`

Hardware note: the GPT experiments are designed for CUDA-capable GPUs, while the BERT/GLUE scripts are usually run in a GPU-enabled environment but can also be inspected and adapted for smaller local runs.

## Data preparation

This repository does not include the large datasets or checkpoint bundles used by the original experiments.

### GPT2 experiments

The GPT side expects a local corpus directory such as:

- `data/openwebtext`
- or another text corpus prepared in the binary format expected by `GPT2/train.py`

The dataset itself must be downloaded from its official source and placed under the expected local path before training or perplexity evaluation.

### BERT / GLUE experiments

The BERT code loads GLUE tasks through Hugging Face `datasets`, including tasks such as SST-2, MRPC, QNLI, QQP, RTE, MNLI, STS-B, and CoLA. No GLUE data is bundled in this repository. The scripts fetch the datasets from the standard Hugging Face dataset loader during execution.

The repository also keeps a local `BERT/bert-base-uncased/` directory and related tokenizer configuration files as part of the original experiment layout. These are retained for code fidelity, but they are not a substitute for a full benchmark or pretrained checkpoint cache.

## Usage

The code is organized around the original project separation and should be run from the corresponding directory unless you are adapting paths manually.

### GPT2 usage

```bash
cd GPT2
python train.py --help
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

Larger-scale experiments can use the original GPT-2 and fineweb training scripts as provided:

```bash
cd GPT2
python train.py config/train_gpt2.py
python perplexity.py
```

The exact local data paths and checkpoint layout may need to be adjusted to match the host environment because the original scripts include author-specific absolute paths.

### BERT / GLUE usage

```bash
cd BERT
python dev_vanilla.py
python dev_TTFS.py
```

The repository also includes task-specific entry points and a generic runner:

```bash
cd BERT
python run_glue.py
```

As with the GPT2 code, the original scripts may assume local paths for pretrained assets or cached experiment artifacts. These references are preserved to match the source repositories and may require small path adjustments in a fresh environment.

## Reproducing experiments

This repository preserves the original experimental structure rather than forcing a new standardized benchmark harness. In practice:

- `GPT2/` contains the GPT-2 language-modeling experiments used to study TTFS behavior.
- `BERT/` contains the BERT/GLUE evaluation experiments used to study downstream task performance.

The main implementation components are:

- GPT2 training and evaluation loops in `GPT2/train.py` and `GPT2/perplexity.py`
- TTFS primitives in `GPT2/base_TTFS.py` and `GPT2/basic_layer_TTFS.py`
- BERT model wrappers in `BERT/model/`
- GLUE evaluation logic in `BERT/dev_TTFS.py`, `BERT/dev_vanilla.py`, and `BERT/run_glue.py`

This should be treated as a faithful source snapshot of the authors' implementation, not as a newly standardized reproducibility package. Exact reproduction depends on the original data, pretrained caches, and environment-specific configuration.

## Results

The paper reports the TTFS approach as a sparse, event-driven alternative to conventional ANN-style language modeling, with the implementation here preserving the associated model and evaluation scripts. This repository does not ship a single turnkey benchmark output package or a complete checkpoint bundle, so performance claims should be read from the paper and the original experiment logs rather than inferred from a generic repository run.

## Citation

```bibtex
@article{zhao2026one,
  title={Large Language Models with At Most One Spike per Neuron},
  author={Zhao, Zhuoya and Omidi, Parsa and Jafari, Aref and Naud, Richard},
  journal={arXiv preprint arXiv:2609.05151},
  year={2026}
}
```

## Acknowledgements

This project integrates the original TTFS research codebases for the language-model and GLUE experiments. It relies on the PyTorch and Hugging Face ecosystem, and it preserves the original code attribution from the source repositories.

## License

This repository does not currently declare a single repository-wide license. The `GPT2/` tree includes the original project license file in `GPT2/LICENSE`. The BERT implementation is retained from the original research code without a separate merged license file in this cleaned repository.
