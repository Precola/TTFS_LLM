# Large Language Models with At Most One Spike per Neuron

Paper link: https://arxiv.org/abs/2609.05151

## Overview

This repository consolidates the research code corresponding to the paper "Large Language Models with At Most One Spike per Neuron". The implementation is organized around two complementary experimental codebases, both centered on a time-to-first-spike (TTFS) spiking formulation for transformer-style language models:

- `GPT/` contains the GPT-2 language-modeling code used to study TTFS-style large-language-model training and perplexity evaluation.
- `Word/` contains the BERT/GLUE evaluation code used to assess downstream language-understanding performance with the same TTFS design.

The core idea of the paper is to push the event-driven, sparse-computation properties of spiking neural networks into LLM components without abandoning the standard transformer-based architecture. In conventional TTFS coding, each neuron emits at most one spike within a temporal window, giving very low firing rates. The paper proposes a reference-based strategy to represent core LLM operations such as embeddings, layer normalization, attention-related transforms, and dropout in a TTFS-compatible form. The implementation in this repository preserves those original research code paths rather than redesigning the modeling stack.

This collection is therefore best viewed as the original experimental implementation of the paper: the model definitions, training loops, and evaluation scripts are retained as they were developed for the associated studies.

## Repository structure

```text
LLM-One-Spike-per-Neuron/
├── README.md
├── .gitignore
├── requirements.txt
├── GPT/
│   ├── train.py
│   ├── model.py
│   ├── sample.py
│   ├── perplexity.py
│   ├── config/
│   ├── base_TTFS.py
│   ├── basic_layer_TTFS.py
│   └── ...
├── Word/
│   ├── dev_TTFS.py
│   ├── dev_vanilla.py
│   ├── run_glue.py
│   ├── model/
│   ├── Tasks/
│   ├── utils/
│   └── ...
└── (original repository licenses remain in GPT/LICENSE and Word/LICENSE)
```

### `GPT/`

This directory corresponds to the GPT-2 training and evaluation code from the original `TTFS_GPT2` repository. It contains the experiment code for:

- GPT-style training loops and checkpoints
- TTFS layer primitives and custom TTFS normalization layers
- perplexity-based evaluation and sampling
- GPT-2 weight initialization and fine-tuning experiments

Key entry points include:

- `GPT/train.py`
- `GPT/train_fineweb.py`
- `GPT/perplexity.py`
- `GPT/sample.py`

### `Word/`

This directory corresponds to the BERT/GLUE evaluation code from the original `test_on_GLUE` repository, checked out from the `vanilla` branch. The implementation is not a generic word-level task name; rather, it is the GLUE benchmark suite for sentence-pair and single-sentence classification problems. In practice, the code includes:

- BERT-style sequence-classification experiments
- TTFS and vanilla GLUE evaluation pipelines
- per-task GLUE metric aggregation and model initialization logic
- task-specific data preparation for downstream classification

Key entry points include:

- `Word/dev_TTFS.py`
- `Word/dev_vanilla.py`
- `Word/run_glue.py`
- `Word/model/DownstreamTasks/`
- `Word/Tasks/`

## Installation

The following environment is the minimum practical setup for running the code in this repository.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

The implementation uses:

- Python 3.10+ is recommended.
- PyTorch with CUDA is strongly recommended for the GPT experiments.
- Hugging Face `transformers`, `datasets`, and `evaluate` are required for the BERT/GLUE benchmarking code.
- `tiktoken`, `wandb`, and `tqdm` are used by the GPT code paths.

Hardware notes:

- GPT training is designed for GPU usage and is most relevant on CUDA-capable hardware.
- The codebase contains examples for single-GPU and multi-GPU training, but full reproduction requires a machine with enough memory and data availability.
- The Word/GLUE scripts are also intended for GPU-enabled training; they can still be inspected and adapted for smaller local runs.

## Data preparation

The repository does not include the large benchmark datasets or model checkpoints used in the original experiments.

### GPT experiments

The GPT training code expects a local data directory such as:

- OpenWebText (`data/openwebtext`)
- or other custom text corpora prepared into the same binary format expected by `train.py`

The original `GPT/` code includes utilities to prepare tokenized text sequences, but the dataset itself must be downloaded from its official source and placed under the expected local path.

### Word / GLUE experiments

The BERT pipeline loads GLUE tasks through Hugging Face `datasets`, for example `glue` tasks such as SST-2, MRPC, QNLI, QQP, RTE, MNLI, STS-B, and CoLA. No GLUE data is stored in this repository. The scripts fetch it from the standard Hugging Face dataset loader during execution.

The repository also contains a local `Word/bert-base-uncased/` directory and references to tokenizer/config assets. These are retained as part of the original experiment layout, but they are not a replacement for a full benchmark or pretrained checkpoint cache.

## Usage

The repository keeps the original project separation and preserves the research code structure. Run scripts from their respective project directories unless you are adapting the paths manually.

### GPT usage

The training code is driven by the scripts in `GPT/` and follows the original `nanoGPT`-style entry points.

```bash
cd GPT
python train.py --help
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

For larger-scale experiments, the original scripts also support variants such as:

```bash
cd GPT
python train.py config/train_gpt2.py
python perplexity.py
```

The exact data path and local checkpoint layout may need to be adjusted to match your local environment, because the original research scripts include author-specific absolute paths.

### Word / GLUE usage

The BERT/GLUE experiment scripts are designed as research entry points for benchmarking TTFS and vanilla BERT implementations.

```bash
cd Word
python dev_vanilla.py
python dev_TTFS.py
```

The repository also includes task-specific entry points and a generic GLUE runner:

```bash
cd Word
python run_glue.py
```

As with the GPT code, some scripts in the original research setup assume local paths to pretrained assets or experiment caches. These references are preserved to match the source repositories and may require minor path adjustment in a fresh environment.

## Reproducing experiments

This repository preserves the original experimental structure rather than attempting a new benchmark harness. In practical terms:

- `GPT/` contains the GPT-2/training/perplexity experiments used to study TTFS language modeling.
- `Word/` contains the BERT/GLUE evaluation experiments used to study downstream task performance.

The original implementation uses:

- GPT training and evaluation loops in `GPT/train.py` and `GPT/perplexity.py`
- TTFS layer primitives in `GPT/base_TTFS.py` and `GPT/basic_layer_TTFS.py`
- BERT and task-specific model wrappers in `Word/model/`
- GLUE evaluation logic in `Word/dev_TTFS.py`, `Word/dev_vanilla.py`, and `Word/run_glue.py`

This combined repository should be treated as a faithful source snapshot for the paper’s implementation, not as a newly standardized benchmark runner. Exact reproducibility depends on the original data, pretrained caches, and local environment constraints.

## Results

The paper reports that the TTFS approach is competitive with ANN baselines on natural-language-understanding and common-sense reasoning tasks while retaining a clear gap on language-modeling perplexity. The aggregated code here preserves the original experimental scripts used to investigate those trade-offs, but the repository does not provide a single turnkey benchmark result package.

It is therefore best to treat the code as an implementation reference and consult the paper for authoritative results. Any numerical claims should be compared against the paper and the original experiment logs, rather than inferred from a generic repository-level run.

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

This project is an integration of the original research codebases for the TTFS language-model experiments. It relies on the PyTorch, Hugging Face Transformers, and Hugging Face Datasets ecosystem, and it retains the original project-specific licenses and code attribution from the source repositories.

## License

This repository combines code from two original project repositories. Their original licenses are retained in their respective directories:

- [GPT/LICENSE](GPT/LICENSE) — MIT
- [Word/LICENSE](Word/LICENSE) — Apache-2.0

No additional single combined license is inferred or imposed beyond the original source licenses already present in the code.
