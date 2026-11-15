import sys
import os
import logging

from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    # AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.utils.versions import require_version
from model import GPTConfig, GPT

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="openai-community/gpt2",#None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )   #TODO gpt2-xl
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
BATCH = 2
@dataclass
class MyTrainingArgs(TrainingArguments):
    # resume_from_checkpoint : str = "/tmp/gpt-xl/checkpoint-1740/"
    fsdp : str =  "full_shard auto_wrap"  #TODO
    bf16 : bool = True
    bf16_full_eval : bool =True
    max_steps: int = 55000  #max training steps
    per_device_train_batch_size: int = field(
        default=BATCH, metadata={"help": "Batch size per device accelerator core/CPU for training."}
    )
    gradient_accumulation_steps: int = 128
    per_device_eval_batch_size: int = field(
        default=BATCH, metadata={"help": "Batch size per device accelerator core/CPU for evaluation."}
    )
    output_dir: Optional[str] = field(
        default="/tmp/gpt-xl",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written. Defaults to 'trainer_output' if not provided."
        },
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_train: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    save_steps: float = field(
        default=1000,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_safetensors: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use safetensors saving and loading for state dicts instead of default torch.load and torch.save."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,   #"wikitext",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="wikitext-2-raw-v1", #None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default="~/.cache/huggingface/datasets/parquet/default-00fdd91de0bcdd95/0.0.0/9c460aabd2aa27d1496e5e38d2060760561f0ac2cd6a110134eefa5b3f153b8d/",
                                      metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=True, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."




parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArgs))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "token": model_args.token,
    "trust_remote_code": model_args.trust_remote_code,
}
if model_args.config_name:
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
else:
    config = CONFIG_MAPPING[model_args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")
    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

gptconf = GPTConfig(**config.to_dict())
model = GPT(gptconf)
print(estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1))