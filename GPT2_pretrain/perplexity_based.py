import math
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.utils.data import DataLoader
from itertools import chain
from tqdm import tqdm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai-community/gpt2"

    # model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

    model = GPT2LMHeadModel.from_pretrained("rhysjones/gpt2-124M-edu-fineweb-10B").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model.eval()

    block_size = model.config.n_positions  # 1024

    dataset = load_dataset(
        "/home/zoe/nanoGPT-old/data/wikitext/wikitext-2-raw-v1",
        split="test"
    )
    column_names = list(dataset.features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    lm_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    batch_size = 1
    dataloader = DataLoader(lm_dataset, batch_size=batch_size)

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)

    print(f"Model:      {model_id}")
    print(f"Dataset:    wikitext-2-raw-v1 (test)")
    print(f"Avg Loss:   {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
