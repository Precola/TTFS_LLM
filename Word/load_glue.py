import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.utils.data import DataLoader

def generate_batch(data_batch, pad_idx=0):
    """
    自动动态 pad 到当前 batch 中最长的句子
    """

    b_input_ids = []
    b_token_type_ids = []
    # b_attention_mask = []
    b_labels = []
#    'input_ids', 'token_type_ids', 'attention_mask',
#     'next_sentence_labels', 'masked_lm_labels'
    for item in data_batch: #TODO!!!!!!! dim
        input_ids = torch.tensor(item["input_ids"]).squeeze(0)
        token_type_ids = torch.tensor(item["token_type_ids"]).squeeze(0)
        # attention_mask =  torch.tensor(item["attention_mask"])
        labels = torch.tensor([item["labels"]])

        b_input_ids.append(input_ids)
        b_token_type_ids.append(token_type_ids)
        # b_attention_mask.append(attention_mask)
        b_labels.append(labels)

    # 动态 pad：根据当前 batch 中最长序列自动补齐
    b_input_ids = pad_sequence(b_input_ids, batch_first=False, padding_value=pad_idx)
    b_token_type_ids = pad_sequence(b_token_type_ids, batch_first=False, padding_value=pad_idx)
    b_labels = pad_sequence(b_labels, batch_first=False, padding_value=pad_idx)

    # 注意力 mask：padding 位置是 1
    b_mask = (b_input_ids == pad_idx).transpose(0, 1)  # shape: [batch_size, seq_len]

    b_labels = torch.tensor(b_labels, dtype=torch.long)  # shape: [batch_size]

    return b_input_ids, b_token_type_ids, b_mask, b_labels


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

mrpc = load_dataset('glue', 'mrpc')


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Preprocessing the raw_datasets
sentence1_key, sentence2_key = task_to_keys['mrpc']


tokenizer = BertTokenizer.from_pretrained('/home/zoe/BertWithPretrained-main/bert-base-uncased')

def preprocess_function(samples):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label = []

    samples = [dict(zip(samples, t)) for t in zip(*samples.values())]
    for sample in samples:
        # Tokenize the texts
        args = (
            (sample[sentence1_key],) if sentence2_key is None else (sample[sentence1_key], sample[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=512, truncation=True, return_tensors='pt')
        # print()
        input_ids.append(result['input_ids'])
        token_type_ids.append(result['token_type_ids'])
        attention_mask.append(result['attention_mask'])
        label.append(sample['label'])

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": label
    }


if is_main_process():
    print("Only main process will run map()")
    dataset = mrpc.map(preprocess_function, batched=True)

# Then broadcast or reload the result for other processes
dataset.set_format(type='torch', columns=[
    'input_ids', 'token_type_ids', 'attention_mask',
    'labels'
])


dataloader = DataLoader(
    dataset['train'],
    batch_size=3,
    shuffle=True,
    collate_fn=lambda x: generate_batch(x, pad_idx=0)
)


for idx, _ in enumerate(dataloader):
    print(idx)
    print(_)
    if idx >4 :
        break