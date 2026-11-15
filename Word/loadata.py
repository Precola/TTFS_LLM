import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
from datasets import get_dataset_config_names
import torch
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


from torch.nn.utils.rnn import pad_sequence
import torch

def generate_batch(data_batch, pad_idx=0):
    """
    自动动态 pad 到当前 batch 中最长的句子
    """

    b_input_ids = []
    b_token_type_ids = []
    # b_attention_mask = []
    b_mlm_labels = []
    b_nsp_labels = []
#    'input_ids', 'token_type_ids', 'attention_mask',
#     'next_sentence_labels', 'masked_lm_labels'
    for item in data_batch:
        input_ids = torch.tensor(item["input_ids"])
        token_type_ids = torch.tensor(item["token_type_ids"])
        # attention_mask =  torch.tensor(item["attention_mask"])
        mlm_labels = torch.tensor(item["masked_lm_labels"])
        nsp_label = item["next_sentence_labels"]

        b_input_ids.append(input_ids)
        b_token_type_ids.append(token_type_ids)
        # b_attention_mask.append(attention_mask)
        b_mlm_labels.append(mlm_labels)
        b_nsp_labels.append(nsp_label)

    # 动态 pad：根据当前 batch 中最长序列自动补齐
    b_input_ids = pad_sequence(b_input_ids, batch_first=False, padding_value=pad_idx)
    b_token_type_ids = pad_sequence(b_token_type_ids, batch_first=False, padding_value=pad_idx)
    b_mlm_labels = pad_sequence(b_mlm_labels, batch_first=False, padding_value=pad_idx)

    # 注意力 mask：padding 位置是 1
    b_mask = (b_input_ids == pad_idx).transpose(0, 1)  # shape: [batch_size, seq_len]

    # NSP 标签转 tensor
    b_nsp_labels = torch.tensor(b_nsp_labels, dtype=torch.long)  # shape: [batch_size]

    return b_input_ids, b_token_type_ids, b_mask, b_mlm_labels, b_nsp_labels

# dataset = load_dataset("lambada", split="test")
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# dataset = load_dataset("ptb_text_only", split="test")
wiki_dataset = load_dataset("wikipedia", "20220301.en")
bookdataset = load_dataset("bookcorpus", split="train")
dataset = concatenate_datasets([wiki_dataset['train'], bookdataset])
# dataset = dataset.select(range(100))
# configs = get_dataset_config_names("wikipedia")
# en_configs = sorted([c for c in configs if c.endswith(".en")])
# print(en_configs[-1])  # 打印最新英文版



import nltk
from nltk.tokenize import sent_tokenize

try:
    sent_tokenize("Test sentence.")  # 尝试使用以触发错误
except LookupError:
    nltk.download('punkt')  # 自动下载所需资源
    nltk.download('punkt_tab')

# # text = "This is the first sentence. This is the second sentence."
# #
# # # 直接使用
# # sentences = sent_tokenize(text)
# # print(sentences)
# # print('success!')
# #
# # save_path = "/home/zoe/processed_wiki_book_data"
# # path = os.path.join(save_path, "wikipedia_book.parquet")
# #
def preprocess_text(text):
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def preprocess_examples(examples):
    processed_texts = []
    for text in examples["text"]:
        processed_texts.extend(preprocess_text(text))
    return {"sentences": processed_texts}

# run TODO
processed_dataset = dataset.map(
    preprocess_examples,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=16  #
)
# # small for test
# # NSPprocessed_dataset = small_dataset.map(
# #     preprocess_examples,
# #     batched=True,
# #     remove_columns=small_dataset.column_names,
# #     num_proc=4  #
# # )
#
#
# # processed_dataset.save_to_disk(path)
# # from datasets import Dataset, load_from_disk
# # reloaded_dataset = Dataset.from_parquet(path)
# # print("exits:", os.path.exists(path))
# # print("or:", "ls" if os.path.isdir(path) else "file")
#
# # 2. loading
# # try:
# #     if os.path.isdir(path):
# #         dataset = load_from_disk(path)
# #     else:
# #         dataset = Dataset.from_parquet(path)
# #     print("successed!")
# # except Exception as e:
# #     print("failed!:", str(e))
# # print()


from transformers import BertTokenizer
import random

tokenizer = BertTokenizer.from_pretrained('/home/zoe/BertWithPretrained-main/bert-base-uncased')


def create_nsp_mlm_examples(examples):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    next_sentence_label = []
    masked_lm_labels = []

    sentences = examples["sentences"]

    for i in range(len(sentences) - 1):
        if random.random() > 0.5:
            text_a = sentences[i]
            text_b = sentences[i + 1]
            label = 1
        else:
            text_a = sentences[i]
            text_b = random.choice(sentences)
            while text_b == sentences[i + 1]:  # 防止随机到了真实下一句
                text_b = random.choice(sentences)
            label = 0

        encoded = tokenizer(
            text_a,
            text_b,
            max_length=512,
            truncation=True,
            padding=False,  #'max_length',TODO
            return_tensors='pt'
        )

        input_id_list = encoded['input_ids'][0].tolist()
        mlm_labels = [0] * len(input_id_list)   #TODO

        special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
        candidate_indices = [i for i, t_id in enumerate(input_id_list) if t_id not in special_token_ids]
        num_to_mask = max(1, int(len(candidate_indices) * 0.15))
        mask_indices = random.sample(candidate_indices, min(512, num_to_mask))  # TODO !!

        for idx in mask_indices:
            original_token = input_id_list[idx]
            mlm_labels[idx] = original_token

            prob = random.random()
            if prob < 0.8:
                input_id_list[idx] = tokenizer.mask_token_id
            elif 0.8 <= prob < 0.9:
                input_id_list[idx] = random.randint(0, tokenizer.vocab_size - 1)
            # 否则保留原 token，不变

        input_ids.append(input_id_list)
        token_type_ids.append(encoded['token_type_ids'][0].tolist())
        attention_mask.append(encoded['attention_mask'][0].tolist())
        next_sentence_label.append(label)
        masked_lm_labels.append(mlm_labels)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "next_sentence_labels": next_sentence_label,
        "masked_lm_labels": masked_lm_labels
    }



# 应用处理 # TODO
# final_dataset = dataset.map(
#     create_nsp_mlm_examples,
#     batched=True,
#     remove_columns=dataset.column_names,
#     num_proc=4
# )

final_dataset = processed_dataset.map(
    create_nsp_mlm_examples,
    batched=True,
    remove_columns=processed_dataset.column_names,
    num_proc=16
)

# 保存处理后的数据集
final_dataset.save_to_disk("data_processed_nsp_mlm4gb", max_shard_size="4GB")
# final_dataset.save_to_disk("data_processed_nsp_mlm4gb_small")

from torch.utils.data import DataLoader
# import torch
# import time
# s = time.time()
# final_dataset = load_from_disk("data_processed_nsp_mlm4gb_small")
# # final_dataset = load_from_disk("data_processed_shard4gb")
# # final_dataset.save_to_disk("data_processed_shard4gb", max_shard_size="4GB")
# e = time.time()
# print(e-s)
# print(final_dataset['input_ids'][0])
# 转换为torch格式
c

#
# from datasets import load_from_disk
#
# # 加载预处理数据
# dataset = load_from_disk("wikipedia_processed_nsp_mlm")

# 划分数据
split_dataset = final_dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

# 设置 PyTorch 格式
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'masked_lm_labels', 'next_sentence_labels']
train_dataset.set_format(type='torch', columns=columns)
val_dataset.set_format(type='torch', columns=columns)

# 创建 DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=lambda x: generate_batch(x, pad_idx=0))
val_loader = DataLoader(val_dataset,
                        batch_size=2,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=lambda x: generate_batch(x, pad_idx=0))

for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(val_loader):
    print(idx,'-------')
    b_token_ids = b_token_ids # [src_len, batch_size]
    print(b_token_ids.shape)
    b_segs = b_segs # [src_len, batch_size]
    print(b_segs.shape)
    b_mask = b_mask  ## [batch_size, src_len]
    print(b_mask.shape)
    b_mlm_label = b_mlm_label # [src_len, batch_size]
    print(b_mlm_label.shape)
    b_nsp_label = b_nsp_label  # [batch_size]
    print(b_nsp_label.shape)