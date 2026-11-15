import sys
import argparse

sys.path.append('../')
from model import BertForSentenceClassification_TTFS as BertForSentenceClassification
from model import BertConfig
from utils import LoadSingleSentenceClassificationDataset
from utils import logger_init
from utils import preprocess_function, tuple_collate_fn_glue
from transformers import BertTokenizer
import logging
import torch
import os
import time
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from datasets import load_dataset
from datasets.features import ClassLabel, Value
from functools import partial
from transformers import get_polynomial_decay_schedule_with_warmup

import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=113, help="Random seed to set")
parser.add_argument("--task_name", type=str, default="rte", help="task name")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="5e-5, 3e-5, 2e-5")
args = parser.parse_args()
def set_seed():
    print("!!!!!Random seed: ", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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

f1_metric = evaluate.load("f1") #F1 scores
accuracy_metric = evaluate.load("accuracy")
spearmanr_metric = evaluate.load("spearmanr")
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

class ModelConfig:
    def __init__(self):
        self.task_name = args.task_name   #"mnli" #TODO

        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  #'/home/zoe/TTFS-BERT'
        # self.dataset_dir = os.path.join(self.project_dir, 'data', 'SingleSentenceClassification')
        self.pretrained_cache_dir = os.path.join(self.project_dir, "cache/24_Lr1e-4_model_2025-09-25_wiki2_TTFS_LA_update_!_386000.bin")
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert-base-uncased")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')          #'/home/zoe/TTFS-BERT/bert_base_uncased'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.train_file_path = os.path.join(self.dataset_dir, 'toutiao_train.txt')
        # self.val_file_path = os.path.join(self.dataset_dir, 'toutiao_val.txt')
        # self.test_file_path = os.path.join(self.dataset_dir, 'toutiao_test.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = '_!_'
        self.is_sample_shuffle = True
        self.batch_size = 16
        self.max_sen_len = 128  #None
        self.num_labels = None
        self.epochs = 3
        self.model_val_per_epoch = 1
        self.learning_rate = args.learning_rate#1e-4


        self.gradient_accumulation_steps = 2# max 8
        self.grad_clip = 1.0
        self.resume = False

        logger_init(log_file_name='single', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Also import the configuration parameters from the original BERT.
        bert_config_path = os.path.join(self.pretrained_model_dir, "config_large.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # Print the current configuration to the log file.
        logging.info(" ### Print the current configuration to the log file. ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


def train(config):
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_dir)

    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[config.task_name]
    wrapped_func = partial(preprocess_function, tokenizer=tokenizer, sentence1_key=sentence1_key, sentence2_key=sentence2_key)

    task = load_dataset('glue', config.task_name) #TODO

    print(f"Loading dataset GLUE:{config.task_name}.........")
    print(f"learning rate: {config.learning_rate}")
    if is_main_process():
        print("Only main process will run map()")
        dataset = task.map(wrapped_func, batched=True)

    # Then broadcast or reload the result for other processes
    dataset.set_format(type='torch', columns=[
        'input_ids', 'token_type_ids', 'attention_mask',
        'labels'
    ])
    # the number of label
    label_feature = dataset["train"].features["label"]
    if isinstance(label_feature, Value):
        print("label's dtype is Value  =", label_feature.dtype)
        num_labels = 1
    else:
        num_labels = label_feature.num_classes
    config.num_labels = num_labels

    train_iter = DataLoader(dataset['train'],
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=lambda x: tuple_collate_fn_glue(x, pad_idx=0, num_labels=num_labels))
    if config.task_name == 'mnli':
        val_iter_m = DataLoader(dataset['validation_matched'],
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=lambda x: tuple_collate_fn_glue(x, pad_idx=0, num_labels=num_labels))
        val_iter_mm = DataLoader(dataset['validation_mismatched'],
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=lambda x: tuple_collate_fn_glue(x, pad_idx=0, num_labels=num_labels))
    else:
        val_iter = DataLoader(dataset['validation'],
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=lambda x: tuple_collate_fn_glue(x, pad_idx=0, num_labels=num_labels))

    # intialize model
    model = BertForSentenceClassification(config,
                                          config.pretrained_cache_dir)

    # TTFS parameters initialization
    t_min, t_max = -1, 0  # for the input layer
    t_min, t_max = model.bert.bert_embeddings.set_params(t_min, t_max)
    for i, layer in enumerate(model.bert.bert_encoder.bert_layers):
        t_tuple = layer.set_params(t_min, t_max)
        t_min, t_max = t_tuple[-2:]
        # print(t_min, t_max, "----------")
    config.T_RES = t_max
    t_min_pooler, t_max_pooler = model.bert.bert_pooler.set_params(t_min, t_max)
    t_tuple = model.dropout.set_params(t_min_pooler, t_max_pooler)
    t_min, t_max = t_tuple[-2:]
    model.classifier.set_params(t_min, t_max)

    # Freezing all parameters in the BERT model
    for param in model.bert.parameters():
        param.requires_grad = False
    # initialize opt
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          num_warmup_steps=int(len(train_iter) * 0.1),    #int(len(train_iter) * 0)
                                                          num_training_steps=int(config.epochs * len(train_iter)))

    # load the model
    model_save_path = os.path.join(config.model_save_dir, f'{config.task_name}_model.pt')
    if os.path.exists(model_save_path) and config.resume:
        checkpoint = torch.load(model_save_path)
        loaded_paras = checkpoint['model']
        # new_state_dict = {}
        # for k, v in loaded_paras.items():
        #     new_key = k.replace('module.', '', 1)
        #     new_state_dict[new_key] = v
        # model.load_state_dict(new_state_dict)
        model.load_state_dict(loaded_paras)

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(config.device)

        scheduler.load_state_dict(checkpoint['scheduler'])

    model = model.to(config.device)



    model.train()

    max_eva = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_segs, b_mask, b_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_token_type_ids = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)  ## [batch_size, src_len]
            b_label = b_label.to(config.device)
            loss, logits = model(
                input_ids=b_token_ids,
                attention_mask=b_mask,
                token_type_ids=b_token_type_ids,
                position_ids=None,
                labels=b_label)

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if ((idx  % config.gradient_accumulation_steps) == (config.gradient_accumulation_steps - 1)):
                if config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            losses += (loss.item() * config.gradient_accumulation_steps)

            # evalute
            if config.task_name in ["qqp", "mrpc"]:
                f1_scores = f1_metric.compute(predictions=logits.argmax(1).tolist(), references=b_label.tolist()).get('f1')
                if idx % 50 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train f1_scores: {f1_scores:.3f}")
            elif config.task_name == "stsb":
                spearmanr_co = spearmanr_metric.compute(predictions=logits.squeeze().tolist(), references=b_label.tolist()).get("spearmanr", 0)
                if idx % 50 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train spearmanr_co: {spearmanr_co:.3f}")
            else:
                acc = (logits.argmax(1) == b_label).float().mean()
                if idx % 50 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")

        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            if config.task_name == "mnli":
                eva = evaluate(val_iter_m, model, config.device, config.task_name)
                logging.info(f"Accuracy on val_m {eva:.3f}")
                eva = evaluate(val_iter_mm, model, config.device, config.task_name)
                logging.info(f"Accuracy on val_mm {eva:.3f}")
            else:
                eva = evaluate(val_iter, model, config.device, config.task_name)
                logging.info(f"Accuracy on val {eva:.3f}")
            if eva > max_eva:
                max_eva = eva
                torch.save(model.state_dict(), model_save_path)
                checkpoint = {
                    'model': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': vars(config),
                }
                print(f"saving checkpoint...........")
                torch.save(checkpoint, model_save_path)


def inference(config):
    model = BertForSentenceClassification(config,
                                          config.pretrained_cache_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## Successfully loaded the existing model for prediction.......")
    model = model.to(config.device)
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=BertTokenizer.from_pretrained(
                                                              config.pretrained_model_dir).tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    acc = evaluate(test_iter, model, device=config.device, PAD_IDX=config.pad_index)
    logging.info(f"Acc on test:{acc:.3f}")


def evaluate(data_iter, model, device, task_name):
    model.eval()
    with ((torch.no_grad())):
        eva, n = 0.0, 0
        all_preds = []
        all_labels = []
        for idx, (b_token_ids, b_segs, b_mask, b_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(device)  # [src_len, batch_size]
            b_token_type_ids = b_segs.to(device)
            b_mask = b_mask.to(device)  ## [batch_size, src_len]
            b_label = b_label.to(device)
            loss, logits = model(
                input_ids=b_token_ids,
                attention_mask=b_mask,
                token_type_ids=b_token_type_ids,
                position_ids=None,
                labels=b_label)

            n += len(b_label)
            if task_name == 'stsb':
                all_preds.append(logits.squeeze().cpu())
            else:
                all_preds.append(logits.argmax(1).cpu())
            all_labels.append(b_label)

        all_preds = torch.cat(all_preds).tolist()
        all_labels = torch.cat(all_labels).tolist()

        if task_name in ["qqp", "mrpc"]:
            # f1_scores
            eva += f1_metric.compute(predictions=all_preds, references=all_labels).get("f1", 0)

        elif task_name == "stsb":
            # spearmanr_co
            eva += spearmanr_metric.compute(predictions=all_preds, references=all_labels).get("spearmanr", 0)
        else:
            # acc_sum
            eva += accuracy_metric.compute(predictions=all_preds, references=all_labels).get("accuracy", 0)
        model.train()
        return eva


if __name__ == '__main__':
    set_seed()
    model_config = ModelConfig()
    train(model_config)
