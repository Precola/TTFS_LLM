import os
import logging
import sys

sys.path.append('../')
from utils import logger_init
from model import BertConfig
from model import BertForPretrainingModel
from utils import LoadBertPretrainingDataset
from transformers import BertTokenizer
# from transformers import AdamW
from torch.optim import AdamW
from datetime import datetime
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch import amp
from utils import LoadBertPretrainingDataset, tuple_collate_fn

from transformers import get_polynomial_decay_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import time

date_str = datetime.now().strftime("%Y-%m-%d")

class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # # ========== wike2 The configuration related to the dataset.
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'WikiText')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.train_file_path = os.path.join(self.dataset_dir, 'wiki.train.tokens')
        self.val_file_path = os.path.join(self.dataset_dir, 'wiki.valid.tokens')
        self.test_file_path = os.path.join(self.dataset_dir, 'wiki.test.tokens')
        self.data_name = 'wiki2'

        # # ========== songci The configuration related to the dataset.
        # self.dataset_dir = os.path.join(self.project_dir, 'data', 'SongCi')
        # self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        # self.train_file_path = os.path.join(self.dataset_dir, 'songci.train.txt')
        # self.val_file_path = os.path.join(self.dataset_dir, 'songci.valid.txt')
        # self.test_file_path = os.path.join(self.dataset_dir, 'songci.test.txt')
        # self.data_name = 'songci'

        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, f'LAbase_model_{self.data_name}_test.bin')
        self.writer = SummaryWriter(f"runs/LAbase_{self.data_name}_{date_str}_test")   # print result
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 1
        self.max_sen_len = None  # When set to None, padding will be applied to the samples in the batch based on the longest sample in that batch.
        self.pad_index = 0
        self.random_state = 2022
        self.learning_rate = 3e-5 #4e-5
        self.weight_decay = 0.1
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.log_level = 30 #logging.DEBUG
        self.use_torch_multi_head = False  # False means using the multi-head implementation from `model/BasicBert/MyTransformer`.
        self.epochs = 40
        self.model_val_per_epoch = 1
        self.model_val_per_iter = 30000
        self.gradient_accumulation_steps = 2# max 8
        self.gradient_accumulation_steps_max = 32# max 8 TODO
        self.i_gradient = 1
        self.i_gradient_max = 16 #TODO

        self.ddp = False # True #False TODO
        self.backend = 'nccl' # 'nccl', 'gloo', etc.
        self.use_amp = False #True
        self.resume = False #False  TODO
        self.grad_clip = 1.0
        logger_init(log_file_name=self.data_name, log_level=self.log_level,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config_large.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        logging.info(" ### Print the current configuration to the log file. ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    # model = BertForPretrainingModel(config,
    #                                 config.pretrained_model_dir)    #TODO
    model = BertForPretrainingModel(config,)

    if config.ddp:
        init_process_group(backend=config.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        config.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(config.device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert config.gradient_accumulation_steps % ddp_world_size == 0
        config.gradient_accumulation_steps //= ddp_world_size

        # training!
        if ddp_local_rank == 0:
            writer = SummaryWriter(f"runs/G_{config.data_name}_{date_str}_LA", comment='ddp-training')
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 10  # 113 #    #0
        ddp_world_size = 1
        writer = SummaryWriter(f"runs/G_{config.data_name}_{date_str}_LA___")
        config.gradient_accumulation_steps = 1
        # config.gradient_accumulation_steps = config.gradient_accumulation_steps * 2

    last_epoch = -1
    # if os.path.exists(config.model_save_path):
        # checkpoint = torch.load(config.model_save_path)
        # last_epoch = checkpoint['last_epoch']
        # loaded_paras = checkpoint['model_state_dict']
        # model.load_state_dict(loaded_paras)
        # logging.info("## Successfully loaded the existing model and continue training. ......")
    model = model.to(config.device)


    # Loading dataset
    # final_dataset = load_from_disk("/home/zoe/BertWithPretrained-main/data_processed_nsp_mlm4gb")
    final_dataset = load_from_disk("/home/zoe/TTFS-BERT/data_processed_nsp_mlm4gb_small")
    split_dataset = final_dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    columns = ['input_ids', 'token_type_ids', 'attention_mask', 'masked_lm_labels', 'next_sentence_labels']
    train_dataset.set_format(type='torch', columns=columns)
    val_dataset.set_format(type='torch', columns=columns)
    if config.ddp:
        sampler_train = DistributedSampler(train_dataset, num_replicas=2,rank=ddp_local_rank)
        sampler_val = DistributedSampler(val_dataset, num_replicas=2, rank=ddp_local_rank)

        train_iter = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                sampler=sampler_train,
                                collate_fn=lambda x: tuple_collate_fn(x, pad_idx=0))
        val_iter = DataLoader(val_dataset,
                              batch_size=config.batch_size,
                              sampler=sampler_val,
                              collate_fn = lambda x: tuple_collate_fn(x, pad_idx=0))
    else:
        train_iter = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                collate_fn=lambda x: tuple_collate_fn(x, pad_idx=0))
        val_iter = DataLoader(val_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              collate_fn = lambda x: tuple_collate_fn(x, pad_idx=0))

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            "initial_lr": config.learning_rate

        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "initial_lr": config.learning_rate
        },
    ]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters,
                        betas=(0.9, 0.999))
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          num_warmup_steps=10000,    #int(len(train_iter) * 0)
                                                          num_training_steps=int(config.epochs * len(train_iter)),
                                                          last_epoch=last_epoch)
    print(f"##################length of one epoch:{len(train_iter)}##################")
    if config.ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    model.train()
    max_acc = 0
    state_dict = None
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(train_iter):
            if config.ddp:
                model.require_backward_grad_sync = (idx % config.gradient_accumulation_steps == config.gradient_accumulation_steps - 1)
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            loss, mlm_logits, nsp_logits = model(input_ids=b_token_ids,
                                                 attention_mask=b_mask,
                                                 token_type_ids=b_segs,
                                                 masked_lm_labels=b_mlm_label,
                                                 next_sentence_labels=b_nsp_label)
            loss = loss / config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            loss.backward()
            if scheduler.last_epoch > 20000 * config.i_gradient:
                if config.i_gradient < config.i_gradient_max:
                    config.i_gradient += 1
                    config.gradient_accumulation_steps += 1
                    print(
                        f"[Step {scheduler.last_epoch}] ➜ Increased gradient_accumulation_steps to {config.gradient_accumulation_steps}!!!!!!!!!!!!!!!!!!!")

            if ((idx  % config.gradient_accumulation_steps) == (config.gradient_accumulation_steps - 1)):
                # with record_function("## optimizer ##"):
                # Step optimizer with scaled gradients
                if config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            losses += (loss.item() * config.gradient_accumulation_steps)
            mlm_acc, _, _, nsp_acc, _, _ = accuracy(mlm_logits, nsp_logits, b_mlm_label,
                                                    b_nsp_label, config.pad_index)
            if idx % 100 == 0 and (not config.ddp or ddp_local_rank == 0):
                # print(f"occupied memory: {torch.cuda.memory_allocated(0)}")
                # print(f"reserved memory: {torch.cuda.memory_reserved(0)}")
                logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f},"
                             f"nsp acc: {nsp_acc:.3f}")
                config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
                config.writer.add_scalars(main_tag='Training/Accuracy',
                                          tag_scalar_dict={'NSP': nsp_acc,
                                                           'MLM': mlm_acc},
                                          global_step=scheduler.last_epoch)
                # config.writer.add_histogram('23_LayerNorm_delta_t', (model.bert.bert_encoder.bert_layers[23].bert_attention.LayerNorm.x_s**2).detach().to(torch.float32).cpu().numpy(), scheduler.last_epoch)
                # config.writer.add_histogram('23_LayerNorm_delta_t', (model.bert.bert_encoder.bert_layers[23].bert_attention.LayerNorm.x_s**2).detach().to(torch.float32).cpu().numpy(), scheduler.last_epoch)

                # print("var", model.bert.bert_encoder.bert_layers[10].bert_attention.LayerNorm.x_s.min(),model.bert.bert_encoder.bert_layers[10].bert_attention.LayerNorm.x_s.max())
                # print('embedding_min:', (model.bert.bert_embeddings.LayerNorm.x_s**2).min())
                # print('embedding_max:', (model.bert.bert_embeddings.LayerNorm.x_s**2).max())
                # print("min", (model.bert.bert_encoder.bert_layers[0].bert_attention.LayerNorm.x_s**2).min())
                # print("max", (model.bert.bert_encoder.bert_layers[0].bert_attention.LayerNorm.x_s**2).max())
                # print("min", (model.bert.bert_encoder.bert_layers[23].bert_attention.LayerNorm.x_s**2).min())
                # print("max", (model.bert.bert_encoder.bert_layers[23].bert_attention.LayerNorm.x_s**2).max())
                print(f"Last_epoch:{scheduler.last_epoch},Iter:{idx},Epoch: [{epoch + 1}/{config.epochs}]")
                print('Training/Loss', loss.item()*config.gradient_accumulation_steps)

        if (idx % config.model_val_per_iter == 0) and (not config.ddp or ddp_local_rank == 0):
            # mlm_acc, nsp_acc = evaluate(config, train_iter, model, data_loader.PAD_IDX)
            checkpoint = {
                'last_epoch': scheduler.last_epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': vars(config),
            }
            print(f"saving checkpoint...........")
            torch.save(checkpoint, config.model_save_path)

        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if ((epoch + 1) % config.model_val_per_epoch == 0) and (not config.ddp or ddp_local_rank == 0):
            mlm_acc, nsp_acc = evaluate(config, val_iter, model, config.pad_index)
            logging.info(f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, "
                         f"NSP Accuracy on val: {round(nsp_acc, 4)}")
            writer.add_scalars(main_tag='Testing/Accuracy',
                               tag_scalar_dict={'NSP': nsp_acc,
                                                'MLM': mlm_acc},
                               global_step=scheduler.last_epoch)

    if config.ddp:
        destroy_process_group()

def accuracy(mlm_logits, nsp_logits, mlm_labels, nsp_label, PAD_IDX):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param nsp_logits:  [batch_size,2]
    :param nsp_label:  [batch_size]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
    mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
    # 将 [src_len,batch_size] 转成 [batch_size， src_len]
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况，得到预测正确的个数（此时还包括有mask位置）
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉mlm_acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total

    nsp_correct = (nsp_logits.argmax(1) == nsp_label).float().sum()
    nsp_total = len(nsp_label)
    nsp_acc = float(nsp_correct) / nsp_total
    return [mlm_acc, mlm_correct, mlm_total, nsp_acc, nsp_correct, nsp_total]


def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    mlm_corrects, mlm_totals, nsp_corrects, nsp_totals = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            mlm_logits, nsp_logits = model(input_ids=b_token_ids,
                                           attention_mask=b_mask,
                                           token_type_ids=b_segs)
            result = accuracy(mlm_logits, nsp_logits, b_mlm_label, b_nsp_label, PAD_IDX)
            _, mlm_cor, mlm_tot, _, nsp_cor, nsp_tot = result
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
            nsp_corrects += nsp_cor
            nsp_totals += nsp_tot
    model.train()
    return [float(mlm_corrects) / mlm_totals, float(nsp_corrects) / nsp_totals]


def inference(config, sentences=None, masked=False, language='en', random_state=None):
    """
    :param config:
    :param sentences:
    :param masked: Whether the sentence is masked during inference.
    :param language: Language.
    :param random_state: Control the random state when masking characters.
    :return:
    """
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
                                             tokenizer=bert_tokenize,
                                             pad_index=config.pad_index,
                                             random_state=config.random_state,
                                             masked_rate=0.15)  # 15% Mask掉
    token_ids, pred_idx, mask = data_loader.make_inference_samples(sentences,
                                                                   masked=masked,
                                                                   language=language,
                                                                   random_state=random_state)
    model = BertForPretrainingModel(config,
                                    config.pretrained_model_dir)
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型进行推理......")
    else:
        raise ValueError(f"模型 {config.model_save_path} 不存在！")
    model = model.to(config.device)
    model.eval()
    with torch.no_grad():
        token_ids = token_ids.to(config.device)  # [src_len, batch_size]
        mask = mask.to(config.device)
        mlm_logits, _ = model(input_ids=token_ids,
                              attention_mask=mask)
    pretty_print(token_ids, mlm_logits, pred_idx,
                 data_loader.vocab.itos, sentences, language)


def pretty_print(token_ids, logits, pred_idx, itos, sentences, language):
    """
    格式化输出结果
    :param token_ids:   [src_len, batch_size]
    :param logits:  [src_len, batch_size, vocab_size]
    :param pred_idx:   二维列表，每个内层列表记录了原始句子中被mask的位置
    :param itos:
    :param sentences: 原始句子
    :return:
    """
    token_ids = token_ids.transpose(0, 1)  # [batch_size,src_len]
    logits = logits.transpose(0, 1)  # [batch_size, src_len,vocab_size]
    y_pred = logits.argmax(axis=2)  # [batch_size, src_len]
    sep = " " if language == 'en' else ""
    for token_id, sentence, y, y_idx in zip(token_ids, sentences, y_pred, pred_idx):
        sen = [itos[id] for id in token_id]
        sen_mask = sep.join(sen).replace(" ##", "").replace("[PAD]", "").replace(" ,", ",")
        sen_mask = sen_mask.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
        logging.info(f" ### 原始: {sentence}")
        logging.info(f"  ## 掩盖: {sen_mask}")
        for idx in y_idx:
            sen[idx] = itos[y[idx]].replace("##", "")
        sen = sep.join(sen).replace("[PAD]", "").replace(" ,", ",")
        sen = sen.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
        logging.info(f"  ## 预测: {sen}")
        logging.info("===============")


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
    sentences_1 = ["I no longer love her, true, but perhaps I love her.",
                   "Love is so short and oblivion so long."]
    inference(config, sentences_1, masked=False, language='en',random_state=2022)

    # sentences_2 = ["十年生死两茫茫。不思量。自难忘。千里孤坟，无处话凄凉。",
    #                "红酥手。黄藤酒。满园春色宫墙柳。"]
    # inference(config, sentences_2, masked=False, language='zh',random_state=2022)
