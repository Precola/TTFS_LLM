import os
import logging
import sys

sys.path.append('../')
import socket
from utils import logger_init
from model import BertConfig
from model import BertForPretrainingModel_TTFS
from utils import LoadBertPretrainingDataset, tuple_collate_fn
from transformers import BertTokenizer
# from transformers import AdamW
from torch.optim import AdamW
from torch.autograd.profiler import record_function
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader

from datetime import datetime

from transformers import get_polynomial_decay_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import time

date_str = datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   filename="log_mem_snap.txt",
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return

def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")




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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{self.data_name}_TTFS_emLN_test3.bin') #TODO
        self.writer = SummaryWriter(f"runs/{self.data_name}_{date_str}_emLN_test3")   # print result #TODO
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 16
        self.max_sen_len = None  # When set to None, padding will be applied to the samples in the batch based on the longest sample in that batch.
        self.pad_index = 0
        self.random_state = 2022
        self.learning_rate = 4e-5
        self.weight_decay = 0.1
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.log_level = 30 #logging.DEBUG
        self.use_torch_multi_head = False  # False means using the multi-head implementation from `model/BasicBert/MyTransformer`.
        self.epochs = 200
        self.model_val_per_epoch = 1

        logger_init(log_file_name=self.data_name, log_level=self.log_level,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        logging.info(" ### Print the current configuration to the log file. ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")

def get_activation_writer_hook(writer, layer_name, global_step):
    def hook_fn(module, input, output):
        writer.add_histogram(f"Activations/{layer_name}", output, global_step)
    return hook_fn

def train(config):
    tokenizer = BertTokenizer.from_pretrained('/home/zoe/BertWithPretrained-main/bert-base-uncased')
    config.vocab_size = tokenizer.vocab_size
    # model = BertForPretrainingModel(config,
    #                                 config.pretrained_model_dir)    #TODO

    model = BertForPretrainingModel_TTFS(config,)
    t_min, t_max = -1, 0  # for the input layer
    t_min, t_max = model.bert.bert_embeddings.set_params(t_min, t_max)
    for i, layer in enumerate(model.bert.bert_encoder.bert_layers):
        t_tuple = layer.set_params(t_min, t_max)
        t_min, t_max = t_tuple[-2:]
        # print(t_min, t_max, "----------")
    config.T_RES = t_max
    t_min_pooler, t_max_pooler = model.bert.bert_pooler.set_params(t_min, t_max)

    model.mlm_prediction.set_params(t_min, t_max)

    model.nsp_prediction.set_params(t_min_pooler, t_max_pooler)


    last_epoch = -1
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        last_epoch = checkpoint['last_epoch']
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## Successfully loaded the existing model and continue training. ......")
    model = model.to(config.device)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
                                             tokenizer=bert_tokenize,
                                             batch_size=config.batch_size,
                                             max_sen_len=config.max_sen_len,
                                             max_position_embeddings=config.max_position_embeddings,
                                             pad_index=config.pad_index,
                                             is_sample_shuffle=config.is_sample_shuffle,
                                             random_state=config.random_state,
                                             data_name=config.data_name,
                                             masked_rate=config.masked_rate,
                                             masked_token_rate=config.masked_token_rate,
                                             masked_token_unchanged_rate=config.masked_token_unchanged_rate)
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                             train_file_path=config.train_file_path,
                                             val_file_path=config.val_file_path)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    trainable_named_params = list(filter(lambda item: item[1].requires_grad, model.named_parameters())) #TODO

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
    ]     #   TODO
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
    #         "weight_decay": config.weight_decay,
    #         "initial_lr": config.learning_rate
    #
    #     },
    #     {
    #         "params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #         "initial_lr": config.learning_rate
    #     },
    # ]


    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          int(len(train_iter) * 0),
                                                          int(config.epochs * len(train_iter)),
                                                          last_epoch=last_epoch)


    max_acc = 0
    state_dict = None

    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        # Start recording memory snapshot history
        # start_record_memory_history()
        # with torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA,
        #         ],
        #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=True,
        #         on_trace_ready=trace_handler,
        # ) as prof:
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(train_iter):


            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)   # [src_len, batch_size]
            b_mask = b_mask.to(config.device)   ## [batch_size, src_len]
            b_mlm_label = b_mlm_label.to(config.device) # [src_len, batch_size]
            b_nsp_label = b_nsp_label.to(config.device) #[batch_size]
            # with record_function("## forward ##"):
            loss, mlm_logits, nsp_logits = model(input_ids=b_token_ids,
                                                 attention_mask=b_mask,
                                                 token_type_ids=b_segs,
                                                 masked_lm_labels=b_mlm_label,
                                                 next_sentence_labels=b_nsp_label)
            # print('+++++++++forward+++++++++++++')
            # print(f"occupied memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
            # print(f"reserved memory: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
            # print(f"max: {torch.cuda.max_memory_reserved(0) / 1024 ** 3:.2f} GB")
            optimizer.zero_grad()

            # with record_function("## backward ##"):
            loss.backward()
            # print('+++++++++backward+++++++++++++')
            # print(f"occupied memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
            # print(f"reserved memory: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
            # with record_function("## optimizer ##"):
            optimizer.step()

            scheduler.step()
            losses += loss.item()
            mlm_acc, _, _, nsp_acc, _, _ = accuracy(mlm_logits, nsp_logits, b_mlm_label,
                                                    b_nsp_label, config.pad_index)
            # prof.step()
            if idx % 20 == 0:
                # print(f"occupied memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
                # print(f"reserved memory: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
                # print(f"max: {torch.cuda.max_memory_reserved(0) / 1024 ** 3:.2f} GB")
                #
                logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f},"
                             f"nsp acc: {nsp_acc:.3f}")
                config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
                config.writer.add_scalars(main_tag='Training/Accuracy',
                                          tag_scalar_dict={'NSP': nsp_acc,
                                                           'MLM': mlm_acc},
                                          global_step=scheduler.last_epoch)

                # config.writer.add_histogram('LayerNorm_delta_t', model.bert.bert_embeddings.LayerNorm.model.delta_t, scheduler.last_epoch)
                # config.writer.add_histogram('LayerNorm/t_root_em', model.bert.bert_embeddings.LayerNorm.model.t_root_em, scheduler.last_epoch)
                # config.writer.add_histogram('LayerNorm/t_root', model.bert.bert_embeddings.LayerNorm.model.t_root, scheduler.last_epoch)
                # config.writer.add_histogram("ttfs_root1_weight.layers", model.bert.bert_encoder.bert_layers[0].LayerNorm.model.ttfs_root1.layers[0].linear.weight, scheduler.last_epoch)
                # config.writer.add_histogram("ttfs_root1.out", model.bert.bert_encoder.bert_layers[0].LayerNorm.model.ttfs_root1.out.linear.weight, scheduler.last_epoch)
                # config.writer.add_histogram("ttfs_root2.layers", model.bert.bert_encoder.bert_layers[0].LayerNorm.model.ttfs_root2.layers[0].linear.weight, scheduler.last_epoch)
                # config.writer.add_histogram("ttfs_root2.out", model.bert.bert_encoder.bert_layers[0].LayerNorm.model.ttfs_root2.out.linear.weight, scheduler.last_epoch)
                # config.writer.add_histogram("bert_pooler", model.bert.bert_pooler.pooled_output, scheduler.last_epoch)

                print(f"Epoch: [{epoch + 1}/{config.epochs}]")
                print('Training/Loss', loss.item())
        # Construct the memory timeline HTML plot.
        # print("save snapshot")
        # Create the memory snapshot file
        # export_memory_snapshot()

        # Stop recording memory snapshot history
        # stop_record_memory_history()

        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            mlm_acc, nsp_acc = evaluate(config, val_iter, model, config.pad_index)
            logging.info(f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, "
                         f"NSP Accuracy on val: {round(nsp_acc, 4)}")
            config.writer.add_scalars(main_tag='Testing/Accuracy',
                                      tag_scalar_dict={'NSP': nsp_acc,
                                                       'MLM': mlm_acc},
                                      global_step=scheduler.last_epoch)
            # mlm_acc, nsp_acc = evaluate(config, train_iter, model, data_loader.PAD_IDX)
            if mlm_acc > max_acc:
                max_acc = mlm_acc
                state_dict = deepcopy(model.state_dict())
            torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                       config.model_save_path)



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
    model = BertForPretrainingModel_TTFS(config,
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
