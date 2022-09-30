import os
import logging
import pandas as pd
import torch
import time
import data_augment
from transformers import BertTokenizer
from bert import Bert
from train_eval import train
from utils import config_to_json_string, read_csv, get_dateset_labels

class ErnieConfig:

    def __init__(self):

        # 预训练模型路径
        self.pretrain_path = '../pretrain_models/ERNIE'
        _config_file = 'bert_config.json' 
        _model_file = 'pytorch_model.bin'
        _tokenizer_file = 'vocab.txt'
        self.config_file = os.path.join(self.pretrain_path, _config_file)
        self.model_name_or_path = os.path.join(self.pretrain_path, _model_file)
        self.tokenizer_file = os.path.join(self.pretrain_path, _tokenizer_file)
        # 数据路径
        self.train_path ='../data/KUAKE/KUAKE-QQR_train.json'
        self.dev_path = '../data/KUAKE/KUAKE-QQR_dev.json'
        self.test_path = '../data/KUAKE/KUAKE-QQR_test.json'
        self.aug_data_path = '../data/KUAKE/KUAKE-QQR_augment.csv'
        # 使用的模型
        self.use_model = 'bert'
        self.task = 'KUAKE'
        self.models_name = 'ernie'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.requires_grad = True
        self.class_list = ['0', '1', '2']
        self.num_labels = len(self.class_list)
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.multi_drop = 5
        self.hidden_size = 768
        self.early_stop = False
        self.require_improvement = 1000
        self.num_train_epochs = 10                  # epoch数
        self.batch_size = 128                      # mini-batch大小
        self.pad_size = 64                         # 每句话处理成的长度
        self.learning_rate = 1e-5                  # 学习率
        self.head_learning_rate = 1e-3             # 后面的分类层的学习率
        self.weight_decay = 0.01                   # 权重衰减因子
        self.warmup_proportion = 0.1               # Proportion of training to perform linear learning rate warmup for.
        # logging
        self.is_logging2file = True
        self.logging_dir = '../logging' + '/' + self.models_name
        # save
        self.save_path = '../my_model'
        self.save_file = self.models_name
        # 增强数据
        self.data_augment = False
        # 差分学习率
        self.diff_learning_rate = False
        # preprocessing
        self.stop_word_valid = True


if __name__ == '__main__':
    config = ErnieConfig()
    print(config.device)
    # labels = get_dateset_labels(config.train_path)
    # label0 = labels.count(0)    # 8946
    # label1 = labels.count(1)    # 2603
    # label2 = labels.count(2)    # 3451
    logging_filename = None
    if config.is_logging2file is True:
        file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        logging_filename = os.path.join(config.logging_dir, file)
        if not os.path.exists(config.logging_dir):
            os.makedirs(config.logging_dir)
    logging.basicConfig(filename=logging_filename, format='%(levelname)s: %(message)s', level=logging.INFO)
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file)
    train_examples = read_csv(config.train_path)
    if config.data_augment:
        augment_examples = data_augment.dataframe_to_list(pd.read_csv(config.aug_data_path))   # 请先运行augment_utils获得增广数据集
        train_examples.extend(augment_examples)
    dev_examples = read_csv(config.dev_path)
    model = Bert(config)
    # print(model)
    logging.info("self config %s", config_to_json_string(config))
    train(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_data=train_examples,
        dev_data=dev_examples,
    )




