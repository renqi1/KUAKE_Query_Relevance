# coding: UTF-8
import os
import torch
from sklearn import metrics
import time
from torch.utils.data import DataLoader
import copy
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.autograd import Variable
from utils import convert_examples_to_features, BuildDataSet
logger = logging.getLogger(__name__)

def train(
    config,
    model,
    tokenizer,
    train_data=None,
    dev_data=None,
):
    dev_acc = 0.
    # 加载模型
    model_example = copy.deepcopy(model).to(config.device)
    best_model = None

    if train_data:

        config.train_num_examples = len(train_data)
        # 特征转化
        train_features = convert_examples_to_features(
            examples=train_data,
            tokenizer=tokenizer,
            max_length=config.pad_size,
            data_type='train'
        )
        train_dataset = BuildDataSet(train_features)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        # dev 数据加载与转换
        if dev_data is not None:
            config.dev_num_examples = len(dev_data)
            dev_features = convert_examples_to_features(
                examples=dev_data,
                tokenizer=tokenizer,
                max_length=config.pad_size,
                data_type='dev'
            )
            dev_dataset = BuildDataSet(dev_features)
            dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
        else:
            dev_loader = None

        model_train(config, model_example, train_loader, dev_loader)


def model_train(config, model, train_iter, dev_iter=None):
    start_time = time.time()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    diff_part = ["bert.embeddings", "bert.encoder"]
    if config.diff_learning_rate is False:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
             },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    else:
        logger.info("use the diff learning rate")
        # the formal is basic_bert part, not include the pooler
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": config.weight_decay,
                "lr": config.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": config.learning_rate
             },
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": config.weight_decay,
                "lr": config.head_learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": config.head_learning_rate
             },
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

    t_total = len(train_iter) * config.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=t_total * config.warmup_proportion, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Train Num examples = %d", config.train_num_examples)
    logger.info("  Dev Num examples = %d", config.dev_num_examples)
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Train device:%s", config.device)

    model_name = config.models_name
    global_batch = 0  # 记录进行到多少batch
    dev_best_acc = 0
    train_dev_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    predict_all = []
    labels_all = []
    best_model = copy.deepcopy(model)

    for epoch in range(config.num_train_epochs):

        # scheduler.step() # 学习率衰减
        for epoch_batch, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_iter):
            global_batch += 1
            model.train()

            input_ids = Variable(input_ids).to(config.device)
            attention_mask = Variable(attention_mask).to(config.device)
            token_type_ids = Variable(token_type_ids).to(config.device)
            labels_tensor = Variable(labels).to(config.device)

            outputs, loss = model(input_ids, attention_mask, token_type_ids, labels_tensor)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            predic = torch.max(outputs.data, 1)[1].cpu()
            labels_all.extend(labels)
            predict_all.extend(predic)

            if (epoch_batch + 1) % 20 == 0:
                train_acc = metrics.accuracy_score(labels_all, predict_all)
                predict_all = []
                labels_all = []
                # dev 数据
                improve = ''
                if dev_iter is not None :
                    dev_acc, dev_loss= model_evaluate(config, model, dev_iter)
                    if dev_acc > dev_best_acc and train_acc > 0.85:
                        dev_best_acc = dev_acc
                        improve = '*'
                        model_save(config, model, name='best_'+model_name)

                    # elif train_acc>0.90 and (train_acc+dev_acc) > train_dev_acc:
                    #     train_dev_acc=train_acc+dev_acc
                    #     model_save(config, model, name='temp_'+model_name)
                    #     improve = improve+'!'
                    else:
                        improve = ''

                time_dif = time.time() - start_time
                msg = 'Iter: {0:>4}/{1:>4},  epoch: {2:>4}/{3:>4},  Train Loss: {4:>5.6f},  Train Acc: {5:>6.2%},  Val Loss: {6:>5.6f},  Val Acc: {7:>6.2%},  Time: {8} {9}'
                logger.info(msg.format(epoch_batch, len(train_iter), epoch+1, config.num_train_epochs, loss.cpu().data.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                print(msg.format(epoch_batch, len(train_iter), epoch+1, config.num_train_epochs, loss.cpu().data.item(), train_acc, dev_loss, dev_acc, time_dif, improve))


def model_evaluate(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_iter):

            input_ids = Variable(input_ids).to(config.device)
            attention_mask = Variable(attention_mask).to(config.device)
            token_type_ids = Variable(token_type_ids).to(config.device)
            labels_tensor = Variable(labels).to(config.device)

            outputs, loss = model(input_ids, attention_mask, token_type_ids, labels_tensor)
            predic = torch.max(outputs.data, 1)[1].cpu()
            predict_all.extend(predic)
            labels_all.extend(labels)
            loss_total += loss.item()
        dev_acc = metrics.accuracy_score(labels_all, predict_all)
    return dev_acc, loss_total / len(data_iter),



def model_save(config, model, name=None):
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    if name is not None:
        file_name = os.path.join(config.save_path, name + '.pkl')
    else:
        file_name = os.path.join(config.save_path, config.save_file+'.pkl')
    torch.save(model.state_dict(), file_name)
    logger.info("model saved, path: %s", file_name)


def model_load(config, model, device='cpu'):
    file_name = os.path.join(config.save_path, config.save_file+'.pkl')
    logger.info('loading model: %s', file_name)
    model.load_state_dict(torch.load(file_name,
                                     map_location=device if device == 'cpu' else "{}:{}".format(device, 0)))

