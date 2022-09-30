import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import logging
logger = logging.getLogger(__name__)


class Bert(nn.Module):

    def __init__(self, config):
        super(Bert, self).__init__()
        model_config = BertConfig.from_pretrained(config.config_file, num_labels=config.num_labels, finetuning_task=config.task,)
        self.bert = BertModel.from_pretrained(config.model_name_or_path, config=model_config,)
        if config.requires_grad:
            for param in self.bert.parameters():
                param.requires_grad = True

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.multi_drop = config.multi_drop
        self.hidden_size = config.hidden_size
        self.classifier=nn.Sequential(nn.Linear(self.hidden_size,  config.num_labels),)



    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # 只取类别部分 [batchsize, hiddensize]
        out = None
        loss = 0
        for i in range(5):
            pooled_output = self.dropout(pooled_output)
            out = self.classifier(pooled_output)
            if labels is not None:
                if i == 0:
                    loss = F.cross_entropy(out, labels) / self.multi_drop
                else:
                    loss += loss / self.multi_drop

        # out = self.classifier(pooled_output)
        # if labels == None:
        #     loss = None
        # else:
        #     loss = F.cross_entropy(out, labels)

        return out, loss
