import json
import torch
import numpy as np
from bert import Bert
from utils import convert_examples_to_features
from transformers import BertTokenizer
from run_ernie import ErnieConfig
from run_large_roberta_pair import RobertaPairConfig
from run_large_roberta_wwm_ext import RobertaLargeConfig

model_name = 'ernie'
model_path = '../my_model/best_ernie.pkl'

config = None
if model_name == 'ernie':
    config = ErnieConfig()
elif model_name == 'pair':
    config = RobertaPairConfig()
elif model_name == 'wwm':
    config = RobertaLargeConfig()


model= Bert(config).cuda()
model.load_state_dict(torch.load(model_path))
model.eval()
tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file)

# 写json文件，本示例代码从测试集KUAKE-QQR_test.json读取数据数据，将预测后的数据写入到KUAKE-QQR_test_pred.json：
with open('../data/KUAKE/KUAKE-QQR_test.json', 'r', encoding='UTF-8') as input_data, \
        open('../prediction_result/{}_pred.json'.format(model_name), 'w', encoding='UTF-8') as output_data:
    json_content = json.load(input_data)
    # 逐条读取记录，并将预测好的label赋值
    for block in json_content:
        query1 = block['query1']
        query2 = block['query2']
        feature = convert_examples_to_features(
            examples=[[block['query1'],block['query2'], 0]],
            tokenizer=tokenizer,
            max_length=config.pad_size,
            data_type='test'
        )
        feature=feature[0]
        input_ids = torch.tensor(np.array(feature.input_ids)).unsqueeze(0).cuda()
        attention_mask = torch.tensor(np.array(feature.attention_mask)).unsqueeze(0).cuda()
        token_type_ids = torch.tensor(np.array(feature.token_type_ids)).unsqueeze(0).cuda()
        output, loss = model(input_ids, attention_mask, token_type_ids,labels=None)
        # 此处调用自己的模型来预测当前记录的label，仅做示例用：
        output = torch.max(output.data, 1)[1].cpu().numpy()
        block['label'] = str(*output)
        # 写json文件
    json.dump(json_content, output_data, indent=2, ensure_ascii=False)