import json
import torch
import numpy as np
from bert import Bert
from transformers import BertTokenizer
from torch.autograd import Variable
from utils import convert_examples_to_features, BuildDataSet, read_csv
from torch.utils.data import DataLoader
from run_ernie import ErnieConfig
from run_large_roberta_pair import RobertaPairConfig
from run_large_roberta_wwm_ext import RobertaLargeConfig

def np_softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def get_outputs(config, path):
    model = Bert(config).cuda()
    model.load_state_dict(torch.load(path))
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file)
    test_examples = read_csv(config.test_path)
    test_features = convert_examples_to_features(examples=test_examples, tokenizer=tokenizer,
                                                  max_length=config.pad_size, data_type='test')
    test_dataset = BuildDataSet(test_features)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
            input_ids = Variable(input_ids).cuda()
            attention_mask = Variable(attention_mask).cuda()
            token_type_ids = Variable(token_type_ids).cuda()
            output, _ = model(input_ids, attention_mask, token_type_ids)
            output = output.data.cpu().numpy()
            outputs.extend(output)
        outputs = np.array(outputs)
    return np_softmax(outputs)


path_pair = '../my_model/best_roberta_large_pair.pkl'
path_wwm = '../my_model/best_roberta_wwm_large.pkl'
path_ernie = '../my_model/best_ernie.pkl'

weight1, weight2, weight3 = 0.5, 0.35, 0.05   # 8672

outputs_ernie = get_outputs(ErnieConfig(), path_ernie)
outputs_pair = get_outputs(RobertaPairConfig(), path_pair)
outputs_wwm = get_outputs(RobertaLargeConfig(), path_wwm)

final_output = weight1*outputs_pair+weight2*outputs_wwm+weight3*outputs_ernie   # 设置不同模型的权重根据单个模型正确率
max = np.max(final_output, axis=1).reshape(-1, 1)
labels = np.where(final_output == max)[1]

# 写json文件，本示例代码从测试集KUAKE-QQR_test.json读取数据数据，将预测后的数据写入到KUAKE-QQR_test_pred.json：
with open('../data/KUAKE/KUAKE-QQR_test.json', 'r', encoding='UTF-8') as input_data, \
        open('../prediction_result/KUAKE-QQR_test_pred_mix3.json', 'w', encoding='UTF-8') as output_data:
    json_content = json.load(input_data)
    # 逐条读取记录，并将预测好的label赋值
    for i, block in enumerate(json_content):
        block['label'] = str(labels[i])
        # 写json文件
    json.dump(json_content, output_data, indent=2, ensure_ascii=False)
