'采用余弦相似度和jaccard相似度，正确率只有70%，参考价值不大，没事传着玩'

import json
import numpy as np
import math

tokenizer = lambda x: [y for y in x]  # char-level

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

def count_word(word,token):
    count=0
    for i in token:
        if i == word:
            count +=1
    return count

def temp_vocab(query1, query2):
    token1=tokenizer(query1)
    token2=tokenizer(query2)
    token1set=set(token1)
    token2set=set(token2)
    token = token1set.union(token2set)
    vocab_dic1 = {}
    vocab_dic2 = {}
    for word in token:
        vocab_dic1[word] = count_word(word,token1)
    for word in token:
        vocab_dic2[word] = count_word(word,token2)
    return vocab_dic1,vocab_dic2

def get_jaccard(a,b):
    count = 0
    for i in a:
        if i in b:
            count+=1
    for j in b:
        if j in a:
            count+=1
    return count, count/(len(a)+(len(b)))

def load_dataset(path):
    contents = []
    with open(path, 'r', encoding='UTF-8') as input_data:
        json_content = json.load(input_data)
        # 逐条读取记录
        labels=[]
        jacs=[]
        coss=[]
        counts=[]
        for block in json_content:
            query1 = block['query1']
            query2 = block['query2']
            label = block['label']
            vocab1, vocab2 = temp_vocab(query1,query2)
            words_line1 = []
            words_line2 = []
            token1 = tokenizer(query1)
            token2 = tokenizer(query2)
            count, jac = get_jaccard(token1, token2)
            for word in vocab1:
                words_line1.append(vocab1.get(word, 0))
            for word in vocab2:
                words_line2.append(vocab2.get(word, 0))
            cos = cos_dist(words_line1,words_line2)
            coss.append(cos)
            counts.append(count)
            jacs.append(jac)
            labels.append(int(label))
    return np.array((coss, counts, jacs)).T, np.array(labels)

jacs,labels=load_dataset('KUAKE/data/KUAKE-QQR_train.json')

from xgboost.sklearn import XGBClassifier
## 定义 XGBoost模型
clf = XGBClassifier(colsample_bytree = 0.8, learning_rate = 0.1, max_depth= 5, subsample = 1)
clf.fit(jacs,labels)
jacs1, labels1 = load_dataset('KUAKE/data/KUAKE-QQR_dev.json')
predict = clf.predict(jacs1)

from sklearn.metrics import accuracy_score
print('dev_acc=', accuracy_score(predict, labels1))

with open('../data/KUAKE/KUAKE-QQR_test.json', 'r', encoding='UTF-8') as input_data, \
        open('../prediction_result/KUAKE-QQR_test_pred_jac.json', 'w', encoding='UTF-8') as output_data:
    json_content = json.load(input_data)
    # 逐条读取记录，并将预测好的label赋值
    for block in json_content:
        query1 = block['query1']
        query2 = block['query2']
        label = block['label']
        vocab1, vocab2 = temp_vocab(query1, query2)
        words_line1 = []
        words_line2 = []
        token1 = tokenizer(query1)
        token2 = tokenizer(query2)
        count, jac = get_jaccard(token1, token2)
        for word in vocab1:
            words_line1.append(vocab1.get(word, 0))
        for word in vocab2:
            words_line2.append(vocab2.get(word, 0))
        cos = cos_dist(words_line1, words_line2)
        pre = clf.predict([np.array((cos, count, jac))])
        block['label'] = str(*pre)
    json.dump(json_content, output_data, indent=2, ensure_ascii=False)

