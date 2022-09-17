# 【NLP】医学搜索Query相关性判断

[竞赛链接](https://tianchi.aliyun.com/competition/entrance/532001/introduction)

------

## 1.任务
Query（即搜索词）之间的相关性是评估两个Query所表述主题的匹配程度，即判断Query-A和Query-B是否发生转义，以及转义的程度。
给定Query-A和Query-B，判断他们之间的相关性。
- 2分：表示A与B等价，表述完全一致
- 1分： B为A的语义子集，B指代范围小于A
- 0分：B为A的语义父集，B指代范围大于A； 或者A与B语义毫无关联

## 2.参考

本人小白一枚，方案照着[“公益AI之星”挑战赛-新冠疫情相似句对判定大赛第二名方案](https://github.com/thunderboom/text_similarity)做的，这是“公益AI之星”挑战赛[比赛链接](https://tianchi.aliyun.com/competition/entrance/231776/introduction)。

## 3.数据增强

原始数据标签0，1，2对应的数量分别为8946，2603，3451

使用传递性生成数据，原始数据qeury1\query2\label=2和query1\quey3\label=2，则可生成数据query2\query3\label=2。

而如果标签为0或1，则传递性可能失效，但是经本人尝试，额外增强0和1标签对准确率略有提，我猜测是大多数据仍满足传递性，数据得到增强，不满足的则降低了模型过拟合

数据增强比例：考虑到正负样本均衡， 标签0，1，2增强比例分别设置为0.1，0.15，0.3


## 4.模型选择

使用预训练模型Bert作为基础模型结构，在pooler层后加了简单的全连接层和Sigmoid激活层。

**预训练模型选择**

- ERNIE

在医疗数据中，往往会存在较多的实体概念；此外文本相似度作为问答任务的子任务，数据描述类型也偏向于口语。ERNIE是百度提出的知识增强的语义表示模型，通过对词、实体等语义单元的掩码，使模型学习完整概念的语义表示，其训练语料包括了百科类文章、新闻资讯、论坛对话。因此ERNIE能够更准确表达语句中实体的语义，且符合口语的情景。

- Roberta_large

Roberta_large是目前大多数NLP任务的SOTA模型。在Roberta_large中文版本使用了动态掩码、全词掩码，增加了训练数据，并改变了生成的方式和语言模型的任务。因此，在医疗文本上，Roberta_large能更好地对文本进行编码。

- Roberta_large_pair

Roberta_large_pair是针对文本对任务提出的专门模型，能够较好地处理语义相似度或句子对问题。因此，在医疗文本相似度任务上，往往能够取得更好的结果。

**预训练模型下载地址**

ERNIE:https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

Roberta_large:https://github.com/ymcui/Chinese-BERT-wwm

Roberta_large_pair:https://github.com/CLUEbenchmark/CLUEPretrainedModels

## 5 模型融合

**单个模型最佳正确率**

- ERNIE: 0.8321

- Roberta_large: 0.8578

- Roberta_large_pair: 0.8534

**融合模型正确率**

我反感模型融合，纯粹是为了提升正确率提升排名，浪费大量的算力获取一点不大的提升。

其实仅用Roberta_large_pair的效果就已经很好了，但为了提升排名还是得这样做

模型融合后正确率为0.8672， 融合方式为使用三个模型的输出概率乘对应权重相加，再取最大值的索引：

```
weight1, weight2, weight3 = 0.5, 0.35, 0.05
final_output = weight1*outputs_pair+weight2*outputs_wwm+weight3*outputs_ernie
max = np.max(final_output, axis=1).reshape(-1, 1)
labels = np.where(final_output == max)[1]  
```

## 6.代码说明

### 6.1 代码文件结构
```
├── code
│   ├── bert.py
│   ├── data_augment.py
│   ├── DataProcessor.py
│   ├── run_ernie.py
│   ├── run_large_roberta_pair.py
│   ├── run_large_roberta_wwm_ext.py
│   ├── test.py
│   ├── test_mix.py
│   ├── train_eval.py
│   └── utils.py
├── data
│   ├── KUAKE
│   │   ├── KUAKE-QQR_dev.json
│   │   ├── KUAKE-QQR_test.json
│   │   └── KUAKE-QQR_train.json
├── longging
│   │── ernie
│   │── roberta_large_pair
│   │── roberta_wwm_large
├── my_model
├── prediction_result
├── pretrain_models
│   │── ERNIE
│   │── roberta_large_pair
│   │── roberta_wwm_large_ext
└── README.md
```

### 6.2 代码文件说明

1、 code部分  
``` 
* bert.py                       bert模型文件   
* utils.py                      包含数据加载，特征转换等一些小的接口   
* train_eval.py                 主要为模型训练、验证和测试的底层接口  
* run_ernie.py                  单独运行ERNIE模型
* run_large_roberta_pair.py     单独运行roberta_large_pair模型
* run_large_roberta_wwm_ext.py  单独运行roberta_large_wwm_ext模型  
* data_augment.py               单独运行，获取增强数据集
* test                          单独使用一个模型预测
* test_mix                      模型融合预测
```

2、 其他部分  
```
data 文件夹                      为官方提供的原始数据
logging 文件夹                   运行每个模型产生的logging日志
my_model 文件夹                  自己训练的模型
pretrain_models 文件夹           预训练模型文件夹
prediction_result 文件夹         预测结果
```

## 7.运行环境

* GPU RTX3090
* ubuntu 20.04.1
* cuda == 11.3
* python == 3.8.13 
* pytorch == 1.10.1 
* transformers==4.21.1   
* numpy==1.22.4

## 8.运行说明

### 8.1 数据增强
```
# 1. 进入code文件夹下
> cd code

# 2. 运行data_augment.py文件，进行数据扩充
> python data_augment.py
```
注： 请先确保项目目录下data/KUAKE存在json文件; 你也可以修改数据增强比例。

### 8.2 训练
```
# 1. 进入code文件夹下
> cd code

# 2. 有三个训练文件，可以随意选择，以run_ernie.py为例, 进行模型训练
> python run_ernie.py
```
注： 请先确保下载预训练权重文件到pretrain_models/ERNIE; 训练后会在logging/ernie下生成log文件；模型会保存到my_model

### 8.3 预测

根据模型先修改test.py设置模型名字和路径，再运行test.py, 例如
```
model_name = 'ernie'
model_path = '../my_model/best_ernie.pkl'
```
结果会保存到prediction_result

如需预测混合模型，需要训练好三个模型，设置test_mix.py中模型路径和模型权重，运行test_mix.py

