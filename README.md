# 【NLP】医学搜索Query相关性判断

[竞赛链接](https://tianchi.aliyun.com/competition/entrance/532001/introduction)

### Index

1. 
2. 
3. 
4. 

------

## 1.任务

## 2.参考

本人小白一枚，方案照着[公益AI之星”挑战赛-新冠疫情相似句对判定大赛第二名方案](https://github.com/thunderboom/text_similarity)做的。[比赛链接](https://tianchi.aliyun.com/competition/entrance/231776/introduction)

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

-  Roberta_large_pair

Roberta_large_pair是针对文本对任务提出的专门模型，能够较好地处理语义相似度或句子对问题。因此，在医疗文本相似度任务上，往往能够取得更好的结果。

**预训练模型下载地址**

ERNIE:https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

Roberta_large:https://github.com/ymcui/Chinese-BERT-wwm

Roberta_large_pair:https://github.com/CLUEbenchmark/CLUEPretrainedModels

## 模型融合

**单个模型最佳正确率**

ERNIE:0.8321  Roberta_large:0.8578  Roberta_large_pair:0.8534

**融合模型正确率**

模型融合部分使用三个模型的输出概率乘对应权重相加：

```
weight1, weight2, weight3 = 0.5, 0.35, 0.05
final_output = weight1*outputs_pair+weight2*outputs_wwm+weight3*outputs_ernie
max = np.max(final_output, axis=1).reshape(-1, 1)
labels = np.where(final_output == max)[1]  
```
Accuracy = 0.8672

## 2.代码说明

### 2.1 代码文件结构
```
.
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
│   │── ERNIR
│   │── roberta_large_pair
│   │── roberta_wwm_large_ext
└── README.md
     
```

### 2.2 说明

1、 code部分  
``` 
* bert.py                       bert模型文件  
* DataProcessor.py              数据加载和处理文件  
* utils.py                      包含数据特征转换等一些小的接口   
* train_eval.py                 主要为模型训练、验证和测试的底层接口
* cross_validation.py           交叉验证部分（对train_eval.py的封装）  
* run_ernie.py                  单独运行ERNIE模型
* run_large_roberta_pair.py     单独运行roberta_large_pair模型
* run_large_roberta_wwm_ext.py  单独运行roberta_large_wwm_ext模型
* main.py                       通过模型融合进行对测试集进行测试
* train.sh                      通过脚本执行三个模型训练文件
* main.sh                       通过脚本执行main.py
* medicine_dict_generate.py     对train.csv和dev.csv抽取药名和病名（不可直接用，后期需要手工再分离）  
* data_augment.py               数据增强部分（包括三部分）
* augment_utils.py              数据增强时用到的一些接口
```

2、 data部分  
```
Dataset 文件夹                   为官方提供的原始数据

External 文件夹                  为用作模型数据增强的数据
* original_chip2019.csv         原始官方chip2019的train文件
* stop_word.txt                 停用词词典
* medicine.txt                  通过medicine_dict_generate.py程序筛选后，人工筛选出的药物词典
* new_category.csv              新扩增的“肺结核”和“支气管炎”两个病种的数据
* train_augment.csv             仅用train.csv通过传递性进行的1:1数据抽样（尽在模型训练对比用）
* train_dev_augment.csv         用train.csv和dev.csv通过传递性进行的1:1数据抽样（在训练生成模型文件时用）
* chip2019.csv                  仅用官方原始数据训练的roberta_large_pair模型对original_chip2019.csv筛选在预测概率在(0.20, 0.80)间的数据做1:1抽样

pretrain_models 文件夹           预训练模型文件夹
```

3、user_data部分
```
logging 文件夹                               运行每个模型产生的logging日志

model_data 文件夹                            训练数据后，保存的模型文件
* roberta_large_pair_for_augment.pkl        用作数据增强保存的模型文件（用官方提供的全部数据训练）
* ernie.pkl                                 ernie模型文件
* roberta_large_pair.pkl                    roberta_large_pair模型文件
* roberta_wwm_large.pkl                     roberta_wwm_large模型文件
```

4、prediction_result部分
```
通过三个模型融合进行预测的结果，会保存在result.csv文件中
```

5、README.md 


## 3.运行环境

* ubuntu 16.04.6
* cuda == 10.2
* CUDNN == 440.33.01
* python == 3.7.4 
* pytorch == 1.31 
* transformers==2.3.0  
* pandas==0.25.1  
* numpy==1.17.2  

## 4.运行说明

### 4.1 数据增强
```bash
# 1. 进入code文件夹下
> cd code

# 2. 运行data_augment.py文件，进行数据扩充
> python data_augment.py
```
注：  
1、请先确保项目目录下data/Dataset 存在train.csv、dev.csv和test.csv文件。（可从官网获取）   
2、该步骤会在data/External/other_data文件下, 生成新增的四份数据，分别是：  

* train_augment.csv  
* train_dev_augment.csv  
* new_category.csv  
* chip2019.csv  

3、若存在用于数据增强的数据文件，可跳过此步。      
4、其中，由于最后训练模型文件时，采用了全数据（train+dev）的形式，故在数据增强会采用train_dev_augment.csv。    


### 4.2 训练模型
```bash
# 1. 进入code文件夹下
> cd code

# 2. 运行train.sh文件, 进行模型训练（会训练三个模型, 需要一定时间）
> bash train.sh
```
注：  
1、 在训练生成模型文件时，确保data/External/other_data文件夹下有以下三份数据：  
* train_dev_augment.csv  
* new_category.csv  
* chip2019.csv  

2、 模型文件会保存在user_data/model_data文件夹下，分别是：  

* ernie.pkl  
* roberta_large_pair.pkl  
* roberta_wwm_large.pkl  

3、若存在可用的模型文件，可跳过此步。    
4、 同时会在user_data/logging下生成logging文件。

### 4.3 模型预测

```bash
# 1. 进入code文件夹下
> cd code

# 2. 运行main.sh文件, 进行预测
> bash main.sh
```
注：  
1、 在用模型文件进行预测时，确保user_data/model_data文件夹有以下文件：  

* ernie.pkl  
* roberta_large_pair.pkl  
* roberta_wwm_large.pkl  

2、预测结果会保存在prediction_result/result.csv文件中。  


