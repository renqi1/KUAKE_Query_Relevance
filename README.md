# “公益AI之星”挑战赛-新冠疫情相似句对判定大赛解决方案

队伍：世界命运共同体

名词：初赛-第2名；答辩-第2名

[“公益AI之星”挑战赛-新冠疫情相似句对判定大赛](https://tianchi.aliyun.com/competition/entrance/231776/introduction)

### Index

1. 算法说明
2. 代码说明
3. 运行环境
4. 运行说明

------

## 1.算法说明

在本次比赛中使用了数据增强、模型选择、模型训练、模型融合、测试数据增强等多模块的相关技术。

### 1.1数据增强

在数据增强部分，根据数据的来源分为外部数据增强和内部数据增强。

- **外部数据增强**  

引入外部数据能够增加训练数据量，使模型拥有更强的能力。但外部数据会导致模型与本身的任务存在一定偏离，所以如何选择合适的外部数据成为该部分的难点。

本次使用外部数据来源为平安医疗科技疾病问答迁移学习比赛（chip2019）https://biendata.com/competition/chip2019/。其数据集是医疗领域的文本相似度任务，但涉及的病种、以及语言形式与该比赛数据差异较大。

为了能够选出符合该任务的外部数据，我们采用模型选择的形式对外部数据进行筛选。首先使用比赛中的原始数据构建筛选模型（采用roberta_large_pair模型)，后对chip2019的数据进行预测。筛选预测的概率处于(0.20~0.80)之间的数据作为外部增强数据。

-  **内部数据增强**

内部数据增强主要针对在该数据集上的以下特点进行：

1. 数据形式比较规范，同一个query1对应多个query2。因此，可以使用**传递性**生成数据，如原始数据qeury1\query2\label=1和query1\quey3\label=1，则可生成数据query2\query3\label=1。此外，为了保持数据类别的平衡，需要对生成的数据按照比例进行采样。

2. 根据竞赛数据中的说明，数据集应包括十个病种，但训练集和验证集只有八种，缺少“肺结核”和“支气管炎”这两个病种。因此，我们决定从“呼吸道感染”、“哮喘”、“支气管炎”等三个类别的原始文本，通过病名/药名的替换生成**缺失类别**的数据。

   药名和病名的词典是通过计算一组句子（在该数据集上常为5对句子）的最长公共字串，然后通过简单的人工筛选得出。

### 1.2模型选择

在本次比赛中，经过多次实验。最终使用预训练模型Bert作为基础模型结构，在pooler层后加了简单的全连接层和Sigmoid激活层。

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

### 1.3模型训练

- Multi-sample-dropout

在训练过程中，由于Bert后接了dropout层。为了加快模型的训练，我们使用multi-sample-dropout技术。通过对Bert后的dropout层进行多次sample，并对其多次输出的loss进行平均，增加了dropout层的稳定性，同时使得Bert后面的全连接层相较于前面的Bert_base部分能够得到更多的训练。

- 数据交叉

通过数据交叉，即训练时使用不同的数据进行组合，能够在数据层面增加模型的多样性。这次比赛中，将难的数据集（外部数据）给更强大的模型，使小的模型能够精准预测，大的模型更具鲁棒性。

| 模型               | 原始数据 | 增强数据（外部数据） | 增强数据（传递性） | 增强数据（新类别） |
| ------------------ | -------- | -------------------- | ------------------ | ------------------ |
| ERNIE              | yes      | no                   | yes                | yes                |
| Roberta_large_pair | yes      | yes                  | yes                | yes                |
| Roberta_large      | yes      | yes                  | yes                | no                 |

### 1.4模型融合

在模型融合部分我们使用简单三个模型的输出概率求平均，得到最总的融合概率。然后根据融合概率和阈值的关系，得到相应对的标签。

### 1.5测试数据增强

在医疗文本相似度任务中，交换两个文本的数据不会改变该文本对的标签。但是对于Bert来说，交换文本对的位置，会改变位置编码，能使模型从不同的角度取观察这两个文本的相似性。在测试数据增强时，通过计算原始文本对与交换文本对的输出概率的平均值，能够使模型更好地在测试数据上的进行预测。

## 2.代码说明

### 2.1 代码文件结构
```
.
├── code
│   ├── augment_utils.py
│   ├── bert.py
│   ├── cross_validation.py
│   ├── data_augment.py
│   ├── DataProcessor.py
│   ├── main.py
│   ├── main.sh
│   ├── medicine_dict_generate.py
│   ├── run_ernie.py
│   ├── run_large_roberta_pair.py
│   ├── run_large_roberta_wwm_ext.py
│   ├── train_eval.py
│   ├── train.sh
│   └── utils.py
├── data
│   ├── Dataset
│   │   ├── dev.csv
│   │   ├── test.csv
│   │   └── train.csv
│   └── External
│       ├── other_data
│       │   ├── chip2019.csv
│       │   ├── medicine.txt
│       │   ├── new_category.csv
│       │   ├── original_chip2019.csv
│       │   ├── stop_word.txt
│       │   ├── train_augment.csv
│       │   └── train_dev_augment.csv
│       └── pretrain_models
│           ├── chinese_roberta_wwm_large_ext_pytorch
│           ├── ERNIE
│           └── roberta_large_pair
├── prediction_result
│   └── result.csv
├── README.md
└── user_data
    ├── logging
    ├── model_data
    │   ├── ernie.pkl
    │   ├── roberta_large_pair_for_augment.pkl
    │   ├── roberta_large_pair.pkl
    │   └── roberta_wwm_large.pkl
    └── tmp_data
        └── try_medicine_sypmtom.txt        
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


