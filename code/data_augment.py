'请先运行此程序获取增强的数据集'
import pandas as pd
from utils import read_csv
import numpy as np


def list_to_dataframe(datalist):
    """
    list to DataFrame
    """
    question1 = []
    question2 = []
    label = []
    for row in datalist:
        question1.append(row[0])
        question2.append(row[1])
        label.append(int(row[2]))
    return pd.DataFrame({'query1': question1, 'query2': question2, 'label': label})

def dataframe_to_list(dataframe):
    """
    DataFrame to list
    """
    data_list = []
    for idx, row in dataframe.iterrows():
        row_list = [row['query1'], row['query2'], int(row['label'])]
        data_list.append(row_list)
    return data_list

def sentence_set_pair(train_examples, file_name=None, random_state=20):
    questions1 = []
    questions2 = []
    labels = []
    df_train = list_to_dataframe(train_examples)
    column1, column2, column3 = 'query1', 'query2', 'label'
    query_1_list = list(np.unique(df_train['query1']))
    for query_tag in query_1_list:
        df_query = df_train[df_train['query1'] == query_tag]
        query_same_set = df_query[df_query['label'] == 2]['query2'].tolist()
        query_sim_set = df_query[df_query['label'] == 1]['query2'].tolist()
        query_diff_set = df_query[df_query['label'] == 0]['query2'].tolist()
        if len(query_same_set) >= 1:  # 如果有与query1相同的问题
            if len(query_diff_set) >= 1:  # 类别间
                for query_1 in query_same_set:
                    for query_2 in query_diff_set:
                        questions1.append(query_1)
                        questions2.append(query_2)
                        labels.append('0')
            if len(query_sim_set) >= 1:  # 相似
                for query_1 in query_same_set:
                    for query_2 in query_sim_set:
                        questions1.append(query_1)
                        questions2.append(query_2)
                        labels.append('1')
            if len(query_same_set) >= 2:  # 类别内
                for i in range(len(query_same_set) - 1):
                    for j in range(i + 1, len(query_same_set)):
                        questions1.append(query_same_set[i])
                        questions2.append(query_same_set[j])
                        labels.append('2')
    new_df = pd.DataFrame({column1: questions1, column2: questions2, column3: labels})
    df_postive = new_df[new_df[column3] == '2']
    df_negative = new_df[new_df[column3] == '0']
    df_similar = new_df[new_df[column3] == '1']

    df_postive = df_postive.sample(frac=0.3, replace=False, random_state=random_state)
    df_negative = df_negative.sample(frac=0.1, replace=False, random_state=random_state)
    df_similar = df_similar.sample(frac=0.15, replace=False, random_state=random_state)

    new_df = pd.concat([df_postive, df_similar, df_negative], ignore_index=True)
    if file_name is not None:
        new_df.to_csv(file_name, index=False)
        print("file {} saved, lens {}".format(file_name, len(new_df)))
    return dataframe_to_list(new_df)

if __name__ == '__main__':
    data_path = '../data/KUAKE/KUAKE-QQR_train.json'
    augment_save_path = '../data/KUAKE/KUAKE-QQR_augment.csv'
    train_examples = read_csv(data_path)
    lis = sentence_set_pair(train_examples, augment_save_path)