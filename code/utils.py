import copy
import json
import logging
import numpy as np
import torch.utils.data as Data



logger = logging.getLogger(__name__)

def read_csv(path):
    """read csv file to list"""
    data_list = []
    with open(path, 'r', encoding='UTF-8') as input_data:
        json_content = json.load(input_data)
            # 逐条读取记录
        for block in json_content:
            text_a = block['query1']
            text_b = block['query2']
            label = block['label']
            if label not in ['0', '1', '2']:
                label = 0
            data_list.append([text_a, text_b, int(label)])
        return data_list


def get_dateset_labels(path):
    label_list = []
    with open(path, "r", encoding="utf-8") as (input_data):
        json_content = json.load(input_data)
        # 逐条读取记录
        for block in json_content:
            label = block['label']
            if label not in ['0', '1', '2']:
                label = 0
            label_list.append(int(label))
    return label_list


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    pad_token=0,
    pad_token_segment_id=0,
    data_type=None,
):
    """
    :param examples: List [ sentences1,sentences2,label]
    :param tokenizer: Instance of a tokenizer that will tokenize the examples
    :param max_length: Maximum example length
    :param pad_token: 0
    :param pad_token_segment_id: 0
    :return: [(input_ids, attention_mask, token_type_ids, label), ......]
    """
    features = []
    for example in examples:
        inputs = tokenizer.encode_plus(example[0], example[1], add_special_tokens=True, max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        label = example[2]
        features.append(
            InputFeatures(input_ids, attention_mask, token_type_ids, label)
        )
    return features


class BuildDataSet(Data.Dataset):
    """
    将经过convert_examples_to_features的数据 包装成 Dataset
    """
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = np.array(feature.input_ids)
        attention_mask = np.array(feature.attention_mask)
        token_type_ids = np.array(feature.token_type_ids)
        label = np.array(feature.label)

        return input_ids, attention_mask, token_type_ids, label

    def __len__(self):
        return len(self.features)


def config_to_dict(config):

    output = copy.deepcopy(config.__dict__)
    if hasattr(config.__class__, "model_type"):
        output["model_type"] = config.__class__.model_type
    output['device'] = config.device.type
    return output


def config_to_json_string(config):
    """Serializes this instance to a JSON string."""
    return json.dumps(config_to_dict(config), indent=2, sort_keys=True) + '\n'
