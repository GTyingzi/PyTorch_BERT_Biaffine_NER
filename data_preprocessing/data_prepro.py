import sys
sys.path.append('./')
import numpy as np
from utils.arguments_parse import args
import json
import torch
from torch.utils.data import DataLoader, Dataset
from data_preprocessing import tools
from tqdm import tqdm

tokenizer=tools.get_tokenizer()
label2id,id2label,num_labels = tools.load_schema()

def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        arguments = []
        for line in lines:
            data = json.loads(line)
            text,entity_list = data['text'],data['entity_list']
            args_dict={}
            if entity_list != []:
                for entity in entity_list:
                    entity_type,entity_argument = entity['type'],entity['argument']

                    if entity_type not in args_dict.keys():
                        args_dict[entity_type] = [entity_argument]
                    else:
                        args_dict[entity_type].append(entity_argument)
                sentences.append(text)
                arguments.append(args_dict)
        return sentences, arguments


def encoder(sentence, argument):
    from utils.arguments_parse import args
    encode_dict = tokenizer.encode_plus(sentence,
                                        max_length=args.max_length,
                                        pad_to_max_length=True)
    encode_sent = encode_dict['input_ids']
    token_type_ids = encode_dict['token_type_ids']
    attention_mask = encode_dict['attention_mask']

    zero = [0 for i in range(args.max_length)]
    span_mask=[ attention_mask for i in range(sum(attention_mask))]
    span_mask.extend([ zero for i in range(sum(attention_mask),args.max_length)])

    span_label = [0 for i in range(args.max_length)]
    span_label = [span_label for i in range(args.max_length)]
    span_label = np.array(span_label)
    for entity_type,args in argument.items():
        for arg in args:
            encode_arg = tokenizer.encode(arg)
            start_idx = tools.search(encode_arg[1:-1], encode_sent)
            end_idx = start_idx + len(encode_arg[1:-1]) - 1
            span_label[start_idx, end_idx] = label2id[entity_type]+1 # 在span_label这个矩阵中，1代表nr,2代表ns,3代表nt

    return encode_sent, token_type_ids, attention_mask, span_label, span_mask



def data_pre(file_path):
    sentences, arguments = load_data(file_path)
    data = []
    for i in tqdm(range(len(sentences))): ##一条条句子读取
        encode_sent, token_type_ids, attention_mask, span_label, span_mask = encoder(
            sentences[i], arguments[i])
        tmp = {}
        tmp['input_ids'] = encode_sent
        tmp['input_seg'] = token_type_ids
        tmp['input_mask'] = attention_mask
        tmp['span_label'] = span_label
        tmp['span_mask'] = span_mask
        data.append(tmp)

    return data


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        one_data = {
            "input_ids": torch.tensor(item['input_ids']).long(),
            "input_seg": torch.tensor(item['input_seg']).long(),
            "input_mask": torch.tensor(item['input_mask']).float(),
            "span_label": torch.tensor(item['span_label']).long(),
            "span_mask": torch.tensor(item['span_mask']).long()
        }
        return one_data

def yield_data(file_path):
    tmp = MyDataset(data_pre(file_path))
    return DataLoader(tmp, batch_size=args.batch_size, shuffle=True)


if __name__ == '__main__':

    data = data_pre(args.train_path)