# -*- encoding: utf-8 -*-
import numpy as np
import torch.nn.functional
import warnings
import torch
import json
from transformers import BertTokenizer
from .loader import map_id_rel
import warnings
import torch
import json
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import random


def get_model(num_labels):
    model = BertForSequenceClassification.from_pretrained('../bert-base-chinese',num_labels)
    return model

#设置随机种子

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
warnings.filterwarnings("ignore")
setup_seed(44)

def test(net_path,text_list,ent1_list,ent2_list,num_labels,show_result=False):
    rel2id, id2rel = map_id_rel()
    USE_CUDA = torch.cuda.is_available()
    model=get_model(num_labels)
    if USE_CUDA:
        model=model.cuda()
    max_length=128
    net=torch.load(net_path)

    # For only CPU device
    #net=torch.load(net_path,map_location=torch.device('cpu') )
    net.eval()
    if USE_CUDA:
        net = net.cuda()
    rel_list = []
    total=0
    with torch.no_grad():
        for text,ent1,ent2 in zip(text_list,ent1_list,ent2_list):
            sent = ent1 + ent2+ text
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            if USE_CUDA:
                indexed_tokens = indexed_tokens.cuda()
                att_mask = att_mask.cuda()
            outputs = net(indexed_tokens, attention_mask=att_mask)
            # print(y)
            if len(outputs) == 1:
                logits = outputs[0] # 保证和旧模型参数的一致性
            else:
                logits = outputs[1]
            _, predicted = torch.max(logits.data, 1)
            result = predicted.cpu().numpy().tolist()[0]
            if show_result:
                print("Source Text: ",text)
                print("Entity1: ",ent1," Entity2: ",ent2," Predict Relation: ",id2rel[result]," True Relation: ",label)
            total+=1
            #print('\n')
            rel_list.append(id2rel[result])
    return rel_list

