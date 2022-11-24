import torch
from torch import nn
from utils.arguments_parse import args
from data_preprocessing import tools

label2id,id2label,num_labels=tools.load_schema()
num_label = num_labels+1

class Span_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self,span_logits,span_label,span_mask):
        # batch_size,seq_len,hidden=span_label.shape
        '''
        span_logits : {batch_size,max_seq_len,out_features,num_label}
        span_label : {batch_size,max_seq_len,out_features=128}
        span_mask : {batch_size,max_seq_len,max_seq_len=128}
        '''
        span_label = span_label.view(size=(-1,)) # {batch_size * max_seq_len * out_features}
        span_logits = span_logits.view(size=(-1, num_label))# {batch_size * max_seq_len * out_features,num_labels}
        span_loss = self.loss_func(input=span_logits, target=span_label) # {batch_size * max_seq_len * out_features}

        span_mask = span_mask.view(size=(-1,)) # {batch_size * max_seq_len * out_features}
        span_loss *=span_mask
        avg_se_loss = torch.sum(span_loss) / span_mask.size()[0]
        return avg_se_loss





