#!/usr/bin/env python3

import torch
from pytorch_pretrained_bert import BertTokenizer, BertConfig, OpenAIGPTModel, OpenAIGPTTokenizer
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification



def load_model(file_str): # path_to_model -> pytorch model
    # Load model file and return the model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.load_state_dict(torch.load(file_str))
    model.eval()
    return model

def inference(tweet_txt, model): # tweet_string -> label_string
# <graded>
    print('Not implemented')
    # TODO
    pass
# </graded>
