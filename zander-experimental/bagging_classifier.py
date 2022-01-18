#!/usr/bin/env python3

# IMPORTS

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# Constants and helper methods
target_names = ['legitimate', 'misinformation', 'irrelevant']

max_length = 96

PATH1 = 'first-augmented-miscov19-covid-twitter-bert-v2'
PATH2 = 'second-augmented-miscov19-covid-twitter-bert-v2'
PATH3 = 'third-augmented-miscov19-covid-twitter-bert-v2'
PATH4 = 'fourth-augmented-miscov19-covid-twitter-bert-v2'
PATH5 = 'fifth-augmented-miscov19-covid-twitter-bert-v2'
PATH6 = 'sixth-augmented-miscov19-covid-twitter-bert-v2'
PATH7 = 'seventh-augmented-miscov19-covid-twitter-bert-v2'

class BaggedTweetClassifier(nn.Module):
    def __init__(self, NUM_MODELS=4):
        super(BaggedTweetClassifier, self).__init__()

        nl =  len(target_names)

        self.num_models = NUM_MODELS

        self.tok1 = AutoTokenizer.from_pretrained(PATH1, local_files_only=True)
        self.bert1 = AutoModelForSequenceClassification \
            .from_pretrained(PATH1, num_labels=nl, local_files_only=True).to("cuda")

        if NUM_MODELS >= 2:
            self.tok2 = AutoTokenizer.from_pretrained(PATH2, local_files_only=True)
            self.bert2 = AutoModelForSequenceClassification \
                .from_pretrained(PATH2, num_labels=nl, local_files_only=True).to("cuda")

        if NUM_MODELS >= 3:
            self.tok3 =  AutoTokenizer.from_pretrained(PATH3, local_files_only=True)
            self.bert3 = AutoModelForSequenceClassification \
                .from_pretrained(PATH3, num_labels=nl, local_files_only=True).to("cuda")

        if NUM_MODELS >= 4:
            self.tok4 =  AutoTokenizer.from_pretrained(PATH4, local_files_only=True)
            self.bert4 = AutoModelForSequenceClassification \
                .from_pretrained(PATH4, num_labels=nl, local_files_only=True).to("cuda")

        if NUM_MODELS >= 5:
            self.tok5 =  AutoTokenizer.from_pretrained(PATH5, local_files_only=True)
            self.bert5 = AutoModelForSequenceClassification \
                .from_pretrained(PATH5, num_labels=nl, local_files_only=True).to("cuda")

        if NUM_MODELS >= 6:
            self.tok6 =  AutoTokenizer.from_pretrained(PATH6, local_files_only=True)
            self.bert6 = AutoModelForSequenceClassification \
                .from_pretrained(PATH6, num_labels=nl, local_files_only=True).to("cuda")

        if NUM_MODELS >= 7:
            self.tok7 =  AutoTokenizer.from_pretrained(PATH7, local_files_only=True)
            self.bert7 = AutoModelForSequenceClassification \
                .from_pretrained(PATH7, num_labels=nl, local_files_only=True).to("cuda")
    # Does not handle preprocessing the input
    def forward(self, x, debug=False):
        # Uncomment this line if x is input without preprocessing.
        # x = clean_text(x)

        probs1 = torch.tensor([0,0,0]).to("cuda")
        probs2 = torch.tensor([0,0,0]).to("cuda")
        probs3 = torch.tensor([0,0,0]).to("cuda")
        probs4 = torch.tensor([0,0,0]).to("cuda")
        probs5 = torch.tensor([0,0,0]).to("cuda")
        probs6 = torch.tensor([0,0,0]).to("cuda")
        probs7 = torch.tensor([0,0,0]).to("cuda")

        in1 = self.tok1(x, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
        out1 = self.bert1(**in1)
        probs1 = out1[0].softmax(1)

        if self.num_models >= 2:
            in2 = self.tok2(x, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            out2 = self.bert2(**in2)
            probs2 = out2[0].softmax(1)

        if self.num_models >= 3:
            in3 = self.tok3(x, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            out3 = self.bert3(**in3)
            probs3 = out3[0].softmax(1)

        if self.num_models >= 4:
            in4 = self.tok4(x, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            out4 = self.bert4(**in4)
            probs4 = out4[0].softmax(1)

        if self.num_models >= 5:
            in5 = self.tok5(x, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            out5 = self.bert5(**in5)
            probs5 = out5[0].softmax(1)

        if self.num_models >= 6:
            in6 = self.tok6(x, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            out6 = self.bert6(**in6)
            probs6 = out6[0].softmax(1)

        if self.num_models >= 7:
            in7 = self.tok7(x, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            out7 = self.bert7(**in7)
            probs7 = out7[0].softmax(1)

        avg_prob = (probs1 + probs2 + probs3 + probs4 + probs5 + probs6 + probs7) / float(self.num_models)

        if debug:
            # Uncomment if using 6 models
            # print(f'-------------------------------------------------------------------------------')
            # print(f'| label        | model1 | model2 | model3 | model4 | model5 | model6 | avg_pr |')
            # print(f'| legitimate   | {probs1[0][0].item():.4f} | {probs2[0][0].item():.4f} | {probs3[0][0].item():.4f} | {probs4[0][0].item():.4f} | {probs5[0][0].item():.4f} | {probs6[0][0].item():.4f} | {avg_prob[0][0].item():.4f} |')
            # print(f'| misinfo      | {probs1[0][1].item():.4f} | {probs2[0][1].item():.4f} | {probs3[0][1].item():.4f} | {probs4[0][1].item():.4f} | {probs5[0][1].item():.4f} | {probs6[0][1].item():.4f} | {avg_prob[0][1].item():.4f} |')
            # print(f'| irrelevant   | {probs1[0][2].item():.4f} | {probs2[0][2].item():.4f} | {probs3[0][2].item():.4f} | {probs4[0][2].item():.4f} | {probs5[0][2].item():.4f} | {probs6[0][2].item():.4f} | {avg_prob[0][2].item():.4f} |')
            # print(f'-------------------------------------------------------------------------------')
            print(f'-------------------------------------------------------------')
            print(f'| label        | model1 | model2 | model3 | model4 | avg_pr |')
            print(f'| legitimate   | {probs1[0][0].item():.4f} | {probs2[0][0].item():.4f} | {probs3[0][0].item():.4f} | {probs4[0][0].item():.4f} | {avg_prob[0][0].item():.4f} |')
            print(f'| misinfo      | {probs1[0][1].item():.4f} | {probs2[0][1].item():.4f} | {probs3[0][1].item():.4f} | {probs4[0][1].item():.4f} | {avg_prob[0][1].item():.4f} |')
            print(f'| irrelevant   | {probs1[0][2].item():.4f} | {probs2[0][2].item():.4f} | {probs3[0][2].item():.4f} | {probs4[0][2].item():.4f} | {avg_prob[0][2].item():.4f} |')
            print(f'-------------------------------------------------------------')
            return target_names[avg_prob.argmax()]

        return avg_prob.argmax()

def to_cpu(x):
    return x.cpu()

model = BaggedTweetClassifier(4)

# Accuracy and confusion matrix for original miscov19 dataset

raw_miscov_df = pd.read_csv('processed_for_bagging_miscov19_p.csv')
df = raw_miscov_df[['text','label']]
df.dropna()
df['text'] = df['text'].astype(str)
df['preds'] = df['text'].apply(model)
df['preds'] = df['preds'].apply(to_cpu)

y_true = df['label'].tolist()
y_pred = df['preds'].tolist()

acc1 = f'Ensemble Accuracy: {accuracy_score(y_true, y_pred)}\n'
print(acc1)

confusion_mat = confusion_matrix(y_true, y_pred)
print(confusion_mat)

results_file = open("results_bagging.txt", "a")
n = results_file.write(acc1)
n = results_file.write("Ensemble Confusion Matrix for miscov19\n")
n = results_file.write(np.array_str(confusion_mat))
results_file.close()

# Accuracy and confusion matrix for augmented miscov19 dataset

raw_combined_df = pd.read_csv('processed_for_bagging_combined_data.csv')
df = raw_combined_df[['text','label']]
df.dropna()
df['text'] = df['text'].astype(str)
df['preds'] = df['text'].apply(model)
df['preds'] = df['preds'].apply(to_cpu)

y_true2 = df['label'].tolist()
y_pred2 = df['preds'].tolist()

acc2 = f'\nEnsemble Accuracy (Augmented): {accuracy_score(y_true2, y_pred2)}\n'
print(acc2)

confusion_mat2 = confusion_matrix(y_true2, y_pred2)
print(confusion_mat2)

results_file = open("results_bagging.txt", "a")
n = results_file.write(acc2)
n = results_file.write("Ensemble Confusion Matrix for augmented miscov19\n")
n = results_file.write(np.array_str(confusion_mat2))
results_file.close()
