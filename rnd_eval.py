
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# import pickle
# import os
import numpy as np
import pandas as pd
import gc

from transformers import AutoTokenizer, RobertaForSequenceClassification, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets, DatasetDict
import evaluate

from statistics import mean

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

mnli_dataset_tok = load_from_disk('./data/mnli_datasets')
# paws_dataset_tok = load_from_disk('./data/paws_datasets')
# winogrande_dataset_tok = load_from_disk('./data/winogrande_datasets')

dataset = mnli_dataset_tok

# name = 'mnli'

"""# 2: Training"""

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

gc.collect()
torch.cuda.empty_cache()

CUDA_VISIBLE_DEVICES=0

"""# 3: Evaluate"""

from sys import argv

assert len(argv) == 3
MODEL_DIR = argv[1]
OUTPUT_DIR = argv[2]


test_model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR).to("cuda")
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

test_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 64   
)

tester = Trainer(
              model = test_model, 
              args = test_args, 
              compute_metrics = compute_metrics)

# get test set accuracy
tester.evaluate(eval_dataset=dataset['test'])

logits, references, _ = tester.predict(dataset['test'])

softmax = nn.Softmax(dim=-1)

soft_logits = [softmax(torch.tensor(logit)) for logit in logits]
soft_logits = np.stack(np.array(soft_logits, dtype=object))

preds = np.argmax(soft_logits, axis=1)
pred_probs = np.amax(soft_logits, axis=1)

# save the results
results = np.concatenate((pred_probs, preds, references), axis=1)
with open('preds.npy', 'wb') as f:
  np.save(f, results)

# to load the file:
# with open('results.npy', 'rb') as f:
#   results = np.load(f)

avg_confidence = np.mean(pred_probs)
print('average confidence: ', avg_confidence)

confidence_1 = []
confidence_0 = []
for i, prob in enumerate(pred_probs):
  if preds[i] == 1:
    confidence_1.append(prob)
  else:
    confidence_0.append(prob)

print('entailment confidence: ', mean(confidence_1))
print('non-entailment confidence: ', mean(confidence_0))