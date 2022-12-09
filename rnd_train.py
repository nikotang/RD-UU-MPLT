
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


tokenizer = AutoTokenizer.from_pretrained('roberta-base')

mnli_dataset_tok = load_from_disk('./data/mnli_datasets')
# paws_dataset_tok = load_from_disk('./data/paws_datasets')
# winogrande_dataset_tok = load_from_disk('./data/winogrande_datasets')

"""# 2: Training"""

# https://huggingface.co/docs/transformers/training

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset = mnli_dataset_tok

name = 'mnli'

gc.collect()
torch.cuda.empty_cache()

CUDA_VISIBLE_DEVICES=0

training_args = TrainingArguments(
    output_dir=f'./{name}_results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=1000,                # number of warmup steps for learning rate scheduler
    learning_rate=5e-5,
    weight_decay=5e-4,               # strength of weight decay
    logging_dir=f'./{name}_logs',            # directory for storing logs
    logging_steps=500,
    evaluation_strategy="steps",      # or 'steps', then specify no. of 'eval_steps'
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'      # determine 'best' according to val acc
)

model = RobertaForSequenceClassification.from_pretrained('roberta-base').to("cuda")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset['train'],         # training dataset
    eval_dataset=dataset['validation'],             # evaluation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]      # checks 3 more steps before early stopping
)

trainer.train()

trainer.save_model()
