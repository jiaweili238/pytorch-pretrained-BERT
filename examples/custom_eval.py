import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

starter_gold_file = "/Users/zhangyue/Desktop/starter_gold.json"
starter_pred_file = "/Users/zhangyue/Desktop/starter_pred.json"
bert_pred_file = "/Users/zhangyue/Desktop/All/18-19_Winter/CS224N/Project/squad_cs224n/bert_train/bert_eval/predictions.json"

with open(starter_gold_file, 'r') as fh:
	starter_gold_dict = json_load(fh)

with open(starter_pred_file, 'r') as fh:
	starter_pred_dict = json_load(fh)

with open(bert_pred_file, 'r') as fh:
	bert_pred_file = json_load(fh)

bert_gold_dict = {}
gold_dict = {}
pred_dict = {}

# Use uuid as dictionary key to construct the gold_dict for bert
for key, value in starter_gold_dict.items():
	bert_gold_dict[value['uuid']] = value

# Filter out bert_pred_file 
counter = 1
for key, value in bert_pred_file.items():
	if key in bert_gold_dict.keys():
		pred_dict[str(counter)] = value
		gold_dict[str(counter)] = bert_gold_dict[key]
		counter += 1

results = util.eval_dicts(gold_dict, pred_dict, no_answer = True)
results = util.eval_dicts(gold_dict, pred_dict, no_answer = True)

print(results)

