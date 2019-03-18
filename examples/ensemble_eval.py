# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange
import utils
from utils import read_squad_examples
from utils import convert_examples_to_features
from utils import write_predictions
from utils import RawResult
from utils import eval_results
from utils import *
from tensorboardX import SummaryWriter
from args import get_args
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnsweringLing, BertForQuestionAnsweringLingRelu
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

"""
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
"""


def main(args):
    
    # set up logging and device
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=True)
    logger = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory () already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Generating the dictionaries
    dep_dict, pos_dict, ent_dict, total_features  = generate_dictionary(args.train_ling_features_file, args.eval_ling_features_file, args.test_ling_features_file)   
#    from IPython import embed; embed()
    # Generating total_dictionary
    total_dict = convert_string_features_to_array(total_features, dep_dict, pos_dict, ent_dict)

    # Prepare model
    model = BertForQuestionAnsweringLing.from_pretrained(args.bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    global_step = 0

    # load eval features
    eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative, total_dictionary=total_dict)
    eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)

    test_examples = read_squad_examples(
            input_file=args.test_file, is_training=False, version_2_with_negative=args.version_2_with_negative, total_dictionary=total_dict)
    test_features = convert_examples_to_features(
            examples=test_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)


    # Save a trained model
    """
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)
        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(output_model_file)
        model = BertForQuestionAnsweringLing.from_pretrained(args.bert_model, state_dict=model_state_dict)
    else:
        model = BertForQuestionAnsweringLing.from_pretrained(args.bert_model)

    model.to(device)
    """

    
    # load the best trained model and eval on the eval set and test set
    models = []
    base_path = '/home/zhangyue/results'
    model_names = ['bert_base_linear', 'bert_base_linear_best', 'bert_large_1', 'bert_large_relu_1e-5']


    best_model_file = os.path.join(base_path, model_name[0], "pytorch_model_best.bin")
    model_state_dict = torch.load(best_model_file)
    model = BertForQuestionAnsweringLing.from_pretrained('bert-base-cased', state_dict=model_state_dict)
        model.to(device)
    models.append(model)

    best_model_file = os.path.join(base_path, model_name[1], "pytorch_model_best.bin")
    model_state_dict = torch.load(best_model_file)
    model = BertForQuestionAnsweringLing.from_pretrained('bert-base-cased', state_dict=model_state_dict)
    model.to(device)
    models.append(model)

    best_model_file = os.path.join(base_path, model_name[2], "pytorch_model_best.bin")
    model_state_dict = torch.load(best_model_file)
    model = BertForQuestionAnsweringLing.from_pretrained('bert-large-cased', state_dict=model_state_dict)
    model.to(device)
    models.append(model)


    best_model_file = os.path.join(base_path, model_name[3], "pytorch_model_best.bin")
    model_state_dict = torch.load(best_model_file)
    model = BertForQuestionAnsweringLingRelu.from_pretrained('bert-large-cased', state_dict=model_state_dict)
    model.to(device)    
    models.append(model)

    
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info('Evaluating at the best model')
        results, all_results  = predict(models, eval_examples, eval_features, device,args, logger, args.version_2_with_negative, args.dev_eval_file)

        logger.info('Write the best eval results')
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
        '''
        write_predictions(eval_examples, eval_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.version_2_with_negative, args.null_score_diff_threshold, 'dev')
        '''


        logger.info('Test set at the best model')
        results, all_results  = predict(models, test_examples, test_features, device,args, logger, args.version_2_with_negative, args.test_eval_file)

        logger.info('Write the best test set results')
        output_prediction_file = os.path.join(args.output_dir, "predictions_test.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_test.json")
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_test.json")
        '''
        write_predictions(test_examples, test_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.version_2_with_negative, args.null_score_diff_threshold, 'test')
        '''


        """
        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        all_ling_features = torch.tensor([f.ling_features for f in eval_features], dtype=torch.float)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))

        results  = evaluate(model, eval_examples, eval_features, device,args)
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
        write_predictions(eval_examples, eval_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.version_2_with_negative, args.null_score_diff_threshold)


   
         """


def predict(models, eval_examples, eval_features, device, args, logger,use_squad_v2, eval_file ):
    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_ling_features = torch.tensor([f.ling_features for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_ling_features, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, ling_features, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        ling_features = ling_features.to(device)
        batch_size = input_ids.shape[0]
        max_seq_length = input_ids.shape[1]
        with torch.no_grad():
            batch_start_logits_ensemble = torch.zeros((batch_size, max_seq_length), dtype = torch.long, device = input_ids.device)
            batch_end_logits_ensemble = torch.zeros((batch_size, max_seq_length), dtype = torch.long, device = input_ids.device)
            for model in models:
                # (batch_size, max_seq_length)
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask, ling_features)
                batch_start_logits_ensemble = batch_start_logits_ensemble + batch_start_logits
                batch_end_logits_ensemble = batch_end_logits_ensemble + batch_end_logits

            batch_start_logits = batch_start_logits_ensemble / len(models)
            batch_end_logits = batch_end_logits_ensemble / len(models)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    results = eval_results(eval_examples, eval_features, all_results, eval_file,
        args.n_best_size, args.max_answer_length,
        args.do_lower_case, args.verbose_logging,
        args.version_2_with_negative, args.null_score_diff_threshold)

    results_list = [('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, all_results


def evaluate(model, eval_examples, eval_features, device, args, logger,use_squad_v2, eval_file ):
    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_ling_features = torch.tensor([f.ling_features for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_ling_features, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, ling_features, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        ling_features = ling_features.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask, ling_features)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    model.train()

    results = eval_results(eval_examples, eval_features, all_results, eval_file,
        args.n_best_size, args.max_answer_length,
        args.do_lower_case, args.verbose_logging,
        args.version_2_with_negative, args.null_score_diff_threshold)

    results_list = [('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, all_results



if __name__ == "__main__":
    #main()
    main(get_args())
