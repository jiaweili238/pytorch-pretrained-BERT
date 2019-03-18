from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
import spacy
from tqdm import tqdm, trange
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from examples.run_squad import *

def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    # read all the data in the train file
    for num, entry in enumerate(tqdm(input_data, desc="Data")):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            # Added by Yue, extracting features from paragraph text
            paragraph_features = extract_feature_matrix(paragraph_text)

            # paragraph_text is the paragraph context
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                # for a given question, we have id, start position, end_postion
                qas_id = qa["id"]
                question_text = qa["question"]

                # Added by Yue, extracting features from quetsion text
                question_features = extract_feature_matrix(question_text)

                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                total_features = concatenate_features(question_features, paragraph_features)

                # example is the original for a certain example
                # containning, qas_id, question_text, 
                example = SquadExample(
                    # Added by Yue
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    ling_features=total_features)
                examples.append(example)
    return examples

def remove_whitespace_entities(doc):
    doc.ents = [e for e in doc.ents if not e.text.isspace()]
    return doc

def extract_feature_matrix(sent):
    # The text "sent" and correpsonding tokens'
    docs = feature(sent)
    dependence_list = []
    pos_list = []
    entity_list = []
    stop_word_list = []
    features = []
    # pos_dict, entity_dict, dep_dict = get_annotation_dict()

    for doc in docs:
        entity = doc.ent_type_
        part_of_speech = doc.pos_
        dependence = doc.dep_
        dependence_list.append(dependence)
        pos_list.append(part_of_speech)
        entity_list.append(entity) 
        stop_word_list.append(doc.is_stop)
    features.append(dependence_list)
    features.append(pos_list)
    features.append(entity_list)
    features.append(stop_word_list)
    return features

def concatenate_features(question_features, paragraph_features):
    total_features = []
    for i in range(4):
        total_features.append(question_features[i] + paragraph_features[i])
    return total_features


if __name__ == '__main__':
    train_file = '../data/train-v2.0.json'
    test_file = '../data/test-v2.0.json'
    eval_file = '../data/dev-v2.0.json' 
   
    spacy.prefer_gpu()
    feature = spacy.load('en_core_web_sm')
    feature.add_pipe(remove_whitespace_entities, after='ner')
    train_ling_features = {}
    eval_ling_features = {}
    test_ling_features = {}    
    
    print('Start Reading training data')
    train_examples = read_squad_examples(
        input_file=train_file, is_training=False, version_2_with_negative=True)    
    for train_example in train_examples:
        train_ling_features[train_example.qas_id] = train_example.ling_features
    with open('train_ling_features.json', 'w') as outfile:  
        json.dump(train_ling_features, outfile)
    
    print('Start Reading dev data')
    eval_examples = read_squad_examples(
        input_file=eval_file, is_training=False, version_2_with_negative=True)
    for eval_example in eval_examples:
        eval_ling_features[eval_example.qas_id] = eval_example.ling_features
    with open('eval_ling_features.json', 'w') as outfile:
        json.dump(eval_ling_features, outfile)

    print('Start Reading test data')    
    test_examples = read_squad_examples(
        input_file=test_file, is_training=False, version_2_with_negative=True)
    for test_example in test_examples:
        test_ling_features[test_example.qas_id] = test_example.ling_features
    with open('test_ling_features.json', 'w') as outfile:
        json.dump(test_ling_features, outfile)
