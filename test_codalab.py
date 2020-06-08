import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

_path = os.path.dirname(__file__)
_rel_dict_filename = os.path.join(_path, './props.json')
_pre_trained_filename = os.path.join(_path, './fewrel0-cpu.model')
_tokenizer_filename = os.path.join(_path, './tokenizer/')
_bert_filename = os.path.join(_path, './bert/')

MODEL = (BertModel, BertTokenizer, 'bert-large-uncased')


def get_new_targets(sentence, targets, tokenizer):
    new_targets = []
    for word, target in zip(sentence.split(), targets):
        new_tokens = tokenizer.tokenize(word)
        if len(new_tokens) == 1:
            new_targets.append(target)
            continue
        new_targets.append(target)
        for _ in new_tokens[1:]:
            new_targets.append(target)
    return new_targets


def start_from_target(target, label):
    for index, item in enumerate(target):
        if item == label:
            return index
    if item == label:
        return index
    return -1


def end_from_target(target, label):
    old_label = ''
    for index, item in enumerate(target):
        if item != label and old_label == label:
            return index
        old_label = item
    if item == label:
        return index + 1
    return -1


def get_sentences_and_targets_from_sentence_tuples(tuples_list):
    all_sentences = []
    all_targets = []
    for tuple in tuples_list:
        sentence = ''
        for item in tuple:
            sentence += item[0] + ' '
        all_sentences.append(sentence[:-1])
        all_targets.append([item[1] for item in tuple])
    return all_sentences, all_targets


class RelTaggerModel(nn.Module):
    _bert_hidden_size = 1024

    def __init__(self, language_model, ninp=200, dropout=0.2):
        super().__init__()
        self.language_model = language_model
        self.model_type = 'BERTREL'
        self.dropout = dropout

        self.input_linear = nn.Linear(self._bert_hidden_size, ninp)

        nout = 2
        self.linear_out1 = nn.Linear(ninp, nout)
        self.linear_out2 = nn.Linear(ninp, nout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.linear_out1.weight.data.uniform_(-initrange, initrange)
        self.linear_out2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, lengths):
        output = self.language_model(src)[0]

        output = self.input_linear(output)
        output = F.relu(output)

        out1 = self.linear_out1(output)
        out2 = self.linear_out2(output)

        subj_start, subj_end = [F.softmax(item[lengths[0]:].transpose(0, 1), dim=-1)
                                for item in out1.transpose(0, 2)]

        obj_start, obj_end = [F.softmax(item[lengths[0]:].transpose(0, 1), dim=-1)
                              for item in out2.transpose(0, 2)]

        return subj_start, subj_end, obj_start, obj_end


def get_discriminative_data_from_tuples_for_fewrel(tuples_list, tokenizer):
    all_data = []
    sentences, targets = get_sentences_and_targets_from_sentence_tuples(tuples_list)
    for sentence, target in zip(sentences, targets):
        target = get_new_targets(sentence, target, tokenizer)
        subj_start = start_from_target(target, 'SUBJECT') + 1
        obj_start = start_from_target(target, 'OBJECT') + 1
        subj_end = end_from_target(target, 'SUBJECT') + 1
        obj_end = end_from_target(target, 'OBJECT') + 1

        all_data.append((sentence, subj_start, subj_end, obj_start, obj_end))

    return all_data


def get_all_relations_from_file(filename):
    data = json.load(open(filename))
    all_candidates = []
    for item in data:
        candidates = []
        for relation in item['meta_train']:
            candidates.append(relation['relation'])
        all_candidates.append(candidates)

    return all_candidates


def get_all_sentences_and_relations_from_fewrel(filename):
    json_data = json.load(open(filename))
    sentences = []
    for item in json_data:
        rel_dict = item['meta_test']
        tokens = rel_dict['tokens']

        subject_indices = rel_dict['h'][2][0]
        object_indices = rel_dict['t'][2][0]

        items = []

        for index, token in enumerate(tokens):

            if index in subject_indices:
                items.append((token, 'SUBJECT'))

            if index in object_indices:
                items.append((token, 'OBJECT'))

            items.append((token, ''))

        sentences.append(items)

    return sentences


def load_rel_dict_for_fewrel(filename):
    rel_dict = {}
    json_data = json.load(open(filename))
    for item in json_data:
        rel_dict[item['id']] = {'sentence': item['description'],
                                'label': item['label'],
                                'aliases': item['aliases'],
                                }

    return rel_dict


model_class, tokenizer_class, pretrained_weights = MODEL
tokenizer = tokenizer_class.from_pretrained(_tokenizer_filename)
language_model = model_class.from_pretrained(_bert_filename)


sentences = get_all_sentences_and_relations_from_fewrel(sys.argv[1])
rel_dict = load_rel_dict_for_fewrel(_rel_dict_filename)

dev_data = get_discriminative_data_from_tuples_for_fewrel(sentences, tokenizer)

all_relations = get_all_relations_from_file(sys.argv[1])


def run_model(model, rel_map, sentence, relation):
    relation_sentence = rel_map[relation]['sentence']

    inputs = torch.tensor([[101] + tokenizer.encode(relation_sentence, add_special_tokens=False)
                           + [102] + tokenizer.encode(sentence, add_special_tokens=False)
                           + [102]
                           ])

    length = torch.tensor([len(tokenizer.encode(relation_sentence, add_special_tokens=False)) + 1])
    subj_starts, subj_ends, obj_starts, obj_ends = model(inputs.cpu(), length)

    return subj_starts[0], subj_ends[0], obj_starts[0], obj_ends[0], inputs[0]


def run_generative_full_matches(model,
                                rel_map,
                                sentence,
                                relation):
    subj_start_ohv, subj_end_ohv, obj_start_ohv, obj_end_ohv, inputs = run_model(model, rel_map, sentence,
                                                                                 relation)

    model_has_candidates = False
    threshold = 1

    adversarial_score = min(subj_start_ohv[0], subj_end_ohv[0], obj_start_ohv[0], obj_end_ohv[0])
    if adversarial_score < threshold:
        model_has_candidates = True

    return model_has_candidates, adversarial_score


def test_with_full_match(model):
    all_relation_indices = []

    for tuple, candidate_relations in zip(dev_data, all_relations):
        sentence, _, _, _, _ = tuple

        old_adversarial_score = 1
        relation_index = -1

        for index, new_relation in enumerate(candidate_relations):
            is_positive_match, adversarial_score = \
                run_generative_full_matches(model,
                                            rel_dict,
                                            sentence,
                                            new_relation)

            if is_positive_match and adversarial_score < old_adversarial_score:
                relation_index = index
                old_adversarial_score = adversarial_score

        all_relation_indices.append(relation_index)

    return all_relation_indices


if __name__ == '__main__':
    model = RelTaggerModel(language_model)
    checkpoint = torch.load(_pre_trained_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cpu()
    labels = test_with_full_match(model)

    print(labels)
