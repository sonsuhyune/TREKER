import json
import argparse
import dill as pk
import numpy as np
from collections import defaultdict

import torch
import torch.nn.utils.rnn as rnn

from info import Info
from util import myprint, prepare_autos, register_hyperparameters_prepare
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from setproctitle import setproctitle

setproctitle("TREK")


class DocInput:

    def __init__(self, title, doc_tokens, sid2tids, eid2tids, eid2etype, eids2rid2sids):
        self.title = title
        self.doc_tokens = doc_tokens
        self.sid2tids = sid2tids

        self.eid2tids = eid2tids
        self.eid2etype = eid2etype
        self.eids2rid2sids = eids2rid2sids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local rank')
    parser.add_argument('--stage', type=str, default='Prepare')
    parser.add_argument('--dataset', type=str, default='DocRED')
    parser.add_argument('--transformer', type=str, default='bert-base-cased')
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--filename', type=str, default='DocRED')

    args = parser.parse_args()
    return args


def type2vec(info, transformer, tokenizer, if_etype=False, if_rtype=False):
    if if_etype:
        id2type, type2word = info.DATA_ID2NER, info.DATA_NER2WORD
    elif if_rtype:
        id2type, type2word = info.DATA_ID2REL, info.DATA_REL2WORD

    type_token_seqs, type_token_masks = [], []
    for type_id in range(len(id2type)):
        words = type2word.get(id2type[type_id], 'not applicable')
        tokens = tokenizer.tokenize(words)
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        type_token_seqs.append(torch.Tensor(tokens))
        type_token_masks.append(torch.ones(len(tokens)))

    type_token_seqs = rnn.pad_sequence(type_token_seqs, batch_first=True, padding_value=0).long()
    type_token_masks = rnn.pad_sequence(type_token_masks, batch_first=True, padding_value=0).float()
    type_embeddings = transformer(input_ids=type_token_seqs, attention_mask=type_token_masks)[0].detach()
    type_embeddings[~type_token_masks.bool()] = info.EXTREME_SMALL_NEGA
    type_embeddings = torch.logsumexp(type_embeddings, dim=1).numpy()

    return type_embeddings


def prepare_doc_input(args, info, tokenizer, doc):
    wid2eids, start_wids, end_wids, eid2etype = defaultdict(set), set(), set(), {}
    for eid, entity in enumerate(doc['vertexSet']):
        eid2etype[eid] = info.DATA_NER2ID[entity[0]['type']]  # entity_id:entity_type_id
        for mid, mention in enumerate(
                entity):
            wid2eids[(mention['sent_id'], mention['pos'][0])].add(eid)
            start_wids.add((mention['sent_id'], mention['pos'][0]))
            end_wids.add((mention['sent_id'], mention['pos'][1] - 1))

    sid2tids, eid2tids, doc_tokens = {}, defaultdict(set), []
    for sid, sent in enumerate(doc['sents']):
        sent_tokens = []
        for wid, word in enumerate(sent):
            tokens = tokenizer.tokenize(word)
            if (sid, wid) in start_wids:
                tokens = [info.MARKER_ENTITY] + tokens
                for eid in wid2eids[(sid, wid)]:
                    eid2tids[eid].add(len(doc_tokens) + len(sent_tokens) + 1)
            if (sid, wid) in end_wids:
                tokens = tokens + [info.MARKER_ENTITY]
            sent_tokens.extend(tokens)
        sent_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)
        sent_tokens = tokenizer.build_inputs_with_special_tokens(sent_tokens)
        sid2tids[sid] = (len(doc_tokens), len(doc_tokens) + len(sent_tokens))
        doc_tokens.extend(sent_tokens)
    if len(doc_tokens) > 2000:
        # print("Pass")
        return None
    doc_tokens = torch.Tensor(doc_tokens).int()
    # print(eid2tids)
    # breakpoint()

    eids2rid2sids = defaultdict(dict)
    if 'labels' in doc:
        for rela in doc['labels']:
            # print()
            # print(rela)
            # print(rela['r'])
            # print(info.DATA_REL2ID[rela['r']])
            # print((rela['h'], rela['t']))
            eids2rid2sids[(rela['h'], rela['t'])][info.DATA_REL2ID[rela['r']]] = [0]
    doc_input = DocInput(doc['title'], doc_tokens, sid2tids, eid2tids, eid2etype, eids2rid2sids)

    return doc_input


def main():
    args = parse_args()
    info = Info(args)

    myprint('=' * 20, info.FILE_STDOUT,0)
    myprint(f'Start {args.stage} {args.dataset}', info.FILE_STDOUT,0)
    myprint('-' * 20, info.FILE_STDOUT,0)

    register_hyperparameters_prepare(args, info)
    myprint('-' * 20, info.FILE_STDOUT,0)

    config, tokenizer, transformer = prepare_autos(args, info)

    myprint('Start Preprocessing the Dataset', info.FILE_STDOUT,0)
    myprint('-' * 20, info.FILE_STDOUT,0)
    total_dic = 0
    total_pass = 0

    mention_hrts, dev_ins = defaultdict(set), defaultdict(set)
    for mode, corpus in info.FILE_CORPUSES.items():
        corpus = json.load(open(corpus, 'r'))

        myprint(f'Start Constructing Doc Inputs for {mode} Files', info.FILE_STDOUT,0)
        doc_inputs, truthes = {}, {}
        for did, doc in enumerate(corpus):

            doc_inputs[doc['title']] = prepare_doc_input(args, info, tokenizer, doc)
            result = prepare_doc_input(args, info, tokenizer, doc)
            if result == None:
                total_pass += 1
                continue
            else:
                doc_inputs[doc['title']] = result
                total_dic += 1

            # if mode != info.MODE_TEST:
            for label in doc['labels']:
                truthes[(doc['title'], label['h'], label['t'], label['r'])] = [0] ############
                if len(doc["vertexSet"]) <= label["h"] or len(doc["vertexSet"]) <= label["t"]:
                    continue
                if mode == info.MODE_TRAIN:
                    for mention_i in doc['vertexSet'][label['h']]:
                        for mention_j in doc['vertexSet'][label['t']]:
                            mention_hrts[mode].add((mention_i['name'], mention_j['name'], label['r']))
                elif mode == info.MODE_DEV:
                    for mention_i in doc['vertexSet'][label['h']]:
                        for mention_j in doc['vertexSet'][label['t']]:
                            if (mention_i['name'], mention_j['name'], label['r']) in mention_hrts[info.MODE_TRAIN]:
                                dev_ins[info.MODE_TRAIN].add((doc['title'], label['h'], label['t'], label['r']))

            if did % 200 == 0: myprint(f'Finish Constructing {did} Doc Inputs for {mode} Files', info.FILE_STDOUT,0)
        pk.dump(doc_inputs, open(info.FILE_INPUTS[mode], 'wb'), -1)
        # if mode != info.MODE_TEST: pk.dump(truthes, open(info.FILE_TRUTHS[mode], 'wb'), -1)
        pk.dump(truthes, open(info.FILE_TRUTHS[mode], 'wb'), -1)
        myprint(f'Finish Constructing Doc Inputs for {mode} Files', info.FILE_STDOUT,0)
        myprint('-' * 20, info.FILE_STDOUT,0)

    for mode, dev_in in info.FILE_DEV_INS.items():
        pk.dump(dev_ins[mode], open(dev_in, 'wb'), -1)
    myprint(f'Finish Preparing dev_ins', info.FILE_STDOUT,0)

    etype_embeddings = type2vec(info, transformer, tokenizer, if_etype=True)
    rtype_embeddings = type2vec(info, transformer, tokenizer, if_rtype=True)
    np.save(info.DATA_NER2VEC, etype_embeddings)
    np.save(info.DATA_REL2VEC, rtype_embeddings)
    myprint('Finish Preparing Entity & Relation Type Embeddings', info.FILE_STDOUT,0)
    myprint('-' * 20, info.FILE_STDOUT,0)

    myprint(f'Finish {args.stage} {args.dataset}', info.FILE_STDOUT,0)
    myprint('=' * 20, info.FILE_STDOUT,0)



if __name__ == '__main__':
    main()