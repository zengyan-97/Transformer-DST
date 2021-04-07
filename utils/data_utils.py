"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import sys
import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
from copy import deepcopy
from .fix_label import fix_general_label_error

flatten = lambda x: [i for s in x for i in s]
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}


def make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, op_code='4', dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]

    op_labels = ['carryover'] * len(slot_meta)
    generate_y = []
    keys = list(turn_dialog_state.keys())
    for k in keys:
        v = turn_dialog_state[k]
        if v == 'none':
            turn_dialog_state.pop(k)
            continue
        vv = last_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv != v:
                if v == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
                    op_labels[idx] = 'dontcare'
                elif v == 'yes' and OP_SET[op_code].get('yes') is not None:
                    op_labels[idx] = 'yes'
                elif v == 'no' and OP_SET[op_code].get('no') is not None:
                    op_labels[idx] = 'no'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([tokenizer.tokenize(v) + ['[EOS]'], idx])
            elif vv == v:
                op_labels[idx] = 'carryover'
        except ValueError:
            continue

    for k, v in last_dialog_state.items():
        vv = turn_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv is None:
                if OP_SET[op_code].get('delete') is not None:
                    op_labels[idx] = 'delete'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([['[NULL]', '[EOS]'], idx])
        except ValueError:
            continue
    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
    if len(generate_y) > 0:
        generate_y = sorted(generate_y, key=lambda lst: lst[1])
        generate_y, _ = [list(e) for e in list(zip(*generate_y))]

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]

    return op_labels, generate_y, gold_state


def postprocessing(slot_meta, ops, last_dialog_state,
                   generated, tokenizer, op_code, gold_gen={}):

    gid = 0

    for st, op in zip(slot_meta, ops):
        if op == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
            last_dialog_state[st] = 'dontcare'
        elif op == 'yes' and OP_SET[op_code].get('yes') is not None:
            last_dialog_state[st] = 'yes'
        elif op == 'no' and OP_SET[op_code].get('no') is not None:
            last_dialog_state[st] = 'no'
        elif op == 'delete' and last_dialog_state.get(st) and OP_SET[op_code].get('delete') is not None:
            last_dialog_state.pop(st)
        elif op == 'update':
            g = tokenizer.convert_ids_to_tokens(generated[gid])
            gen = []
            for gg in g:
                if gg == '[EOS]':
                    break
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gid += 1
            gen = gen.replace(' : ', ':').replace('##', '')
            if gold_gen and gold_gen.get(st) and gold_gen[st] not in ['dontcare']:
                gen = gold_gen[st]

            if gen == '[NULL]' and last_dialog_state.get(st) and not OP_SET[op_code].get('delete') is not None:
                last_dialog_state.pop(st)
            else:
                last_dialog_state[st] = gen

    return generated, last_dialog_state


def make_slot_meta(ontology):
    meta = []
    change = {}
    idx = 0
    max_len = 0
    for i, k in enumerate(ontology.keys()):
        d, s = k.split('-')
        if d not in EXPERIMENT_DOMAINS:
            continue
        if 'price' in s or 'leave' in s or 'arrive' in s:
            s = s.replace(' ', '')
        ss = s.split()
        if len(ss) + 1 > max_len:
            max_len = len(ss) + 1
        meta.append('-'.join([d, s]))
        change[meta[-1]] = ontology[k]
    return sorted(meta), change


def prepare_dataset(data_path, tokenizer, slot_meta,
                    n_history, max_seq_length, diag_level=False, op_code='4'):
    dials = json.load(open(data_path))
    data = []
    domain_counter = {}
    max_resp_len, max_value_len = 0, 0
    max_line = None

    c = 0

    for i, dial_dict in enumerate(dials):
        if (i+1) % 200 == 0:
            print("prepare {:}/{:}".format(i+1, len(dials)))
            sys.stdout.flush()

        for domain in dial_dict["domains"]:
            if domain not in EXPERIMENT_DOMAINS:
                continue
            if domain not in domain_counter.keys():
                domain_counter[domain] = 0
            domain_counter[domain] += 1

        dialog_history = []
        last_dialog_state = {}
        last_uttr = ""
        for ti, turn in enumerate(dial_dict["dialogue"]):
            turn_domain = turn["domain"]
            if turn_domain not in EXPERIMENT_DOMAINS:
                continue
            turn_id = turn["turn_idx"]
            turn_uttr = (turn["system_transcript"] + ' ; ' + turn["transcript"]).strip()
            dialog_history.append(last_uttr)
            turn_dialog_state = fix_general_label_error(turn["belief_state"], False, slot_meta)
            last_uttr = turn_uttr

            op_labels, generate_y, gold_state = make_turn_label(slot_meta, last_dialog_state,
                                                                turn_dialog_state,
                                                                tokenizer, op_code)

            if (ti + 1) == len(dial_dict["dialogue"]):
                is_last_turn = True
            else:
                is_last_turn = False

            turn_uttr = tokenizer.tokenize(turn_uttr)
            dial_his = tokenizer.tokenize(' '.join(dialog_history[-n_history:]))

            slot_id = tokenizer.convert_tokens_to_ids(['[SLOT]'])[0]
            mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

            instance = TrainingInstance(dial_dict["dialogue_idx"], turn_domain,
                                        turn_id, turn_uttr, dial_his,
                                        last_dialog_state, op_labels,
                                        generate_y, gold_state, max_seq_length, slot_meta,
                                        is_last_turn, op_code=op_code, slot_id=slot_id, mask_id=mask_id)

            instance.make_instance(tokenizer)
            data.append(instance)

            c += 1

            last_dialog_state = turn_dialog_state

    return data


class TrainingInstance:
    def __init__(self, ID,
                 turn_domain,
                 turn_id,
                 turn_utter,
                 dialog_history,
                 last_dialog_state,
                 op_labels,
                 generate_y,
                 gold_state,
                 max_seq_length,
                 slot_meta,
                 is_last_turn,
                 op_code='4', slot_id=1, mask_id=4):
        self.id = ID
        self.turn_domain = turn_domain
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.dialog_history = dialog_history
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.generate_y = generate_y
        self.op_labels = op_labels
        self.gold_state = gold_state
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        self.op2id = OP_SET[op_code]

        self.update_id = self.op2id['update']
        self.slot_id = slot_id
        self.mask_id = mask_id  # For generator

    def shuffle_state(self, rng, slot_meta=None):
        new_y = []
        gid = 0
        for idx, aa in enumerate(self.op_labels):
            if aa == 'update':
                new_y.append(self.generate_y[gid])
                gid += 1
            else:
                new_y.append(["dummy"])
        if slot_meta is None:
            temp = list(zip(self.op_labels, self.slot_meta, new_y))
            rng.shuffle(temp)
        else:
            indices = list(range(len(slot_meta)))
            for idx, st in enumerate(slot_meta):
                indices[self.slot_meta.index(st)] = idx
            temp = list(zip(self.op_labels, self.slot_meta, new_y, indices))
            temp = sorted(temp, key=lambda x: x[-1])
        temp = list(zip(*temp))
        self.op_labels = list(temp[0])
        self.slot_meta = list(temp[1])
        self.generate_y = [yy for yy in temp[2] if yy != ["dummy"]]

    def make_instance(self, tokenizer, max_seq_length=None,
                      word_dropout=0., slot_token='[SLOT]'):
        """
        TODO: Do not wrap into Tensor at this step. (Some errors might occur)
        """
        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        self.domain_id = domain2id[self.turn_domain]
        self.op_ids = [self.op2id[a] for a in self.op_labels]

        # TODO: For generator
        self.generate_ids = [tokenizer.convert_tokens_to_ids(y[:-1] + ['[SEP]']) for y in self.generate_y]

        # It could be [];
        #   Then, input_id_g, input_mask_rel_pos would be [], n_updates would be 0
        # print("generate_y, ", self.generate_y)  # y ends with '[EOS]'
        # print("generate_ids, ", self.generate_ids)  # val, sep

        self.input_id_g, self.input_mask_rel_pos, self.lm_label_ids = [], [], []  # nested list: n updates
        # input_mask_rel_pos: to predict next token (AR objective)
        self.input_id_g_max_len, self.gen_max_len = 0, 0

        self.i_to_update = set()
        self.i_dslen_map = {}

        state = []
        c = 0  # After applying self.shuffle_state, generate_y, slot_meta, and op_ids still correspond to each other
        for i, (s, op) in enumerate(zip(self.slot_meta, self.op_ids)):  # might be shuffled
            state.append(slot_token)
            k = s.split('-')

            # we need this when testing too (w/o ground-truth update_id at that moment)
            self.i_dslen_map[i] = len(tokenizer.tokenize(' '.join(k)))

            # For generator (if using teacher forcing)
            if op == self.update_id:

                # TODO: v2 special
                self.i_to_update.add(i)

                ds_inp = tokenizer.convert_tokens_to_ids(['-'])  # list, domain slot -

                # print("ds_inp, ", k+['-'])
                # print("ds_inp, ", ds_inp)  # ds-
                # print("self.generate_y[c], ",  self.generate_y[c])  # to check: correspond to ds-

                value_gen = self.generate_ids[c]
                # print("value_gen,  ", value_gen)  # value, sep

                self.input_id_g.append(ds_inp + value_gen)
                self.input_id_g_max_len = max(self.input_id_g_max_len, len(self.input_id_g[-1]))
                self.input_mask_rel_pos.append(list(range(len(ds_inp)-1, len(ds_inp)+len(value_gen)-1)))  #  AR objective
                self.lm_label_ids.append(value_gen)
                self.gen_max_len = max(self.gen_max_len, len(value_gen))

                c += 1

            v = self.last_dialog_state.get(s)
            if v is not None:
                k.extend(['-', v])
                t = tokenizer.tokenize(' '.join(k))
            else:
                t = tokenizer.tokenize(' '.join(k))
                t.extend(['-', '[NULL]'])

            state.extend(t)

        self.n_updates = len(self.input_id_g)

        # For predictor
        avail_length_1 = max_seq_length - (len(state) + self.input_id_g_max_len) - 3  # considering predictor & generator

        # TODO: To speed up
        # diag_1 = tokenizer.tokenize(self.dialog_history)
        # diag_2 = tokenizer.tokenize(self.turn_utter)
        diag_1 = deepcopy(self.dialog_history)
        diag_2 = deepcopy(self.turn_utter)

        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncated
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]

        if len(diag_1) == 0 and len(diag_2) > avail_length_1:
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]

        drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
        diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]

        self.diag_1_len = len(diag_1)

        diag_2 = diag_2 + ["[SEP]"]

        diag = diag_1 + diag_2
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]

        self.diag_len = len(diag)

        # FullBert
        self.input_id_p = tokenizer.convert_tokens_to_ids(diag + state)  # For predictor
        self.input_id_p_len = len(self.input_id_p)
        self.segment_id_p = [0] * len(diag_1) + [1] * len(diag_2) + [2] * len(state)
        # position_ids_p: will be automatically created in model

        # self.input_id_g, finished
        self.segment_id_g = [[3]*len(input_id_g) for input_id_g in self.input_id_g]

        # position_ids_g: for position embedding
        # TODO: Here is the bug for my previous codes.
        #  Pos Emb for target side should follow source side instead of re-start.
        self.position_ids_g = []
        for inp_g in self.input_id_g:
            self.position_ids_g.append(list(range(self.input_id_p_len, self.input_id_p_len+len(inp_g))))

        # print("self.input_id_p, ", self.input_id_p)
        # print("self.segment_id_p, ", self.segment_id_p)
        # print('\n')
        # print("self.input_id_g, ", self.input_id_g)
        # print("self.segment_id_g, ", self.segment_id_g)
        # print("self.position_ids_g, ", self.position_ids_g)
        # print('\n')
        # print("self.input_mask_rel_pos, ", self.input_mask_rel_pos)
        # print("self.lm_label_ids, ", self.lm_label_ids)
        # exit()

        # TODO: Leave Padding for the next step (I need to get the max_len of a batch)
        # TODO: Since I do padding later, I can only get these later:
        #   slot_position & input_mask, ...


class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, slot_meta, max_seq_length, rng,
                 ontology, word_dropout=0.1, shuffle_state=False, shuffle_p=0.5,
                 decoder_teacher_forcing=0.5, pad_id=0, slot_id=1,
                 use_full_slot=False, use_dt_only=False, no_dial=False, use_cls_only=False):

        self.use_full_slot = use_full_slot
        self.use_dt_only = use_dt_only
        self.no_dial = no_dial
        self.use_cls_only = use_cls_only

        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.slot_meta = slot_meta
        self.max_seq_length = max_seq_length
        self.ontology = ontology
        self.word_dropout = word_dropout
        self.shuffle_state = shuffle_state
        self.shuffle_p = shuffle_p
        self.rng = rng

        # For teacher forcing
        if decoder_teacher_forcing < 1.0:
            print("### Force to use teacher forcing all the time")
            decoder_teacher_forcing = 1.0

        self.decoder_teacher_forcing = decoder_teacher_forcing
        print("### decoder_teacher_forcing: {:}".format(decoder_teacher_forcing))

        self.pad_id = pad_id
        self.slot_id = slot_id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.shuffle_state and self.shuffle_p > 0.:
            if self.rng.random() < self.shuffle_p:
                self.data[idx].shuffle_state(self.rng, None)
            else:
                self.data[idx].shuffle_state(self.rng, self.slot_meta)
        if self.word_dropout > 0 or self.shuffle_state:
            self.data[idx].make_instance(self.tokenizer,
                                         word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        """
        For generator, two training samples might have different n_updates. Thus, we stack tensors in order.
        """
        return wrap_into_tensor(batch, pad_id=self.pad_id, slot_id=self.slot_id, use_teacher=True,
                                use_full_slot=self.use_full_slot, use_dt_only=self.use_dt_only, no_dial=self.no_dial,
                                use_cls_only=self.use_cls_only)


def do_pad(x, batch_max_len, pad_val):
    if isinstance(x, list):
        assert not isinstance(x[0], list)
        n_pad = batch_max_len - len(x)
        if n_pad > 0:
            return x + [pad_val] * n_pad
        else:
            assert n_pad == 0
            return x
    else:
        raise NotImplementedError


def get_bi_attn_mask(input_id_p_len, input_id_p_max_len):
    return [1]*input_id_p_len + [0]*(input_id_p_max_len-input_id_p_len)


def get_seq_attn_mask(inp_p_len, input_id_g, max_p_len, max_g_len, slot_to_update, diag_len,
                      use_full_slot=False,
                      diag_1_len=0, use_dt_only=False, no_dial=False, use_cls_only=False):
    assert isinstance(input_id_g, list)
    assert isinstance(slot_to_update, list)  # pos

    # print("inp_p_len, ", inp_p_len)
    # print("max_p_len, ", max_p_len)
    # print("max_g_len, ", max_g_len)

    max_len = max_p_len+max_g_len

    # tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))
    tril_matrix = np.tril(np.ones((max_len, max_len), dtype=np.long))

    res = []
    for inp_g, to_update in zip(input_id_g, slot_to_update):
        assert isinstance(to_update, list)
        if not use_full_slot:
            assert len(to_update) == 1

        attn_mask = deepcopy(tril_matrix[-max_g_len:, :])

        #  TODO: v2 special
        # attn_mask[:, inp_p_len:max_p_len].fill_(0)

        if no_dial:
            attn_mask[:, :max_p_len].fill(0)
            if use_cls_only:
                attn_mask[:, 0].fill(1)

        else:
            attn_mask[:, diag_len:max_p_len].fill(0)
            if use_dt_only:
                attn_mask[:, :diag_1_len].fill(0)

        attn_mask[:, to_update[0]:to_update[-1]+1].fill(1)

        # print("diag_len, ", diag_len)
        # print("max_p_len, ", max_p_len)
        # print("to_update, ", to_update)

        attn_mask[:, max_p_len+len(inp_g):].fill(0)

        # print(attn_mask.tolist())

        res.append(attn_mask)

    #     print("attn_mask")
    #     print(attn_mask.tolist())
    #     print('\n')
    #
    # print("-"*20)

    return res


def wrap_into_tensor(batch, pad_id=0, slot_id=1, use_teacher=True, use_full_slot=False, use_dt_only=False, no_dial=False, use_cls_only=False):
    """
    For generator, two training samples might have different n_updates. Thus, we stack tensors in order.

    When evaluating, do not use input_ids_g,  ...
        I have prepared these inputs in the model module.
    """
    assert use_teacher is True

    batch_size = len(batch)

    # Get max length
    input_id_p_max_len = max([f.input_id_p_len for f in batch])
    input_id_g_max_len = max([f.input_id_g_max_len for f in batch])
    gen_max_len = max([f.gen_max_len for f in batch])

    # TODO: Predictor
    input_ids_p = torch.tensor([do_pad(f.input_id_p, input_id_p_max_len, pad_id) for f in batch], dtype=torch.long)
    segment_ids_p = torch.tensor([do_pad(f.segment_id_p, input_id_p_max_len, 0) for f in batch], dtype=torch.long)
    input_mask_p = torch.tensor([get_bi_attn_mask(f.input_id_p_len, input_id_p_max_len) for f in batch], dtype=torch.long)

    op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)
    domain_ids = torch.tensor([f.domain_id for f in batch], dtype=torch.long)

    slot_position = []
    slot_to_update = []  # TODO: v2 special, a nested list

    for i in range(batch_size):
        pos = []
        for j in range(input_ids_p.size(1)):
            if input_ids_p[i, j] == slot_id:
                pos.append(j)

        slot_position.append(pos)

        to_update = []  # it is a nested list (each list for each slot to update)
        for c, p in enumerate(pos):
            if c in batch[i].i_to_update:
                if use_full_slot:
                    try:
                        to_update.append(list(range(p, p+batch[i].i_dslen_map[c]+1)))
                    except IndexError:
                        raise IndexError("Here should not encounter IndexError. ")

                else:
                    to_update.append([p])

        slot_to_update.append(to_update)

        # print("i, ", i)
        # print("pos, ", pos)
        # print("to_update, ", to_update)

    state_position_ids = torch.tensor(slot_position, dtype=torch.long)

    # TODO: Generator
    input_ids_g, segment_ids_g, position_ids_g = [], [], []
    id_n_map = {}
    for i, f in enumerate(batch):
        i_input_ids_g, i_segment_ids_g, i_position_ids_g = [], [], []

        if f.n_updates > 0:
            id_n_map[i] = f.n_updates

            for inp, seg, pos in zip(f.input_id_g, f.segment_id_g, f.position_ids_g):
                i_input_ids_g.append(do_pad(inp, input_id_g_max_len, pad_id))
                i_segment_ids_g.append(do_pad(seg, input_id_g_max_len, 0))
                i_position_ids_g.append(do_pad(pos, input_id_g_max_len, 0))

        input_ids_g.append(i_input_ids_g)
        segment_ids_g.append(i_segment_ids_g)
        position_ids_g.append(i_position_ids_g)

    # input_ids_g = torch.tensor(input_ids_g, dtype=torch.long)
    # segment_ids_g = torch.tensor(segment_ids_g, dtype=torch.long)
    # position_ids_g = torch.tensor(position_ids_g, dtype=torch.long)

    # list (batch) of list (n_updates) of attn tensor
    input_mask_g = [get_seq_attn_mask(f.input_id_p_len, f.input_id_g, input_id_p_max_len, input_id_g_max_len,
                                              slot_to_update[i], f.diag_len,
                                      use_full_slot=use_full_slot,
                                      diag_1_len=f.diag_1_len, use_dt_only=use_dt_only,
                                      no_dial=no_dial, use_cls_only=use_cls_only) for i, f in enumerate(batch)]

    # input_mask_g = flatten(input_mask_g)
    # if len(input_mask_g) > 0:
    #     input_mask_g = torch.stack(input_mask_g, dim=0)
    # else:
    #     input_mask_g = torch.tensor([], dtype=torch.long)

    # Generation  Label
    masked_pos, masked_weights = [], []
    n_total_pred = 0
    for f in batch:
        i_masked_pos, i_masked_weights = [], []
        for mask_rel_pos in f.input_mask_rel_pos:  # if n_updates > 0
            n_pred = len(mask_rel_pos)
            n_total_pred += n_pred
            i_masked_pos.append(do_pad(mask_rel_pos, gen_max_len, 0))
            i_masked_weights.append([1]*n_pred+[0]*(gen_max_len-n_pred))

        masked_pos.append(i_masked_pos)
        masked_weights.append(i_masked_weights)

    # masked_pos = torch.tensor(masked_pos, dtype=torch.long)
    # masked_weights = torch.tensor(masked_weights, dtype=torch.float)

    lm_label_ids = []
    for f in batch:
        i_lm_label_ids = []
        for v in f.lm_label_ids:  # if n_updates > 0
            i_lm_label_ids.append(do_pad(v, gen_max_len, 0))

        lm_label_ids.append(i_lm_label_ids)

    # lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)

    return input_ids_p, segment_ids_p, input_mask_p, \
           state_position_ids, op_ids, domain_ids, input_ids_g, segment_ids_g, position_ids_g, input_mask_g, \
           masked_pos, masked_weights, lm_label_ids, id_n_map, gen_max_len, n_total_pred
