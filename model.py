import math
import numpy as np
import torch
import torch.nn as nn
from modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, BertPreTrainingHeads
from copy import deepcopy


class TransformerDST(BertPreTrainedModel):
    def __init__(self, config, dec_config, n_op, n_domain,
                 update_id, mask_word_id, eos_id, pad_id, val_sep_id,
                 type_vocab_size, exclude_domain=False):
        super(TransformerDST, self).__init__(config)

        self.val_sep_id = val_sep_id  # TODO: v2 special
        print("### word index of '-', ", self.val_sep_id)

        self.hidden_size = config.hidden_size
        self.n_op = n_op
        self.update_id = update_id
        self.mask_word_id = mask_word_id

        self.bert = BertModel(config, type_vocab_size)

        # predictor
        self.encoder = Encoder(config, self.bert, n_op, n_domain, update_id, exclude_domain)

        self.decoder = BertForSeq2SeqDecoder(config, dec_config, self.bert, self.bert.embeddings.word_embeddings.weight, mask_word_id, eos_id, pad_id)

        self.apply(self.init_weights)

    def expand(self, x, id_n_map):
        # TODO: not all idx in the batch have a key in id_n_map
        if isinstance(x, list) or isinstance(x, tuple):  # list of tensor
            results = []
            for t in x:
                tmp = []
                for i, v in enumerate(torch.split(t, 1, dim=0)):  # keep dims
                    try:
                        tmp.append(v.expand(id_n_map[i], -1, -1))
                    except KeyError:
                        pass

                results.append(torch.cat(tmp, dim=0))

            return results

        else:  # tensor (only)
            results = []
            for i, v in enumerate(torch.split(x, 1, dim=0)):  # keep dims
                try:
                    results.append(v.expand(id_n_map[i], -1, -1))
                except KeyError:
                    pass

            return torch.cat(results, dim=0)

    def wrap_into_tensor(self, input_ids_g, segment_ids_g, position_ids_g, input_mask_g,
                         masked_pos, masked_weights, lm_label_ids, device):

        flatten = lambda x: [i for s in x for i in s]

        input_ids_g = torch.tensor(flatten(input_ids_g), dtype=torch.long, device=device)
        segment_ids_g = torch.tensor(flatten(segment_ids_g), dtype=torch.long, device=device)
        position_ids_g = torch.tensor(flatten(position_ids_g), dtype=torch.long, device=device)

        input_mask_g = flatten(input_mask_g)
        input_mask_g = [torch.tensor(v).to(device) for v in input_mask_g]
        if len(input_mask_g) > 0:
            input_mask_g = torch.stack(input_mask_g, dim=0)
        else:
            input_mask_g = torch.tensor([], dtype=torch.long)

        masked_pos = torch.tensor(flatten(masked_pos), dtype=torch.long, device=device)
        masked_weights = torch.tensor(flatten(masked_weights), dtype=torch.float, device=device)
        lm_label_ids = torch.tensor(flatten(lm_label_ids), dtype=torch.long, device=device)

        return input_ids_g, segment_ids_g, position_ids_g, input_mask_g, \
               masked_pos, masked_weights, lm_label_ids

    def forward(self, input_ids_p, segment_ids_p, input_mask_p, state_position_ids,
                input_ids_g_, segment_ids_g_, position_ids_g_, input_mask_g_,
                masked_pos_, masked_weights_, lm_label_ids_, id_n_map_, gen_max_len, only_pred_op=False, n_gpu=0):
        """
        :param input_ids_p: (batch, n1)
        :param input_ids_g: (batch, n3)
        :param input_mask_p: (batch, n1)
        :param input_mask_g: (batch, n1+n3)
        :param segment_ids_p: (batch, n1)
        :param segment_ids_g: (batch, n3)
        :param state_position_ids: x
        :param op_ids: x
        :param lm_label_ids: (batch*avg_n_updates, n4)
        """
        if n_gpu > 2:
            raise NotImplementedError

        device = input_ids_p.device

        # TODO: Input Preparation For Two GPUs
        if n_gpu == 2:  # id_n_map is for parallel training. nothing about model
            batch_size = input_ids_p.size(0)
            id_n_map = {}

            target_id = []  # could be a empty list
            if device == torch.device('cuda:0'):
                for id, n in id_n_map_.items():
                    if id < batch_size:
                        id_n_map[id] = n
                        target_id.append(id)
            else:
                for id, n in id_n_map_.items():
                    if id >= batch_size:
                        id_n_map[id-batch_size] = n
                        target_id.append(id)

            input_ids_g, segment_ids_g, position_ids_g, input_mask_g, masked_pos, masked_weights, lm_label_ids = [], [], [], [], [], [], []
            for id in target_id:
                input_ids_g.append(input_ids_g_[id])
                segment_ids_g.append(segment_ids_g_[id])
                position_ids_g.append(position_ids_g_[id])
                input_mask_g.append(input_mask_g_[id])
                masked_pos.append(masked_pos_[id])
                masked_weights.append(masked_weights_[id])
                lm_label_ids.append(lm_label_ids_[id])

            input_ids_g, segment_ids_g, position_ids_g, input_mask_g, \
            masked_pos, masked_weights, lm_label_ids = self.wrap_into_tensor(input_ids_g, segment_ids_g, position_ids_g, input_mask_g,
                                                                             masked_pos, masked_weights, lm_label_ids, device)

        else:
            input_ids_g, segment_ids_g, position_ids_g, input_mask_g, \
            masked_pos, masked_weights, lm_label_ids = self.wrap_into_tensor(input_ids_g_, segment_ids_g_, position_ids_g_, input_mask_g_,
                         masked_pos_, masked_weights_, lm_label_ids_, device)

            id_n_map = id_n_map_

        # TODO: Encoder
        enc_outputs = self.encoder(input_ids=input_ids_p,
                                   token_type_ids=segment_ids_p,
                                   state_positions=state_position_ids,
                                   attention_mask=input_mask_p)

        domain_scores, state_scores, embedding_output, all_hidden_states = enc_outputs

        if (not only_pred_op) and len(id_n_map) > 0:
            # embedding_output: (batch, n1, 768)
            # all_hidden_states: list  of (batch, n1, 768)
            loss_g = self.decoder(input_ids=input_ids_g, token_type_ids=segment_ids_g, position_ids=position_ids_g, attention_mask=input_mask_g,
                                  masked_pos=masked_pos, masked_weights=masked_weights, masked_lm_labels=lm_label_ids,
                                  prev_embedding=self.expand(embedding_output, id_n_map), prev_encoded_layers=self.expand(all_hidden_states, id_n_map))

        else:
            loss_g = torch.zeros((1, gen_max_len), dtype=torch.float, device=device)

        return domain_scores, state_scores, loss_g

    def output(self, input_ids_p, segment_ids_p, input_mask_p,
               state_position_ids, diag_len, op_ids=None, gen_max_len=9, use_full_slot=False, use_dt_only=False,
               diag_1_len=0, no_dial=False, use_cls_only=False, i_dslen_map=None):
        """
        Evaluation

        Tensor (batch, max_len), here batch should be 1 * n_updates to match evaluation.py
        however, self.encoder and self.decoder.generate themselves support batch iteration.

        """

        assert isinstance(i_dslen_map, dict)

        if input_ids_p.size(0) > 1:
            raise NotImplementedError("The code doesn't support a batch of inputs.")

        device = input_ids_p.device

        enc_outputs = self.encoder(input_ids=input_ids_p,
                                   token_type_ids=segment_ids_p,
                                   state_positions=state_position_ids,
                                   attention_mask=input_mask_p)

        domain_scores, state_scores, embedding_output, all_hidden_states = enc_outputs
        inp_p_len = embedding_output.size(1)
        #  print("embedding_output, ", embedding_output.shape)

        # TODO: Generation
        if op_ids is None:  # do not use ground-truth
            op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(-1).tolist()
        else:
            op_ids = op_ids.view(-1).tolist()

        state_position_ids = state_position_ids.view(-1).tolist()

        id_n_map = {0: 0}  # only a sample
        slot_to_update = []
        for i_ds, op in enumerate(op_ids):
            if op == self.update_id:
                id_n_map[0] += 1

                if use_full_slot:
                    p = state_position_ids[i_ds]
                    try:
                        slot_to_update.append(list(range(p, p+i_dslen_map[i_ds]+1)))
                    except IndexError:
                        raise IndexError("Here should not encounter IndexError. ")

                else:
                    slot_to_update.append([state_position_ids[i_ds]])

        if id_n_map[0] > 0:
            input_ids_g, segment_ids_g, position_ids_g, input_mask_g = [], [], [], []  # nested list

            max_g_len = 1 + gen_max_len
            max_len = inp_p_len + max_g_len

            tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))

            for to_update in slot_to_update:
                input_ids_g.append([self.val_sep_id])
                segment_ids_g.append([3] + [3] * gen_max_len)

                n_ds = 1
                position_ids_g.append(list(range(inp_p_len, inp_p_len + n_ds)) +
                                      list(range(inp_p_len + n_ds, inp_p_len + n_ds + gen_max_len)))  # checked

                # Get attention mask
                attn_mask = deepcopy(tril_matrix[-max_g_len:, :])

                if no_dial:
                    attn_mask[:, :inp_p_len].fill_(0)
                    if use_cls_only:
                        attn_mask[:, 0].fill_(1)

                else:
                    attn_mask[:, diag_len:inp_p_len].fill_(0)
                    if use_dt_only:
                        attn_mask[:, :diag_1_len].fill_(0)

                attn_mask[:, to_update[0]:to_update[-1]+1].fill_(1)

                input_mask_g.append(attn_mask)

            input_ids_g = torch.tensor(input_ids_g, dtype=torch.long).to(device)
            segment_ids_g = torch.tensor(segment_ids_g, dtype=torch.long).to(device)
            position_ids_g = torch.tensor(position_ids_g, dtype=torch.long).to(device)

            input_mask_g = torch.stack(input_mask_g, dim=0).to(device)

            output_ids = self.decoder.generate(input_ids=input_ids_g, token_type_ids=segment_ids_g, position_ids=position_ids_g,
                                               attention_mask=input_mask_g, max_ds_len=1,
                                               prev_embedding=self.expand(embedding_output, id_n_map),
                                               prev_encoded_layers=self.expand(all_hidden_states, id_n_map))

            try:
                output_ids = output_ids.tolist()
            except AttributeError:
                assert isinstance(output_ids, list)

        else:
            output_ids = []

        return domain_scores, state_scores, output_ids


class Encoder(nn.Module):
    def __init__(self, config, bert, n_op, n_domain, update_id, exclude_domain=False):
        super(Encoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.exclude_domain = exclude_domain
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id

        assert isinstance(bert, BertModel)
        self.bert_model = bert

        self.dropout = nn.Dropout(config.dropout)
        self.action_cls = nn.Linear(config.hidden_size, n_op)
        if self.exclude_domain is not True:
            self.domain_cls = nn.Linear(config.hidden_size, n_domain)

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask):

        sequence_output, pooled_output, \
        embedding_output, all_hidden_states = self.bert_model(input_ids, token_type_ids, attention_mask)

        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)
        state_scores = self.action_cls(self.dropout(state_output))  # B,J,4

        if self.exclude_domain:
            domain_scores = torch.zeros(1, device=input_ids.device)  # dummy
        else:
            domain_scores = self.domain_cls(self.dropout(pooled_output))

        return domain_scores, state_scores, embedding_output, all_hidden_states


class BertForSeq2SeqDecoder(nn.Module):
    """refer to BertForPreTraining"""

    def __init__(self, config, dec_config, bert, bert_model_embedding_weights, mask_word_id, eos_id, pad_id):
        super(BertForSeq2SeqDecoder, self).__init__()
        assert isinstance(bert, BertModel)
        self.bert_model = bert

        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

        # For training
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')

        # For evaluation
        self.mask_word_id = mask_word_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.search_beam_size = dec_config.beam_size
        self.length_penalty = dec_config.length_penalty
        self.forbid_duplicate_ngrams = dec_config.forbid_duplicate_ngrams
        self.forbid_ignore_set = None
        self.ngram_size = dec_config.ngram_size
        self.min_len = dec_config.min_len
        self.mode = "s2s"
        self.pos_shift = False
        self.not_predict_set = None

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, masked_pos, masked_weights, masked_lm_labels, prev_embedding, prev_encoded_layers):
        """
        Applied in training process
        """
        def gather_seq_out_by_pos(seq, pos):
            try:
                return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
            except RuntimeError:
                print("RuntimeError: gather_seq_out_by_pos")
                print(seq.shape)
                print(pos.shape)
                print(pos)
                exit()

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask

            # denominator = torch.sum(mask) + 1e-5
            # node = loss / denominator

            # return node.sum()
            return loss

        sequence_output, _, _, _ = self.bert_model(
            input_ids, token_type_ids, attention_mask, position_ids=position_ids, prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers)

        # TODO: Get Loss
        sequence_output_masked = gather_seq_out_by_pos(
            sequence_output, masked_pos)

        prediction_scores = self.predictions(sequence_output_masked)
        masked_lm_loss = self.crit_mask_lm(
            prediction_scores.transpose(1, 2).float(), masked_lm_labels)

        masked_lm_loss = loss_mask_and_normalize(
            masked_lm_loss.float(), masked_weights)

        return masked_lm_loss

    def generate(self, input_ids, token_type_ids, position_ids, attention_mask, max_ds_len,
                 prev_embedding, prev_encoded_layers):

        self.pos_shift = True

        if self.search_beam_size > 1:
            return self.beam_search(input_ids, token_type_ids, position_ids, attention_mask, max_ds_len,
                                    prev_embedding, prev_encoded_layers)

        batch_size = token_type_ids.size(0)
        output_length = token_type_ids.size(1) - input_ids.size(1)
        input_length = prev_embedding.size(1)
        device = prev_embedding.device

        output_ids = []
        curr_ids = None

        step = -1
        while step+1 < output_length:
            step += 1

            if curr_ids is None:  # first step
                # x_input_ids = sos_ids
                x_input_ids = input_ids
                cur_token_type_ids = token_type_ids[:, :max_ds_len]
                cur_position_ids = position_ids[:, :max_ds_len]
                cur_attention_mask = attention_mask[:, :max_ds_len, :input_length+max_ds_len]

            else:  # step >= 1
                x_input_ids = curr_ids
                cur_token_type_ids = token_type_ids[:, max_ds_len+step-1:max_ds_len+step]
                cur_position_ids = position_ids[:, max_ds_len+step-1:max_ds_len+step]
                cur_attention_mask = attention_mask[:, max_ds_len+step-1:max_ds_len+step,
                                      :input_length+max_ds_len+step]

            _, _, new_embedding, new_encoded_layers = \
                self.bert_model(x_input_ids, cur_token_type_ids, cur_attention_mask, position_ids=cur_position_ids,
                          prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores = self.predictions(last_hidden)

            if self.min_len and (step + 1 <= self.min_len):
                prediction_scores[:, :, self.eos_id].fill_(-10000.0)

            if self.not_predict_set:
                for token_id in self.not_predict_set:
                    prediction_scores[:, :, token_id].fill_(-10000.0)

            _, max_ids = torch.max(prediction_scores, dim=-1)
            output_ids.append(max_ids)

            if prev_embedding is None:
                prev_embedding = new_embedding
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding), dim=1)

            if prev_encoded_layers is None:
                prev_encoded_layers = [x for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                    prev_encoded_layers, new_encoded_layers)]

            curr_ids = max_ids

        output_ids = torch.cat(output_ids, dim=1)  # Tensor (batch, max_len)

        outputs = []
        for w_ids in output_ids.tolist():
            output_ = []
            for w in w_ids:
                if w in (self.eos_id, self.pad_id):
                    break
                output_.append(w)

            # print("output_, ", output_)
            outputs.append(output_)

        return outputs

    def beam_search(self, input_ids, token_type_ids, position_ids, attention_mask, max_ds_len,
                    prev_embedding, prev_encoded_layers):

        self.pos_shift = True

        batch_size = token_type_ids.size(0)
        output_length = token_type_ids.size(1) - input_ids.size(1)
        input_length = prev_embedding.size(1)
        device = prev_embedding.device

        curr_ids = None
        # sos_ids = torch.zeros((batch_size, 1), dtype=torch.long).fill_(self.sos_id).to(device)

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        step = -1
        while step+1 < output_length:

            step += 1

            if curr_ids is None:  # first step
                # x_input_ids = sos_ids
                x_input_ids = input_ids
                cur_token_type_ids = token_type_ids[:, :max_ds_len]
                cur_position_ids = position_ids[:, :max_ds_len]
                cur_attention_mask = attention_mask[:, :max_ds_len, :input_length+max_ds_len]

            else:  # step >= 1
                x_input_ids = curr_ids
                cur_token_type_ids = token_type_ids[:, max_ds_len+step-1:max_ds_len+step]
                cur_position_ids = position_ids[:, max_ds_len+step-1:max_ds_len+step]
                cur_attention_mask = attention_mask[:, max_ds_len+step-1:max_ds_len+step,
                                      :input_length+max_ds_len+step]

            _, _, new_embedding, new_encoded_layers = \
                self.bert_model(x_input_ids, cur_token_type_ids, cur_attention_mask, position_ids=cur_position_ids,
                          prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores = self.predictions(last_hidden)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)

            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (step+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            if self.not_predict_set:
                for token_id in self.not_predict_set:
                    log_scores[:, :, token_id].fill_(-10000.0)

            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            # print("step, ", step)
            # print("log_scores, ",  log_scores.shape)
            # print("kk_scores, ", kk_scores.shape)

            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)

            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float())
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (step == 0)

            if self.pos_shift:
                if step == 0:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                    prev_embedding = first_expand(prev_embedding)
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if step == 0:
                    prev_encoded_layers = [first_expand(torch.cat((x[0], x[1]), dim=1)) for x in zip(prev_encoded_layers, new_encoded_layers)]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                        prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]
            else:
                raise NotImplementedError

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                position_ids = first_expand(position_ids)
                token_type_ids = first_expand(token_type_ids)
                attention_mask = first_expand(attention_mask)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, padding_value=0).to(device)

        traces = {k: v.tolist() for k, v in traces.items()}
        output_ids = []
        for w_ids in traces['pred_seq']:
            output_ = []
            for w in w_ids:
                if w in (self.eos_id, self.pad_id):
                    break
                output_.append(w)

            output_ids.append(output_)

        return output_ids

