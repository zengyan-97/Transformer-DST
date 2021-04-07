import sys
from model import TransformerDST
from pytorch_transformers import BertTokenizer, AdamW, WarmupLinearSchedule, BertConfig
from utils.data_utils import prepare_dataset, MultiWozDataset
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
from utils.ckpt_utils import download_ckpt, convert_ckpt_compatible
from evaluation import model_evaluation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import argparse
import random
import os
import json
import time


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


def save(args, epoch, model, enc_optimizer, dec_optimizer=None):
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    model_file = os.path.join(
        args.save_dir, "model.e{:}.bin".format(epoch))
    torch.save(model_to_save.state_dict(), model_file)

    # enc_optim_file = os.path.join(
    #     args.save_dir, "enc_optim.e{:}.bin".format(epoch))
    # torch.save(enc_optimizer.state_dict(), enc_optim_file)
    #
    # if dec_optimizer is not None:
    #     dec_optim_file = os.path.join(
    #         args.save_dir, "dec_optim.e{:}.bin".format(epoch))
    #     torch.save(dec_optimizer.state_dict(), dec_optim_file)


def load(args, epoch):
    model_file = os.path.join(
        args.save_dir, "model.e{:}.bin".format(epoch))
    model_recover = torch.load(model_file, map_location='cpu')

    enc_optim_file = os.path.join(
        args.save_dir, "enc_optim.e{:}.bin".format(epoch))
    enc_recover = torch.load(enc_optim_file, map_location='cpu')
    if hasattr(enc_recover, 'state_dict'):
        enc_recover = enc_recover.state_dict()

    dec_optim_file = os.path.join(
        args.save_dir, "dec_optim.e{:}.bin".format(epoch))
    dec_recover = torch.load(dec_optim_file, map_location='cpu')
    if hasattr(dec_recover, 'state_dict'):
        dec_recover = dec_recover.state_dict()

    return model_recover, enc_recover, dec_recover


def main(args):

    assert args.use_one_optim is True

    if args.use_cls_only:
        args.no_dial = True

    print("### use_cls_only: {:}".format(args.use_cls_only))
    print("### no_dial: {:}".format(args.no_dial))

    if args.recover_e > 0:
        raise NotImplementedError("This option is from my oldest code version. "
                                  "I have not checked it for this code version.")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        print("### mkdir {:}".format(args.save_dir))

    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    n_gpu = 0
    if torch.cuda.is_available() and (not args.use_cpu):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda')
        print("### Device: {:}".format(device))
    else:
        print("### Use CPU (Debugging)")
        device = torch.device("cpu")

    if args.random_seed < 0:
        print("### Pick a random seed")
        args.random_seed = random.sample(list(range(0, 100000)), 1)[0]

    print("### Random Seed: {:}".format(args.random_seed))
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)

    if n_gpu > 0:
        if args.random_seed >= 0:
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    ontology = json.load(open(args.ontology_data))
    slot_meta, ontology = make_slot_meta(ontology)
    op2id = OP_SET[args.op_code]
    print(op2id)

    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    train_path = os.path.join(args.data_root, "train.pt")
    dev_path = os.path.join(args.data_root, "dev.pt")
    test_path = os.path.join(args.data_root, "test.pt")

    if not os.path.exists(test_path):
        test_data_raw = prepare_dataset(data_path=args.test_data_path,
                                        tokenizer=tokenizer,
                                        slot_meta=slot_meta,
                                        n_history=args.n_history,
                                        max_seq_length=args.max_seq_length,
                                        op_code=args.op_code)
        torch.save(test_data_raw, test_path)
    else:
        test_data_raw = torch.load(test_path)

    print("# test examples %d" % len(test_data_raw))

    if not os.path.exists(train_path):
        train_data_raw = prepare_dataset(data_path=args.train_data_path,
                                         tokenizer=tokenizer,
                                         slot_meta=slot_meta,
                                         n_history=args.n_history,
                                         max_seq_length=args.max_seq_length,
                                         op_code=args.op_code)

        torch.save(train_data_raw, train_path)
    else:
        train_data_raw = torch.load(train_path)

    train_data = MultiWozDataset(train_data_raw,
                                 tokenizer,
                                 slot_meta,
                                 args.max_seq_length,
                                 rng,
                                 ontology,
                                 args.word_dropout,
                                 args.shuffle_state,
                                 args.shuffle_p, pad_id=tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
                                 slot_id=tokenizer.convert_tokens_to_ids(['[SLOT]'])[0],
                                 decoder_teacher_forcing=args.decoder_teacher_forcing,
                                 use_full_slot=args.use_full_slot,
                                 use_dt_only=args.use_dt_only, no_dial=args.no_dial,
                                 use_cls_only=args.use_cls_only)

    print("# train examples %d" % len(train_data_raw))

    if not os.path.exists(dev_path):
        dev_data_raw = prepare_dataset(data_path=args.dev_data_path,
                                       tokenizer=tokenizer,
                                       slot_meta=slot_meta,
                                       n_history=args.n_history,
                                       max_seq_length=args.max_seq_length,
                                       op_code=args.op_code)
        torch.save(dev_data_raw,  dev_path)
    else:
        dev_data_raw = torch.load(dev_path)

    print("# dev examples %d" % len(dev_data_raw))

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model_config.hidden_dropout_prob = args.hidden_dropout_prob

    type_vocab_size = 4
    dec_config = args
    model = TransformerDST(model_config, dec_config, len(op2id), len(domain2id),
                           op2id['update'],
                           tokenizer.convert_tokens_to_ids(['[MASK]'])[0],
                           tokenizer.convert_tokens_to_ids(['[SEP]'])[0],
                           tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
                           tokenizer.convert_tokens_to_ids(['-'])[0],
                           type_vocab_size, args.exclude_domain)

    if not os.path.exists(args.bert_ckpt_path):
        args.bert_ckpt_path = download_ckpt(args.bert_ckpt_path, args.bert_config_path, 'assets')

    state_dict = torch.load(args.bert_ckpt_path, map_location='cpu')
    _k = 'embeddings.token_type_embeddings.weight'
    print("config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(
            type_vocab_size, state_dict[_k].shape[0]))
    state_dict[_k].resize_(
        type_vocab_size, state_dict[_k].shape[1])
    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
    state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
    model.bert.load_state_dict(state_dict)
    print("\n### Done Load BERT")
    sys.stdout.flush()

    # re-initialize added special tokens ([SLOT], [NULL], [EOS])
    model.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
    model.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    model.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)

    # re-initialize seg-2, seg-3
    model.bert.embeddings.token_type_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    model.bert.embeddings.token_type_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
    model.to(device)

    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)

    if args.use_one_optim:
        print("### Use One Optim")
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.enc_lr)
        scheduler = WarmupLinearSchedule(optimizer, int(num_train_steps * args.enc_warmup),
                                             t_total=num_train_steps)
    else:
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        enc_param_optimizer = list(model.bert.named_parameters())  # TODO: For BERT only
        print('### Optim BERT: {:}'.format(len(enc_param_optimizer)))
        enc_optimizer_grouped_parameters = [
            {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.enc_lr)
        enc_scheduler = WarmupLinearSchedule(enc_optimizer, int(num_train_steps * args.enc_warmup),
                                             t_total=num_train_steps)

        dec_param_optimizer = list(model.named_parameters())  # TODO:  For other parameters
        print('### Optim All: {:}'.format(len(dec_param_optimizer)))
        dec_param_optimizer = [p for (n, p) in dec_param_optimizer if 'bert' not in n]
        print('### Optim OTH: {:}'.format(len(dec_param_optimizer)))
        dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr)
        dec_scheduler = WarmupLinearSchedule(dec_optimizer, int(num_train_steps * args.dec_warmup),
                                             t_total=num_train_steps)

    if args.recover_e > 0:
        model_recover, enc_recover, dec_recover = load(args, str(args.recover_e))
        print("### Recover Model E{:}".format(args.recover_e))
        sys.stdout.flush()
        model.load_state_dict(model_recover)
        print("### Recover Optim E{:}".format(args.recover_e))
        sys.stdout.flush()
        enc_optimizer.load_state_dict(enc_recover)
        dec_optimizer.load_state_dict(dec_optimizer)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)

    loss_fnc = nn.CrossEntropyLoss()
    best_score = {'epoch': 0, 'joint_acc': 0, 'op_acc': 0, 'final_slot_f1': 0}

    start_time = time.time()

    for epoch in range(args.n_epochs):
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):

            batch = [b.to(device) if (not isinstance(b, int)) and (not isinstance(b, dict) and (not isinstance(b, list)) and (not isinstance(b, np.ndarray))) else b for b in batch]

            input_ids_p, segment_ids_p, input_mask_p, \
            state_position_ids, op_ids, domain_ids, input_ids_g, segment_ids_g, position_ids_g, input_mask_g, \
            masked_pos, masked_weights, lm_label_ids, id_n_map, gen_max_len, n_total_pred = batch

            domain_scores, state_scores, loss_g = model(input_ids_p, segment_ids_p, input_mask_p, state_position_ids,
                input_ids_g, segment_ids_g, position_ids_g, input_mask_g,
                masked_pos, masked_weights, lm_label_ids, id_n_map, gen_max_len, only_pred_op=args.only_pred_op, n_gpu=n_gpu)

            if n_total_pred > 0:
                loss_g = loss_g.sum() / n_total_pred
            else:
                loss_g = 0

            loss_s = loss_fnc(state_scores.view(-1, len(op2id)), op_ids.view(-1))

            if args.only_pred_op:
                loss = loss_s
            else:
                loss = loss_s + loss_g

            if args.exclude_domain is not True:
                loss_d = loss_fnc(domain_scores.view(-1, len(domain2id)), domain_ids.view(-1))
                loss = loss + loss_d

            batch_loss.append(loss.item())

            loss.backward()

            if args.use_one_optim:
                optimizer.step()
                scheduler.step()
            else:
                enc_optimizer.step()
                enc_scheduler.step()
                dec_optimizer.step()
                dec_scheduler.step()

            model.zero_grad()

            if step < 1:
                print("JUST DEBUG")
                try:
                    eval_res = model_evaluation(model, dev_data_raw[:10], tokenizer, slot_meta, epoch + 1, args.op_code,
                                                use_full_slot=args.use_full_slot, use_dt_only=args.use_dt_only, no_dial=args.no_dial, use_cls_only=args.use_cls_only, n_gpu=n_gpu)
                except ZeroDivisionError:
                    pass
                print("DONE")
                model.train()

            if step % 100 == 0:
                try:
                    loss_g = loss_g.item()
                except AttributeError:
                    loss_g = loss_g

                if args.exclude_domain is not True:
                    print("time %.1f min, [%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f, dom_loss : %.3f" \
                          % ((time.time()-start_time)/60, epoch+1, args.n_epochs, step,
                             len(train_dataloader), np.mean(batch_loss),
                             loss_s.item(), loss_g, loss_d.item()))
                else:
                    print("time %.1f min, [%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f" \
                          % ((time.time()-start_time)/60, epoch+1, args.n_epochs, step,
                             len(train_dataloader), np.mean(batch_loss),
                             loss_s.item(), loss_g))

                sys.stdout.flush()
                batch_loss = []

        if args.use_one_optim:
            save(args, epoch + 1, model, optimizer)
        else:
            save(args, epoch + 1, model, enc_optimizer, dec_optimizer)

        if ((epoch+1) % args.eval_epoch == 0) and (epoch+1 >= 8):
            eval_res = model_evaluation(model, dev_data_raw, tokenizer, slot_meta, epoch+1, args.op_code,
                                        use_full_slot=args.use_full_slot, use_dt_only=args.use_dt_only, no_dial=args.no_dial, use_cls_only=args.use_cls_only, n_gpu=n_gpu)
            print("### Epoch {:} Score : ".format(epoch+1), eval_res)

            if eval_res['joint_acc'] > best_score['joint_acc']:
                best_score = eval_res
                print("### Best Joint Acc: {:} ###".format(best_score['joint_acc']))
                print('\n')

                if epoch+1 >= 8:  # To speed up
                    eval_res_test = model_evaluation(model, test_data_raw, tokenizer, slot_meta, epoch + 1, args.op_code,
                                                     use_full_slot=args.use_full_slot, use_dt_only=args.use_dt_only, no_dial=args.no_dial, use_cls_only=args.use_cls_only, n_gpu=n_gpu)
                    print("### Epoch {:} Test Score : ".format(epoch + 1), eval_res_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_cpu", action='store_true')  # Just for my debugging. I have not tested whether it can be used for training model.

    # Using only [CLS]
    parser.add_argument("--use_cls_only", action='store_true')

    # w/o re-using dialogue
    parser.add_argument("--no_dial", action='store_true')

    # Using only D_t in generation
    parser.add_argument("--use_dt_only", action='store_true')

    # By default, "decoder" only attend on a specific [SLOT] position.
    # If using this option, the "decoder" can access to this group of "[SLOT] domain slot - value".
    # NEW: exclude "- value"
    parser.add_argument("--use_full_slot", action='store_true')

    parser.add_argument("--only_pred_op", action='store_true')  # only train to predict state operation just for debugging

    parser.add_argument("--use_one_optim", action='store_true')  # I use one optim

    parser.add_argument("--recover_e", default=0, type=int)

    # Required parameters
    parser.add_argument("--data_root", default='data/mwz2.1', type=str)
    parser.add_argument("--train_data", default='train_dials.json', type=str)
    parser.add_argument("--dev_data", default='dev_dials.json', type=str)
    parser.add_argument("--test_data", default='test_dials.json', type=str)
    parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='./assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--bert_ckpt_path", default='./assets/bert-base-uncased-pytorch_model.bin', type=str)
    parser.add_argument("--save_dir", default='outputs', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=3e-5, type=float)  # my Transformer-AR uses 3e-5
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=1, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--not_shuffle_state", default=False, action='store_true')
    parser.add_argument("--shuffle_p", default=0.5, type=float)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--msg", default=None, type=str)
    parser.add_argument("--exclude_domain", default=False, action='store_true')

    # generator
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument('--ngram_size', type=int, default=2)

    args = parser.parse_args()
    args.train_data_path = os.path.join(args.data_root, args.train_data)
    args.dev_data_path = os.path.join(args.data_root, args.dev_data)
    args.test_data_path = os.path.join(args.data_root, args.test_data)
    args.ontology_data = os.path.join(args.data_root, args.ontology_data)
    args.shuffle_state = False if args.not_shuffle_state else True
    print('pytorch version: ', torch.__version__)
    print(args)
    main(args)
