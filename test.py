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
        args.random_seed = random.sample(list(range(1, 100000)), 1)[0]

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
    train_data_raw = torch.load(train_path)[:5000]
    print("# train examples %d" % len(train_data_raw))

    test_path = os.path.join(args.data_root, "test.pt")
    test_data_raw = torch.load(test_path)
    print("# test examples %d" % len(test_data_raw))

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

    test_epochs = [int(e) for e in args.load_epoch.strip().lower().split('-')]
    for best_epoch in test_epochs:
        print("### Epoch {:}...".format(best_epoch))
        sys.stdout.flush()
        ckpt_path = os.path.join(args.save_dir, 'model.e{:}.bin'.format(best_epoch))
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt)
        model.to(device)

        # eval_res = model_evaluation(model, train_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
        #                             use_full_slot=args.use_full_slot, use_dt_only=args.use_dt_only, no_dial=args.no_dial, n_gpu=n_gpu,
        #                             is_gt_op=False, is_gt_p_state=False, is_gt_gen=False)
        #
        # print("### Epoch {:} Train Score : ".format(best_epoch), eval_res)
        # print('\n'*2)
        # sys.stdout.flush()

        eval_res = model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, args.op_code,
                                    use_full_slot=args.use_full_slot, use_dt_only=args.use_dt_only, no_dial=args.no_dial, n_gpu=n_gpu,
                                    is_gt_op=False, is_gt_p_state=False, is_gt_gen=False)

        print("### Epoch {:} Test Score : ".format(best_epoch), eval_res)
        print('\n'*2)
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_epoch", required=True, type=str, help="example: '10-11-12' ")


    parser.add_argument("--use_cpu", action='store_true')  # Just for my debugging. I have not tested whether it can be used for training model.

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
