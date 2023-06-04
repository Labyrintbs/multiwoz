from __future__ import division, print_function, unicode_literals

import argparse
import json
import random
import time
import datetime
from io import open
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import torch
from torch.optim import Adam

from utils import util
from model.model import Model


parser = argparse.ArgumentParser(description='S2S')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--vocab_size', type=int, default=400, metavar='V')

parser.add_argument('--use_attn', type=util.str2bool, nargs='?', const=True, default=False)
parser.add_argument('--attention_type', type=str, default='bahdanau')
parser.add_argument('--use_emb',  type=util.str2bool, nargs='?', const=True, default=False)

parser.add_argument('--emb_size', type=int, default=50)
parser.add_argument('--hid_size_enc', type=int, default=150)
parser.add_argument('--hid_size_dec', type=int, default=150)
parser.add_argument('--hid_size_pol', type=int, default=150)
parser.add_argument('--db_size', type=int, default=30)
parser.add_argument('--bs_size', type=int, default=94)

parser.add_argument('--cell_type', type=str, default='lstm')
parser.add_argument('--depth', type=int, default=1, help='depth of rnn')
parser.add_argument('--max_len', type=int, default=50)

parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr_rate', type=float, default=0.005)
parser.add_argument('--lr_decay', type=float, default=0.0)
parser.add_argument('--l2_norm', type=float, default=0.00001)
parser.add_argument('--clip', type=float, default=5.0, help='clip the gradient by norm')

parser.add_argument('--teacher_ratio', type=float, default=1.0, help='probability of using targets for learning')
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--no_cuda',  type=util.str2bool, nargs='?', const=True, default=True)

parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
parser.add_argument('--train_output', type=str, default='data/train_dials/', help='Training output dir path')

parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--early_stop_count', type=int, default=5)
parser.add_argument('--model_dir', type=str, default='model/model/')
parser.add_argument('--model_name', type=str, default='translate.ckpt')

parser.add_argument('--load_param', type=util.str2bool, nargs='?', const=True, default=False)
parser.add_argument('--epoch_load', type=int, default=0)

parser.add_argument('--mode', type=str, default='train', help='training or testing: test, train, RL')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


def train(print_loss_total,print_act_total, print_grad_total, input_tensor, target_tensor, bs_tensor, db_tensor, name=None):
    # create an empty matrix with padding tokens
    input_tensor, input_lengths = util.padSequence(input_tensor)
    target_tensor, target_lengths = util.padSequence(target_tensor)
    bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
    db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

    loss, loss_acts, grad = model.train(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor,
                             bs_tensor, name)

    #print(loss, loss_acts)
    print_loss_total += loss
    print_act_total += loss_acts
    print_grad_total += grad

    model.global_step += 1
    model.sup_loss = torch.zeros(1)

    return print_loss_total, print_act_total, print_grad_total


def trainIters(model, n_epochs=10, args=args):
    prev_min_loss, early_stop_count = 1 << 30, args.early_stop_count
    start = time.time()
    no_improve_count = 0
    for epoch in range(1, n_epochs + 1):
        if no_improve_count > early_stop_count:
            break

        print_loss_total = 0; print_grad_total = 0; print_act_total = 0  # Reset every print_every
        start_time = time.time()
        # watch out where do you put it
        model.optimizer = Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad, model.parameters()), weight_decay=args.l2_norm)
        model.optimizer_policy = Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad, model.policy.parameters()), weight_decay=args.l2_norm)
    
        #add list for python2 to 3 TypeError: 'dict_keys' object does not support indexing
        #see https://blog.csdn.net/qq_33363973/article/details/77877485 for details
        dials = list(train_dials.keys())
        random.shuffle(dials)
        input_tensor = [];target_tensor = [];bs_tensor = [];db_tensor = []
        for name in dials:
            val_file = train_dials[name]
            model.optimizer.zero_grad()
            model.optimizer_policy.zero_grad()

            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor, target_tensor, bs_tensor, db_tensor)

            if len(db_tensor) > args.batch_size:
                print_loss_total, print_act_total, print_grad_total = train(print_loss_total, print_act_total, print_grad_total, input_tensor, target_tensor, bs_tensor, db_tensor)
                input_tensor = [];target_tensor = [];bs_tensor = [];db_tensor = [];

        print_loss_avg = print_loss_total / len(train_dials)
        print_act_total_avg = print_act_total / len(train_dials)
        print_grad_avg = print_grad_total / len(train_dials)
        print('TIME:', time.time() - start_time)
        print('Time since %s (Epoch:%d %d%%) Loss: %.4f, Loss act: %.4f, Grad: %.4f' % (util.timeSince(start, epoch / n_epochs),
                                                            epoch, epoch / n_epochs * 100, print_loss_avg, print_act_total_avg, print_grad_avg))

        # VALIDATION
        valid_loss = 0
        best_loss = float("inf")
        for name, val_file in val_dials.items():
            input_tensor = []; target_tensor = []; bs_tensor = [];db_tensor = []
            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor,
                                                                                         target_tensor, bs_tensor,
                                                                                         db_tensor)
            # create an empty matrix with padding tokens
            input_tensor, input_lengths = util.padSequence(input_tensor)
            target_tensor, target_lengths = util.padSequence(target_tensor)
            bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
            db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

            proba, _, _ = model.forward(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor)
            proba = proba.view(-1, model.vocab_size) # flatten all predictions
            proba = proba.clone().detach().requires_grad_(True).to(device)
            loss = model.gen_criterion(proba, target_tensor.view(-1))
            valid_loss += loss.item()
        valid_loss /= len(val_dials)
        print('Current Valid LOSS:', valid_loss)
        if epoch % 5 == 0:
            if valid_loss < best_loss:
                saveModel(model)
            else:
                no_improve_count += 1



def loadDictionaries():
    # load data and dictionaries
    with open('data/input_lang.index2word.json') as f:
        input_lang_index2word = json.load(f)
    with open('data/input_lang.word2index.json') as f:
        input_lang_word2index = json.load(f)
    with open('data/output_lang.index2word.json') as f:
        output_lang_index2word = json.load(f)
    with open('data/output_lang.word2index.json') as f:
        output_lang_word2index = json.load(f)

    return input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index

def saveModel(model):
    print('Saving parameters..')
    if not os.path.exists(model.model_dir):
        os.makedirs(model.model_dir)
    current_time = datetime.datetime.now().strftime("%Y%m%d")
    model_name = f"model_{current_time}_best.pt"
    with open(model.model_dir + 'Multiwoz_train' + '.config', 'w') as f:
        f.write(json.dumps(vars(model.args), ensure_ascii=False, indent=4))
    torch.save(model.state_dict(), model_name)



if __name__ == '__main__':
    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = loadDictionaries()
    # Load training file list:
    with open('data/train_dials.json') as outfile:
        train_dials = json.load(outfile)

    # Load validation file list:
    with open('data/val_dials.json') as outfile:
        val_dials = json.load(outfile)

    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
    if args.load_param:
        model.loadModel(args.epoch_load)

    trainIters(model, n_epochs=args.max_epochs, args=args)
