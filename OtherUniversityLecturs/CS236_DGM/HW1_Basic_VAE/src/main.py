import argparse

import sys
import os
import yaml
import torch
import numpy as np
import pickle as pkl
from dataset import NIPS2015Dataset
from model import RNN

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAMPLE_SEQ_LEN = 1000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory of saving checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory of papers.csv')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory of putting logs')
    parser.add_argument('--gpu', action='store_true', help="Turn on GPU mode")

    args = parser.parse_args()
    return args


def dict2namespace(config):
    new_config = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(new_config, key, value)
    return new_config


def parse_config(args):
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return dict2namespace(config)

def plot_log_p(filename, dataset, rnn):
    with open(filename + '.pkl', 'rb') as f:
        lls = []
        data = pkl.load(f)
        for i, str in data.items():
            str_np = np.asarray([dataset.char2idx[c] for c in str])
            lls.append(rnn.compute_prob(str_np))

    with open(filename + '_raw.pkl', 'wb') as f:
        pkl.dump(lls, f, protocol=pkl.HIGHEST_PROTOCOL)

    plt.figure()
    plt.hist(lls)
    plt.xlabel('Log-likelihood')
    plt.xlim([-800, -50])
    plt.ylabel('Counts')
    plt.title(filename)
    plt.savefig(filename + '.png', bbox_inches='tight')
    plt.show()
    plt.close()
    print("# Figure written to %s.png." % filename)


def main():
    args = parse_args()
    config = parse_config(args)
    np.random.seed(config.seed)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(config.seed)

    dataset = NIPS2015Dataset(batch_size=config.batch_size,
                              seq_len=config.seq_len,
                              data_folder=args.data_dir)

    rnn = RNN(
        vocab_size=dataset.voc_len,
        embedding_dim=config.embedding_dim,
        num_lstm_units=config.num_lstm_units,
        num_lstm_layers=config.num_lstm_layers,
        dataset=dataset,
        device=device
    )

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'), map_location=device)
    rnn.load_state_dict(checkpoint['rnn'])
    print("# RNN weights restored.")

    # question 5)
    with open('snippets.pkl', 'rb') as f:
        snippets = pkl.load(f)
    lbls = []

    with open('random_raw.pkl', 'rb') as f:
        random_lls = pkl.load(f)
    with open('shakespeare_raw.pkl', 'rb') as f:
        shakespeare_lls = pkl.load(f)
    with open('nips_raw.pkl', 'rb') as f:
        nips_lls = pkl.load(f)

    avg_random = np.mean(random_lls)
    avg_shakespeare = np.mean(shakespeare_lls)
    avg_nips = np.mean(nips_lls)

    for snippet in snippets:
        ll = rnn.compute_prob(np.asarray([dataset.char2idx[c] for c in snippet]))
        dists = [
            abs(ll - avg_random),
            abs(ll - avg_shakespeare),
            abs(ll - avg_nips)
        ]
        label = int(np.argmin(dists))
        lbls.append(label)

if __name__ == '__main__':
    sys.exit(main())
