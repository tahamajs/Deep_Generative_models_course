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
matplotlib.use('Agg')  # Use non-interactive backend
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
    with open('configs/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return dict2namespace(config)

def plot_log_p(filename, dataset, rnn, data_dict):
    lls = []
    for i, str in data_dict.items():
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

    print("# Loading data...")
    # question 5)
    with open(os.path.join(args.checkpoint_dir, 'snippets.pkl'), 'rb') as f:
        snippets = pkl.load(f)
    lbls = []

    # Load datasets and compute average likelihoods
    with open(os.path.join(args.checkpoint_dir, 'random.pkl'), 'rb') as f:
        random_data = pkl.load(f)
    with open(os.path.join(args.checkpoint_dir, 'shakespeare.pkl'), 'rb') as f:
        shakespeare_data = pkl.load(f)
    with open(os.path.join(args.checkpoint_dir, 'nips.pkl'), 'rb') as f:
        nips_data = pkl.load(f)

    print("# Computing likelihoods...")
    # Compute likelihoods for each dataset
    random_lls = [rnn.compute_prob(np.asarray([dataset.char2idx[c] for c in text])) for text in random_data.values()]
    shakespeare_lls = [rnn.compute_prob(np.asarray([dataset.char2idx[c] for c in text])) for text in shakespeare_data.values()]
    nips_lls = [rnn.compute_prob(np.asarray([dataset.char2idx[c] for c in text])) for text in nips_data.values()]

    avg_random = np.mean(random_lls)
    avg_shakespeare = np.mean(shakespeare_lls)
    avg_nips = np.mean(nips_lls)
    print("# Likelihoods computed.")

    print("# Computing snippet labels...")
    for i, snippet in enumerate(snippets):
        if (i + 1) % 50 == 0:
            print(f"# Processing snippet {i+1}/{len(snippets)}")
        ll = rnn.compute_prob(np.asarray([dataset.char2idx[c] for c in snippet]))
        dists = [
            abs(ll - avg_random),
            abs(ll - avg_shakespeare),
            abs(ll - avg_nips)
        ]
        label = int(np.argmin(dists))
        lbls.append(label)

    # Save answers.pkl
    with open('answers.pkl', 'wb') as f:
        pkl.dump(lbls, f, protocol=pkl.HIGHEST_PROTOCOL)
    print("# answers.pkl saved.")

    # Generate samples
    print("# Generating samples...")
    samples = []
    for i in range(3):  # Generate fewer samples for testing
        print(f"# Generating sample {i+1}/3")
        sample_tokens = rnn.sample(500)  # Shorter sequences
        sample_text = ''.join([dataset.idx2char[token] for token in sample_tokens])
        samples.append(sample_text)

    with open('samples.txt', 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample + '\n\n')
    print("# samples.txt saved.")

    # Generate plots
    print("# Generating plots...")
    plot_log_p('random', dataset, rnn, random_data)
    plot_log_p('shakespeare', dataset, rnn, shakespeare_data)
    plot_log_p('nips', dataset, rnn, nips_data)

    print("# All required files generated successfully!")

if __name__ == '__main__':
    sys.exit(main())
