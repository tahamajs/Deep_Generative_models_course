import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_lstm_units, num_lstm_layers, dataset, device):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=num_lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        self.h2o = nn.Linear(num_lstm_units, vocab_size)
        self.device = device
        self.dataset = dataset
        self.to(device)

    def forward(self, input, hidden=None):
        """
        Predict the next token's logits given an input token and a hidden state.
        :param input [torch.tensor]: The input token tensor with shape
            (batch_size, 1), where batch_size is the number of inputs to process
            in parallel.
        :param hidden [(torch.tensor, torch.tensor)]: The hidden state, or None if
            it's the first token.
        :return [(torch.tensor, (torch.tensor, torch.tensor))]: A tuple consisting of
            the logits for the next token, of shape (batch_size, vocab_size), and
            the next hidden state.
        """
        embeddings = self.embedding(input)
        if hidden is None:
            lstm_out, (h, c) = self.lstm(embeddings)
        else:
            lstm_out, (h, c) = self.lstm(embeddings, hidden)

        lstm_out = lstm_out.contiguous().view(-1, lstm_out.shape[2])
        logits = self.h2o(lstm_out)
        return logits, (h.detach(), c.detach())

    def sample(self, seq_len):
        """
        Sample a sequence of tokens of length `seq_len` from the model.
        :param seq_len [int]: Sequence length.
        :return [list]: A list of length `seq_len` containing the generated tokens.
                        Tokens are integers in the range [0, vocab_size - 1].
        """
        voc_freq = self.dataset.voc_freq  
        self.eval()  
        texts = []
        with torch.no_grad():
            initial_token = np.random.choice(np.arange(voc_freq.shape[0]), p=voc_freq)
            x = torch.tensor([[initial_token]], dtype=torch.int64, device=self.device)
            texts.append(initial_token)
            hidden = None

            for _ in range(seq_len - 1):
                logits, hidden = self.forward(x, hidden)
                probs = F.softmax(logits, dim=1) 
                next_token = torch.multinomial(probs, num_samples=1)
                token_int = next_token.item()
                texts.append(token_int)
                x = torch.tensor([[token_int]], dtype=torch.int64, device=self.device)
        return texts

    def compute_prob(self, string):
        """
        Compute the log-likelihood of generating a string of tokens.
        :param string [np.ndarray or list]: a sequence (list or 1D numpy array) 
                of integer tokens, e.g., [t0, t1, t2, ...].
        :return [float]: the log-likelihood of the sequence.
        """
        voc_freq = self.dataset.voc_freq
        self.eval()
        ll = 0.0
        with torch.no_grad():
            first_token = string[0]
            ll += np.log(voc_freq[first_token])
            hidden = None
            x = torch.tensor([[first_token]], dtype=torch.int64, device=self.device)
            
            for token in string[1:]:
                logits, hidden = self.forward(x, hidden)
                probs = F.softmax(logits, dim=1)  
                token_prob = probs[0, token].item()  
                ll += np.log(token_prob)
                x = torch.tensor([[token]], dtype=torch.int64, device=self.device)
        return ll
