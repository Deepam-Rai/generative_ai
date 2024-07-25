from pathlib import Path
import random
from typing import List
import re
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from utils import get_text
import torch.nn as nn
from datetime import datetime


class TextDatasetPlus(Dataset):
    """
    Dataset class that
    1. takes path of data-files,
    2. extracts texts
    This class itself becomes the Dataset class that can be used for DataLoader
    """
    def __init__(self, data_paths: List[Path], seq_length: int = 10, batch_size: int = 64):
        self.paths: List[Path] = data_paths
        self.text: str = get_text(self.paths)[:1000]
        # self.text: str = self.process_text(self.text)
        self.tokens = list(set(self.tokenize_text(self.text)))
        self.vocab_size = len(self.tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.text_as_tokens = self.tokenize_text(self.text)
        self.text_as_token_ids = [self.token_to_id[w] for w in self.tokenize_text(self.text)]

    def __len__(self):
        return len(self.text_as_token_ids) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.text_as_token_ids[index:index+self.seq_length], dtype=torch.long).to(device),
            torch.tensor(self.text_as_token_ids[index+1:index+self.seq_length+1], dtype=torch.long).to(device),
        )

    def tokenize_text(self, text: str) -> list:
        return list(text)

    def textify_tokens(self, tokens: list) -> str:
        return ''.join(tokens)

    def one_hot(self, tokens: list) -> List[list]:
        one_hots = []
        for token in tokens:
            enc = [0] * self.vocab_size
            enc[self.token_to_id[token]] = 1
            one_hots.append(enc)
        return one_hots
    def process_text(self, text: str) -> str:
        """
        1. Replaces "newline" and "tab" with a space.
        2. Inserts space before and after punctuations.
        :param text:
        :return: cleaned text
        """
        cleaned_text = text
        cleaned_text = cleaned_text.replace('\n', ' ').replace('\t', ' ')
        cleaned_text = cleaned_text.replace('“', ' " ').replace('”', ' " ')
        for punc in ['.', '-', ',', '!', '?', '(', '—', ')', '/', '\'', ';', ':', '‘', '$']:
            cleaned_text = cleaned_text.replace(punc, f' {punc} ')
        cleaned_text = re.sub("\s\s+", " ", cleaned_text)
        return cleaned_text


class RNNGen(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.gru_layers = num_layers
        self.gru_dim = hidden_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, states):
        x = self.embedding(x)
        gru_out, states = self.gru(x, states)
        # gru_out = gru_out[:, -1, :]
        out = self.out(gru_out)
        return out, states

    def init_state(self, batch_size: int):
        return torch.zeros(self.gru_layers, batch_size, self.gru_dim).to(device)


def train(dataloader, model, max_epochs, sequence_length, checkpoints: bool = True):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, max_epochs+1):
        print(f"\nEPOCH : {epoch}")
        state = model.init_state(batch_size=batch_size)
        for batch, (x, y) in (pbar := tqdm(enumerate(dataloader))):
            optimizer.zero_grad()

            y_pred, state = model(x, state)
            loss = criterion(y_pred.transpose(1, 2), y)

            state = state.detach()

            loss.backward()
            optimizer.step()
            pbar.set_description(f'epoch: {epoch}/{max_epochs} \t batch: {batch} \t loss: {loss.item()}')
        if checkpoints:
            torch.save(
                model.state_dict(),
                Path(f'./models/checkpoints/{datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}_epochs_{epoch}_device_{device}')
            )
            print(
                f"Generated text:\n"
                f"-------------------\n"
                f"{predict(dataset, model, eval_mode=False)}\n"
                f"-------------------\n"
            )


def predict(dataset, model, seed: str = None, gen_length=100, eval_mode: bool=True):
    if seed is None:
        # tokens = random.choices(dataset.tokens, k=dataset.seq_length)
        tokens = [random.choice(dataset.tokens)]
    else:
        tokens = dataset.tokenize_text(seed)
    if eval_mode:
        model.eval()
    state = model.init_state(batch_size=1)
    for i in range(0, gen_length):
        x = torch.tensor([[dataset.token_to_id[w] for w in tokens[i:]]]).to(device)
        y_pred, state = model(x, state)
        state = state.detach()
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        token_index = np.random.choice(len(last_word_logits), p=p)
        tokens.append(dataset.id_to_token[token_index])
    return dataset.textify_tokens(tokens)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
embed_dim = 256
hidden_dim = 1024
num_layers = 1
dropout = 0.2
batch_size = 64
seq_length = 100
epochs = 10


data_path = Path('./data/Shakespeare')
dataset = TextDatasetPlus([data_path], seq_length=seq_length, batch_size=batch_size)
vocab_size = dataset.vocab_size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(
    f"DATASET:  {data_path}\n"
    f"Vocab size: {dataset.vocab_size}  Corpus size: {len(dataset.text_as_token_ids)}"
    f"Number of batches:{len(dataloader)} \t batch_size = {batch_size} \t seq_length = {seq_length}"
)


model = RNNGen(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
# model.load_state_dict(torch.load(Path('./models/checkpoints/25_07_2024__17_03_07_epochs_19_device_cuda'), weights_only=True))
print(f"\nMODEL: {model}")


X, Y = next(iter(dataloader))
y_pred, _ = model(X, model.init_state(batch_size=batch_size))
print("\nPRETRAIN PERFORMANCE:\n---------- Y - Original target")
print(''.join([dataset.id_to_token[x] for x in Y[0].cpu().tolist()]))
print("---------- Generated")
print(''.join([random.choices(dataset.tokens, p.cpu().detach().numpy())[0] for p in y_pred[0]]))
print("---------- END")

print("\nSTARTING TRAINING:")
train(dataloader, model, max_epochs=epochs, sequence_length=seq_length)
print("TRAINING COMPLETE.\n")

print(f"\nSAMPLE GENERATED TEXT:\n{predict(dataset, model, gen_length=100)}")
