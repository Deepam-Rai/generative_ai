# Reference: https://www.kdnuggets.com/2019/11/markov-chains-train-text-generation.html
import random
import re
import sys
from pathlib import Path
from typing import List, Dict

from scipy.sparse import dok_matrix
import logging
import coloredlogs
from constants import *
from utils import get_text

logging.basicConfig()
logger = logging.getLogger(__name__)
coloredlogs.install(
    level=logging.NOTSET,
    logger=logger,
    fmt='%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s():%(lineno)s: %(message)s',
    field_styles=LOGGER_FIELD_STYLE,
    level_styles=LOGGER_LEVEL_STYLES
)


class MarkovGen:
    def __init__(self, file_paths: List[Path]):
        self.k: int = None
        self.k_words_index_map: dict = None
        self.k_matrix: dok_matrix = None
        self.paths: List[Path] = file_paths
        text = get_text(self.paths)
        self.text: str = self.clean_text(text)
        corpus = list(self.text.split(' '))
        corpus = list(filter(lambda x: x != '', corpus))
        logger.info(f"Corpus size: {len(corpus)}")
        self.tokens_index_map = {word: idx for idx, word in enumerate(list(set(corpus)))}

    def clean_text(self, text: str) -> str:
        """
        1. Replaces "newline" and "tab" with a space.
        2. Inserts space before and after punctuations.
        :param text:
        :return: cleaned text
        """
        cleaned_text = text.replace('\n', ' ').replace('\t', ' ')
        cleaned_text = cleaned_text.replace('“', ' " ').replace('”', ' " ')
        for punc in ['.', '-', ',', '!', '?', '(', '—', ')', '/', '\'', ';', ':', '‘', '$']:
            cleaned_text = cleaned_text.replace(punc, f' {punc} ')
        cleaned_text = re.sub("\s\s+", " ", cleaned_text)
        return cleaned_text

    def get_k_matrix(self, k: int):
        """
        Creates a matrix that maps conditional probability of next word in the sentence,
        given past k words in the sentence.
        :param tokens_index_map:
        :param k: past-k words to be considered
        :param text:
        :return:
        """
        if k < 1:
            raise ValueError("k should be greater than 0.")
        self.k = k
        corpus = list(self.text.split(' '))
        corpus = list(filter(lambda x: x != '', corpus))
        set_of_k_words = [' '.join(corpus[i:i+k]) for i in range(len(corpus)-k)]
        count_distinct_words = len(self.tokens_index_map)
        k_words_distinct = list(set(set_of_k_words))
        self.k_words_index_map = {s: idx for idx, s in enumerate(k_words_distinct)}
        k_matrix = dok_matrix((len(k_words_distinct), count_distinct_words))
        for i, last_k in enumerate(set_of_k_words[:-k]):
            last_k_words_row = self.k_words_index_map[last_k]
            next_word_col = self.tokens_index_map[corpus[i + k]]
            k_matrix[last_k_words_row, next_word_col] += 1
        self.k_matrix = k_matrix
        return self.k_matrix

    def sample_next_word_after_seq(self, seq: str, alpha: float = 0) -> str:
        """
        Given sequence of last k characters, samples next word
        :param seq:
        :param alpha: Creativity hyperparameter, large the value => distribution approches uniormity => choice of random words
        :return:
        """
        next_word_vector = self.k_matrix[self.k_words_index_map[seq]] + alpha
        likelihoods = next_word_vector/next_word_vector.sum()
        return random.choices(list(self.tokens_index_map.keys()), likelihoods.toarray()[0])[0]

    def stochastic_chain(self, seed: str = None, chain_length: int = 15):
        """
        Returns sentence with added words equal to 'chain_length'.
        The last k-word phrase of the seed sentence need to be present in some part of the training dataset.

        :param seed: If None, randomly generates seed sentence.
        :param chain_length: Added words to be generated.
        :return: seed sentence + generated sentence.
        """
        if seed is None:
            seed = random.choice(list(self.k_words_index_map.keys()))
        current_words = seed.split(' ')
        if len(current_words) < self.k:
            raise ValueError(f"Seed length must be at-least as long as {self.k}")
        sentence = seed
        current_words = current_words[-self.k:]
        if self.k_words_index_map.get(' '.join(current_words), None) is None:
            raise ValueError("The last k-word phrase of the seed sentence need to be present in some part of the training dataset.")
        for _ in range(chain_length):
            sentence += ' '
            next_word = self.sample_next_word_after_seq(' '.join(current_words))
            sentence += next_word
            current_words = current_words[1:] + [next_word]
        return sentence


k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
chain_length = int(sys.argv[2]) if len(sys.argv) > 2 else 100
data_path = sys.argv[3] if len(sys.argv) > 3 else "./data/Sherlock_Holmes"


MG = MarkovGen(file_paths=[
    Path(data_path)
])
logger.info(f"Unique tokens count = {len(MG.tokens_index_map)} \t k = {k}")
MG.get_k_matrix(k=k)

print(MG.stochastic_chain(seed=None, chain_length=chain_length))
