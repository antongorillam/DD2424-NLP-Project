from utils import read_data_shakespeare, load_model
import torch
import pandas as pd
import numpy as np
import time

import pandas as pd
import torch
from utils import read_data_shakespeare

import lstm
import rnn
from metrics import getPerplexity, getSpellPercentage, getAdjustedBLEU, getMetrics
from utils import read_data


class Benchmark:
    """
    Class that 
    """
    def __init__(self):

        self.data_dict = read_data_shakespeare()
        self.train_text = self.data_dict["train_text"]
        self.test_text = self.data_dict["test_text"]
        self.index2char = self.data_dict["index2char"]
        self.char2index = self.data_dict["char2index"]

    def run_benchmark(self, model_dir, hidden_size):
        """
        Performs benchmarking
        ---------------------
        params:
        ------
        model_dir (string) : 
            Directory and model name we want to perform benchmarking with
        """
        TEST_BIGRAMS = '../data/bigrams/testBigramsMerged.txt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        lstm_gen = load_model(
            dir=model_dir,
            hidden_size=hidden_size,
            num_layers=2,
            )
        generated_text = lstm_gen.generate(generated_seq_length=100, temperature=1)
        metrics = getMetrics(generated_text, self.index2char, TEST_BIGRAMS)
        
        return metrics

# if __name__ == "__main__":

#     MODEL_DIR = "../results/hidden_vs_loss/learning_rate_0_005/lstm_hidden500_epoch10000_lr0.005_nlayer2.pth"
#     TEST_BIGRAMS = '../data/bigrams/testBigramsMerged.txt'
#     benchmark = Benchmark()
#     metrics = benchmark.run_benchmark(model_dir=MODEL_DIR, hidden_size=500)
    