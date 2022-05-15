import time

import numpy as np
import pandas as pd
import torch

import lstm
import rnn
from metrics import getAdjustedBLEU, getMetrics, getPerplexity, getSpellPercentage
from utils import load_model, read_data, read_data_shakespeare


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

    def run_benchmark(
        self,
        model_dir,
        hidden_size,
        temperature,
        save_dir,
        initial_str="ROMEO",
        generated_seq_length=200,
    ):
        """
        Performs benchmarking
        --------------------
        params:
        ------
        model_dir (string) :
            Directory and model name we want to perform benchmarking with
        """
        TEST_BIGRAMS = "../data/bigrams/testBigramsShakespeare.txt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_gen = load_model(
            dir=model_dir,
            hidden_size=hidden_size,
            num_layers=2,
        )
        generated_text = lstm_gen.generate(
            initial_str=initial_str,
            generated_seq_length=generated_seq_length,
            temperature=temperature,
        )
        metrics = getMetrics(generated_text, self.test_text, TEST_BIGRAMS)

        spelling_percentage = metrics["spelling_percentage"]
        perplexity = metrics["perplexity"]
        bleu = metrics["bleu"]
        ngram_precisions = metrics["ngram_precisions"]
        bartscore = metrics["bartscore"]
        bertscore = metrics["bertscore"]

        with open(f"{save_dir}/hidden_{hidden_size}_temperature_{temperature}.txt", "w") as f:
            f.write(
                f"Configuration, hidden size: {hidden_size}, temperature:{temperature}, sequence length: {generated_seq_length}\n"
            )
            f.write(f"generated_text:\n{generated_text}\n")
            f.write(f"spelling_percentage: {spelling_percentage}\n")
            f.write(f"perplexity: {perplexity}\n")
            f.write(f"bleu: {bleu}\n")
            f.write(f"bartscore: {bartscore}\n")
            f.write(f"ngram_precisions: {ngram_precisions}\n")
            f.write(f"bertscore: {bertscore}\n")


if __name__ == "__main__":

    MODEL_DIR = "../results/hidden_vs_loss/learning_rate_0_005/lstm_hidden500_epoch10000_lr0.005_nlayer2.pth"
    TEST_BIGRAMS = "../data/bigrams/testBigramsShakespeare.txt"
    benchmark = Benchmark()

    TEMPERATURES = [0.2, 0.6, 1, 1.2, 2, 3]
    HIDDEN_SIZES = [500]

    for temp in TEMPERATURES:
        metrics = benchmark.run_benchmark(
            model_dir=MODEL_DIR,
            hidden_size=500,
            temperature=temp,
            initial_str="MON",
            generated_seq_length=200,
            save_dir="../results/score_check",
        )
