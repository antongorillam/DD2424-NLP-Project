from utils import read_data, load_model
import torch
import pandas as pd
import numpy as np
import lstm
import rnn
import time

import pandas as pd
import torch


import lstm
import rnn
from metrics import getPerplexity, getSpellPercentage, getAdjustedBLEU, getMetrics
from utils import read_data

"""
Deciding on what metrics to use
"""
if __name__ == "__main__":
    data_dict = read_data()
    text = data_dict["train_text"]
    test_text = data_dict["test_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    DIR = "../results/rnn_vs_lstm"
    TEST_BIGRAMS = '../data/bigrams/testBigramsMerged.txt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DIR = "../results/rnn_vs_lstm"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    lstm_gen = load_model(
        dir=f"{DIR}/lstm_hidden100_epoch100000_lr0.01_nlayer2.pth",
        hidden_size=100,
        num_layers=2,
        )
    lstm_gen.generate()

    gen_1 = lstm_gen.generate(generated_seq_length=100, temperature=0.1)
    gen_3 = lstm_gen.generate(generated_seq_length=100, temperature=0.3)
    gen_9 = lstm_gen.generate(generated_seq_length=100, temperature=0.9)
    # print(getPerplexity('../data/bigrams/old_man_model.txt',gen))
    print(f"temp = 0.1 {gen_1}\n")
    print(f"temp = 0.3 {gen_3}\n")
    # print(f"temp = 0.3 {lstm_gen.generate(generated_seq_length=100, temperature=.3)}\n")
    # print(f"temp = 0.4 {lstm_gen.generate(generated_seq_length=100, temperature=.4)}\n")
    # print(f"temp = 0.5 {lstm_gen.generate(generated_seq_length=100, temperature=.5)}\n")
    # print(f"temp = 0.6 {lstm_gen.generate(generated_seq_length=100, temperature=.6)}\n")
    print(f"temp = 0.99 {gen_9}\n")

    # print(getPerplexity(TEST_BIGRAMS, gen_1))
    # print(getPerplexity(TEST_BIGRAMS, gen_3))
    # print(getPerplexity(TEST_BIGRAMS, gen_9))

    # print(getSpellPercentage(gen_1))
    # print(getSpellPercentage(gen_3))
    # print(getSpellPercentage(gen_9))

    # # toc = time.perf_counter()
    # scorer = Jury(metrics=['bleu'])
    # predictions = [gen_9]
    # references = [test_text]
    # score = scorer.evaluate(predictions=predictions, references=references)
    # # time_elapsed_sec = time.perf_counter() - toc
    # # time_elapsed = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time_elapsed_sec))
    # print(f'score: {score}')
    # adjBleu, _ = getAdjustedBLEU(gen_9, test_text)
    # print('bleu: ', adjBleu)
    print(getMetrics(gen_9, test_text, TEST_BIGRAMS))
    # print(f'Time elapsed: {time_elapsed}') 
