import time

import pandas as pd
import torch
from jury import Jury

import lstm
import rnn
from perplexity import getPerplexity
from utils import read_data

"""
Test of how to
"""
if __name__ == "__main__":

    DIR_TRAIN = "../data/The_Sun_Also_Rises.txt"
    DIR_TEST = "../data/Old_Man_And_The_Sea.txt"

    data_dict = read_data(DIR_TRAIN, DIR_TEST)
    text = data_dict["train_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    DIR = "../results"
    SEQUENCE_LENGTH = 25
    BATCH_SIZE = 1
    NUM_EPOCHS = 10000
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2
    TEMPERATURE = 0.28
    LEARNING_RATE = 0.01
    LABEL_SMOOTHING = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm_model = lstm.RNN(
        input_size=len(index2char),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(index2char),
    )

    lstm_model.load_state_dict(
        torch.load(f"{DIR}/lstm_epoch10000_lr0.01_nlayer2.pth", map_location=device)
    )

    lstm_gen = lstm.Generator(
        input_string=text,
        index2char=index2char,
        char2index=char2index,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
    )

    # print(lstm_gen)

    lstm_gen.lstm = lstm_model

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

    print(getPerplexity("../data/bigrams/old_man_model.txt", gen_1))
    print(getPerplexity("../data/bigrams/old_man_model.txt", gen_3))
    print(getPerplexity("../data/bigrams/old_man_model.txt", gen_9))

    # toc = time.perf_counter()
    # scorer = Jury(metrics=['bleu','rouge','bertscore'])
    # predictions = [gen]
    # references = [text]
    # score = scorer.evaluate(predictions=predictions, references=references)
    # time_elapsed_sec = time.perf_counter() - toc
    # time_elapsed = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time_elapsed_sec))
    # print(f'score: {score}')
    # print(f'Time elapsed: {time_elapsed}')

    """
    Example of how to generate a text, George will have to 
    - modify the function generate() in lstm.py so that it performs nucleaus sampling
    - Also implement Beam Search there is time
    """
