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
        iters=1,
    ):
        """
        Performs benchmarking
        --------------------
        params:
        ------
        model_dir (string) :
            Directory and model name we want to perform benchmarking with
        """
        import time
        from statistics import mean  # Screw numpys

        TEST_BIGRAMS = "../data/bigrams/testBigramsShakespeare.txt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_gen = load_model(
            dir=model_dir,
            hidden_size=hidden_size,
            num_layers=2,
        )

        dic = {
            "temperature": temperature,
            "hidden_size": hidden_size,
            "generated_text": None,
            "spelling_percentage": [],
            "perplexity": [],
            "bleu1": [],
            "bleu2": [],
            "bleu3": [],
            "bleu4": [],
            "ngram_precisions_1": [],
            "ngram_precisions_2": [],
            "ngram_precisions_3": [],
            "ngram_precisions_4": [],
            "bartscore": [],
            "bertscore": [],
            "bertscore_precision": [],
            "bertscore_recall": [],
            "bertscore_f1": [],
        }
        tic = time.perf_counter()
        print("Starting Benchmarking...")
        for i in range(iters):
            if initial_str == None:
                captial_letters = [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "H",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "N",
                    "O",
                    "P",
                    "Q",
                    "R",
                    "S",
                    "T",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]
                temp_inital_str = np.random.choice(captial_letters)
            else:
                temp_inital_str = initial_str

            generated_text = lstm_gen.generate(
                initial_str=temp_inital_str,
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

            dic["generated_text"] = generated_text
            dic["spelling_percentage"].append(spelling_percentage)
            dic["perplexity"].append(perplexity)
            dic["bleu1"].append(bleu[1])
            dic["bleu2"].append(bleu[2])
            dic["bleu3"].append(bleu[3])
            dic["bleu4"].append(bleu[4])
            dic["ngram_precisions_1"].append(ngram_precisions[1])
            dic["ngram_precisions_2"].append(ngram_precisions[2])
            dic["ngram_precisions_3"].append(ngram_precisions[3])
            dic["ngram_precisions_4"].append(ngram_precisions[4])
            dic["bartscore"].append(bartscore["score"])
            dic["bertscore"].append(bertscore["score"])
            dic["bertscore_precision"].append(bertscore["precision"][0])
            dic["bertscore_recall"].append(bertscore["recall"][0])
            dic["bertscore_f1"].append(bertscore["f1"][0])

            print(
                f"On iteration {i+1}/{iters}, time elapsed: {time.perf_counter()-tic}"
            )

        # Take the mean of the metrics is several runs were made
        dic["spelling_percentage"] = mean(dic["spelling_percentage"])
        dic["perplexity"] = mean(dic["perplexity"])
        dic["bleu1"] = mean(dic["bleu1"])
        dic["bleu2"] = mean(dic["bleu2"])
        dic["bleu3"] = mean(dic["bleu3"])
        dic["bleu4"] = mean(dic["bleu4"])
        dic["ngram_precisions_1"] = mean(dic["ngram_precisions_1"])
        dic["ngram_precisions_2"] = mean(dic["ngram_precisions_2"])
        dic["ngram_precisions_3"] = mean(dic["ngram_precisions_3"])
        dic["ngram_precisions_4"] = mean(dic["ngram_precisions_4"])
        dic["bartscore"] = mean(dic["bartscore"])
        dic["bertscore"] = mean(dic["bertscore"])
        dic["bertscore_precision"] = mean(dic["bertscore_precision"])
        dic["bertscore_recall"] = mean(dic["bertscore_recall"])
        dic["bertscore_f1"] = mean(dic["bertscore_f1"])

        temp_df = pd.DataFrame([dic])
        temp_df.to_csv(
            f"{SAVE_DIR}/hidden{hidden_size}_temp{temperature}_check.csv", index=False
        )


if __name__ == "__main__":

    SAVE_DIR = "../results/score_check"
    # MODEL_DIR = "../results/hidden_vs_loss/learning_rate_0_005/lstm_hidden500_epoch10000_lr0.005_nlayer2.pth"
    TEST_BIGRAMS = "../data/bigrams/testBigramsShakespeare.txt"
    benchmark = Benchmark()

    TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
    # df_check = pd.DataFrame(
    #     columns=[
    #         "temperature",
    #         "hidden_size",
    #         "generated_text",
    #         "spelling_percentage",
    #         "perplexity",
    #         "bleu1",
    #         "bleu2",
    #         "bleu3",
    #         "bleu4",
    #         "ngram_precisions_1",
    #         "ngram_precisions_2",
    #         "ngram_precisions_3",
    #         "ngram_precisions_4",
    #         "bartscore",
    #         "bertscore",
    #         "bertscore_precision",
    #         "bertscore_recall",
    #         "bertscore_f1",
    #     ]
    # )
    # for temp in TEMPERATURES:
    #     temp_dic = benchmark.run_benchmark(
    #         model_dir=MODEL_DIR,
    #         hidden_size=500,
    #         temperature=temp,
    #         initial_str="MON",
    #         generated_seq_length=400,
    #         save_dir="../results/score_check",
    #         iters=1,
    #     )
    #     temp_df = pd.DataFrame([temp_dic])
    #     df_check = pd.concat([df_check, temp_df], ignore_index=True)

    # df_check.to_csv(f"{SAVE_DIR}/metric_check.csv", index=False)

    BEST_TEMPERATURE = 0.3
    HIDDEN_SIZES = [25, 50, 250, 500]
    MODEL_DIRS = [
        "../results/hidden_vs_loss/learning_rate_0_005/lstm_hidden25_epoch10000_lr0.005_nlayer2.pth",
        "../results/hidden_vs_loss/learning_rate_0_005/lstm_hidden50_epoch10000_lr0.005_nlayer2.pth",
        "../results/hidden_vs_loss/learning_rate_0_005/lstm_hidden250_epoch10000_lr0.005_nlayer2.pth",
        "../results/hidden_vs_loss/learning_rate_0_005/lstm_hidden500_epoch10000_lr0.005_nlayer2.pth",
    ]

    for model, hidden in zip(MODEL_DIRS, HIDDEN_SIZES):
        temp_dic = benchmark.run_benchmark(
            model_dir=model,
            hidden_size=hidden,
            temperature=BEST_TEMPERATURE,
            initial_str=None,
            generated_seq_length=400,
            save_dir="../results/score_check",
            iters=5,
        )
    # temp_df = pd.DataFrame([temp_dic])
    # df_hidden = pd.concat([df_hidden, temp_df], ignore_index=True)

    # df_hidden.to_csv(f"{SAVE_DIR}/hidden{HIDDEN_SIZES[0]}_check.csv", index=False)
