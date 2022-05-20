from turtle import color
from utils import read_data_shakespeare
import torch
import pandas as pd
import lstm
import rnn
import seaborn as sns
import hyperParameters
import matplotlib.pyplot as plt


class Run:

    def __init__(self):
        self.data_dict = read_data_shakespeare()
        self.train_text = self.data_dict["train_text"]
        self.test_text = self.data_dict["test_text"]
        self.index2char = self.data_dict["index2char"]
        self.char2index = self.data_dict["char2index"]

    def run_lstm(self, hyper_params, save=True, print_every=500, return_model=False):

        lstm_gen = lstm.Generator(
        input_string=self.train_text,
        test_string=self.test_text,
        index2char=self.index2char,
        char2index=self.char2index,
        sequence_length=hyper_params.SEQUENCE_LENGTH,
        batch_size=hyper_params.BATCH_SIZE
        )

        lstm_model = lstm.RNN(
            input_size=len(self.index2char),
            hidden_size=hyper_params.HIDDEN_SIZE,
            num_layers=hyper_params.NUM_LAYERS,
            output_size=len(self.index2char),
        ).to(lstm_gen.device)

        lstm_gen.train(
            lstm=lstm_model,
            num_epchs=hyper_params.NUM_EPOCHS,
            print_every=print_every,
            lr=hyper_params.LEARNING_RATE,
            temperature=hyper_params.TEMPERATURE,
            label_smoothing=hyper_params.LABEL_SMOOTHING,
        )

        if (return_model):
            return lstm_gen

        if (save):
            """
            Save LSTM model
            """
            torch.save(lstm_gen.lstm.state_dict(), f"{hyper_params.DIR}/lstm_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.pth")
            lstm_df = pd.DataFrame(lstm_gen.history)
            lstm_df.to_csv(f'{hyper_params.DIR}/lstm_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.csv', index=False)

            sns.set_style("whitegrid")
            title_string = f"LSTM: Loss vs iterations\nHidden Layers:{hyper_params.HIDDEN_SIZE}, lr:{hyper_params.LEARNING_RATE}"
            lstm_fig = sns.lineplot(data=lstm_df, x="iterations", y="loss").set_title(title_string)
            lstm_fig.get_figure().savefig(f'{hyper_params.DIR}/lstm_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.png')
            """
            To load model to cuda but it was saved on cpu (or vice versa) use:
            device = torch.device("cuda")
            lstm = lstm.RNN(*args, **kwargs)
            lstm.load_state_dict(torch.loadc(PATH, map_location=device))
            """

    def run_rnn(self, hyper_params):
        rnn_gen =  rnn.Generator(
            input_string=self.train_text,
            test_string=self.test_text,
            index2char=self.index2char,
            char2index=self.char2index,
            sequence_length=hyper_params.SEQUENCE_LENGTH,
            batch_size=hyper_params.BATCH_SIZE
            )

        rnn_model =  rnn.RNN(
            input_size=len(self.index2char),
            hidden_size=hyper_params.HIDDEN_SIZE,
            num_layers=hyper_params.NUM_LAYERS,
            output_size=len(self.index2char),
        ).to(rnn_gen.device)

        rnn_gen.train(
            rnn=rnn_model,
            num_epchs=hyper_params.NUM_EPOCHS,
            print_every=500,
            lr=hyper_params.LEARNING_RATE,
            temperature=hyper_params.TEMPERATURE,
            label_smoothing=hyper_params.LABEL_SMOOTHING,
        )

        """
        Save RNN model
        """
        torch.save(rnn_model.rnn.state_dict(), f"{hyper_params.DIR}/rnn_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.pth")
        rnn_df = pd.DataFrame(rnn_gen.history)
        rnn_df.to_csv(f"{hyper_params.DIR}/rnn_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.csv", index=False)

        sns.set_style("whitegrid")
        title_string = f"RNN: Loss vs iterations\nHidden Layers:{hyper_params.HIDDEN_SIZE}, lr:{hyper_params.LEARNING_RATE}"
        rnn_fig = sns.lineplot(data=rnn_df, x="iterations", y="loss").set_title(title_string)
        rnn_fig.get_figure().savefig(f'{hyper_params.DIR}/rnn_hidden{hyper_params.HIDDEN_SIZE}_epoch{hyper_params.NUM_EPOCHS}_lr{hyper_params.LEARNING_RATE}_nlayer{hyper_params.NUM_LAYERS}.png')

# if __name__=='__main__':
#     h = hyperParameters.Hyper_params(hidden_size=500, num_epochs=10000, learning_rate=0.005, dir="../results/anton_test/learning_rate_0005")
#     run = Run()
#     run.run_lstm(hyper_params=h, save=True, print_every=100)  

# if __name__=='__main__':
    
#     hidden_25 = pd.read_csv("../results/anton_test/learning_rate_0005/lstm_hidden25_epoch10000_lr0.005_nlayer2.csv")
#     hidden_50 = pd.read_csv("../results/anton_test/learning_rate_0005/lstm_hidden50_epoch10000_lr0.005_nlayer2.csv")
#     hidden_250 = pd.read_csv("../results/anton_test/learning_rate_0005/lstm_hidden250_epoch10000_lr0.005_nlayer2.csv")
#     hidden_500 = pd.read_csv("../results/anton_test/learning_rate_0005/lstm_hidden500_epoch10000_lr0.005_nlayer2.csv")

#     hidden_25["Hidden Size"] = 25
#     hidden_50["Hidden Size"] = 50        
#     hidden_250["Hidden Size"] = 250      
#     hidden_500["Hidden Size"] = 500      

#     df = pd.concat([hidden_25, hidden_50, hidden_250, hidden_500], ignore_index=True, axis=0)

#     sns.set_style("whitegrid")

#     title_string = f"Loss vs iterations, lr:{0.005}"
#     line_plot = sns.lineplot(data=df, x="iterations", y="loss", hue="Hidden Size", palette=["r","b","g","orange"]).set_title(title_string)
#     line_plot.get_figure().savefig(f'combined_line_plot_lr{0.005}_nlayer{2}.png')

#     plt.figure()
#     title_string = f"Final loss, lr:{0.005}"
#     hidden_size = [25, 50, 250, 500]
#     loss = [hidden_25["loss"][100], hidden_50["loss"][100], hidden_250["loss"][100], hidden_500["loss"][100]]
#     bar_plot = sns.barplot(x=hidden_size, y=loss, palette=["r","b","g","orange"]).set_title(title_string)
#     plt.ylabel("loss")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'combined_bar_plot_lr{0.005}_nlayer{2}.png')


# if __name__=='__main__':
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     temp_check = pd.read_csv("../results/score_check/hidden_check_temp_09.csv")
#     SAVE_DIR = "../results/score_check/plots"
#     sns.set_style("whitegrid")
#     """
#     Get spelling percentage
#     """
#     plt.figure()
#     title_string = f"Spelling pecetange (the higher the better), temperature: {temp_check.temperature[0]}"
#     bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.spelling_percentage, palette=["r","b","g","orange"]).set_title(title_string)
#     plt.ylabel("Spelling Percentage")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_spelling_pecetange_temp_{temp_check.temperature[0]}.png')

#     plt.figure()
#     title_string = f"Perplexity score (the lower the better), temperature: {temp_check.temperature[0]}"
#     bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.perplexity, palette=["r","b","g","orange"]).set_title(title_string)
#     plt.ylabel("Perplexity Score")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_perplexity_temp_{temp_check.temperature[0]}.png')


#     plt.figure()
#     title_string = f"Bartscore (the higher the better), temperature: {temp_check.temperature[0]}"
#     bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.bartscore, palette=["r","b","g","orange"]).set_title(title_string)
#     plt.ylabel("Bartscore")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bartscore_temp_{temp_check.temperature[0]}.png')

#     bleu1 = temp_check[["temperature", "hidden_size", "bleu1"]] 
#     bleu1["bleu type"] = "bleu1"
#     bleu1 = bleu1.rename(columns={"bleu1":"bleu"})

#     bleu2 = temp_check[["temperature", "hidden_size", "bleu2"]] 
#     bleu2["bleu type"] = "bleu2"
#     bleu2 = bleu2.rename(columns={"bleu2":"bleu"})

#     bleu3 = temp_check[["temperature", "hidden_size", "bleu3"]] 
#     bleu3["bleu type"] = "bleu3"
#     bleu3 = bleu3.rename(columns={"bleu3":"bleu"})

#     bleu4 = temp_check[["temperature", "hidden_size", "bleu4"]] 
#     bleu4["bleu type"] = "bleu4"
#     bleu4 = bleu4.rename(columns={"bleu4":"bleu"})

#     bleu_df = pd.concat([bleu1, bleu2, bleu3, bleu4])

#     plt.figure()
#     title_string = f"BLEU Score, temperature (the higher the better): {temp_check.temperature[0]}"
#     bar_plot = sns.barplot(x="hidden_size", y="bleu", data=bleu_df, hue="bleu type", palette="crest").set_title(title_string)
#     plt.ylabel("BLEU Score")
#     plt.xlabel("Hidden Size")
#     plt.legend(bbox_to_anchor=(0.93, 1), loc='upper left', borderaxespad=0)
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bleu_temp_{temp_check.temperature[0]}.png')

#     bertscore = temp_check[["temperature", "hidden_size", "bertscore"]] 
#     bertscore["bertscore type"] = "bertscore"
#     bertscore = bertscore.rename(columns={"bertscore":"bert"})

#     bertscore_precision = temp_check[["temperature", "hidden_size", "bertscore_precision"]] 
#     bertscore_precision["bertscore type"] = "bertscore_precision"
#     bertscore_precision = bertscore_precision.rename(columns={"bertscore_precision":"bert"})

#     bertscore_recall = temp_check[["temperature", "hidden_size", "bertscore_recall"]] 
#     bertscore_recall["bertscore type"] = "bertscore_recall"
#     bertscore_recall = bertscore_recall.rename(columns={"bertscore_recall":"bert"})

#     bertscore_f1 = temp_check[["temperature", "hidden_size", "bertscore_f1"]] 
#     bertscore_f1["bertscore type"] = "bertscore_f1"
#     bertscore_f1 = bertscore_f1.rename(columns={"bertscore_f1":"bert"})

#     bertscore = pd.concat([bertscore, bertscore_precision, bertscore_recall, bertscore_f1])

#     plt.figure()
#     title_string = f"BertScore, temperature (the higher the better): {temp_check.temperature[0]}"
#     bar_plot = sns.barplot(x="hidden_size", y="bert", data=bertscore, hue="bertscore type", palette="crest").set_title(title_string)
#     plt.legend(bbox_to_anchor=(0.75, 0.3), loc='upper left', borderaxespad=0)
#     plt.ylabel("Bert Score")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bert_types_temp_{temp_check.temperature[0]}.png')
    

