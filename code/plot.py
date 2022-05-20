import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metrics(this_temp):
    import matplotlib.pyplot as plt
    import pandas as pd

    temp_check = pd.read_csv("../results/score_check/metric_check.csv")
    temp_check = temp_check[temp_check["temperature"]==this_temp]
    SAVE_DIR = "../results/score_check/plots"

    sns.set_style("whitegrid")
    """
    Get spelling percentage
    """
    plt.figure()
    title_string = f"Spelling pecetange (the higher the better), temperature: {this_temp}"
    bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.spelling_percentage, palette="crest").set_title(title_string)
    plt.ylabel("Spelling Percentage")
    plt.xlabel("Hidden Size")
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_spelling_pecetange_temp_{this_temp}.png')

    plt.figure()
    title_string = f"Perplexity score (the lower the better), temperature: {this_temp}"
    bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.perplexity, palette="crest").set_title(title_string)
    plt.ylabel("Perplexity Score")
    plt.xlabel("Hidden Size")
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_perplexity_temp_{this_temp}.png')

    plt.figure()
    title_string = f"TTR score (the higher the better), temperature: {this_temp}"
    bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.TTR, palette="crest").set_title(title_string)
    plt.ylabel("TTR Score")
    plt.xlabel("Hidden Size")
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_TTR_temp_{this_temp}.png')

    plt.figure()
    title_string = f"Bartscore (the higher the better), temperature: {this_temp}"
    bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.bartscore, palette="crest").set_title(title_string)
    plt.ylabel("Bartscore")
    plt.xlabel("Hidden Size")
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bartscore_temp_{this_temp}.png')

    bleu1 = temp_check[["temperature", "hidden_size", "bleu1"]] 
    bleu1["bleu type"] = "bleu1"
    bleu1 = bleu1.rename(columns={"bleu1":"bleu"})

    bleu2 = temp_check[["temperature", "hidden_size", "bleu2"]] 
    bleu2["bleu type"] = "bleu2"
    bleu2 = bleu2.rename(columns={"bleu2":"bleu"})

    bleu3 = temp_check[["temperature", "hidden_size", "bleu3"]] 
    bleu3["bleu type"] = "bleu3"
    bleu3 = bleu3.rename(columns={"bleu3":"bleu"})

    bleu4 = temp_check[["temperature", "hidden_size", "bleu4"]] 
    bleu4["bleu type"] = "bleu4"
    bleu4 = bleu4.rename(columns={"bleu4":"bleu"})

    bleu_df = pd.concat([bleu1, bleu2, bleu3, bleu4])

    plt.figure()
    title_string = f"BLEU Score, temperature (the higher the better): {this_temp}"
    bar_plot = sns.barplot(x="hidden_size", y="bleu", data=bleu_df, hue="bleu type", palette="crest").set_title(title_string)
    plt.ylabel("BLEU Score")
    plt.xlabel("Hidden Size")
    plt.legend(bbox_to_anchor=(0.93, 1), loc='upper left', borderaxespad=0)
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bleu_temp_{this_temp}.png')

    bertscore = temp_check[["temperature", "hidden_size", "bertscore"]] 
    bertscore["bertscore type"] = "bertscore"
    bertscore = bertscore.rename(columns={"bertscore":"bert"})

    bertscore_precision = temp_check[["temperature", "hidden_size", "bertscore_precision"]] 
    bertscore_precision["bertscore type"] = "bertscore_precision"
    bertscore_precision = bertscore_precision.rename(columns={"bertscore_precision":"bert"})

    bertscore_recall = temp_check[["temperature", "hidden_size", "bertscore_recall"]] 
    bertscore_recall["bertscore type"] = "bertscore_recall"
    bertscore_recall = bertscore_recall.rename(columns={"bertscore_recall":"bert"})

    bertscore_f1 = temp_check[["temperature", "hidden_size", "bertscore_f1"]] 
    bertscore_f1["bertscore type"] = "bertscore_f1"
    bertscore_f1 = bertscore_f1.rename(columns={"bertscore_f1":"bert"})

    bertscore = pd.concat([bertscore, bertscore_precision, bertscore_recall, bertscore_f1])

    plt.figure()
    title_string = f"BertScore, temperature (the higher the better): {this_temp}"
    bar_plot = sns.barplot(x="hidden_size", y="bert", data=bertscore, hue="bertscore type", palette="crest").set_title(title_string)
    plt.legend(bbox_to_anchor=(0.75, 0.3), loc='upper left', borderaxespad=0)
    plt.ylabel("Bert Score")
    plt.xlabel("Hidden Size")
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bert_types_temp_{this_temp}.png')
    
def plot_string():
    import matplotlib.pyplot as plt
    import pandas as pd

    strings = pd.read_csv("../results/score_check/str_check.csv")
    labels = ['1','2','3']
    strings['label'] = labels
    #print(strings.head)
    # original = strings.loc[0]
    # repetitions = strings.loc[1]
    # random = strings.loc[2]
    # #print(random)
    # string_2 = pd.read_csv("../results/score_check/metric_check.csv")
    # string_3 = pd.read_csv("../results/score_check/metric_check.csv")
    #labels = ['original','repetition','random']
    
    
    #temp_check = temp_check[temp_check["temperature"]==this_temp]
    SAVE_DIR = "../results/score_check/plots/text_validate"

    xlabel = 'generated text number'

    sns.set_style("whitegrid")
    """
    Get spelling percentage
    """
    plt.figure()
    title_string = f"Spelling pecetange (the higher the better)"
    bar_plot = sns.barplot(x=labels, y=strings['spelling_percentage'], palette="crest").set_title(title_string)
    plt.ylabel("Spelling Percentage")
    plt.xlabel(xlabel)
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_spelling_percentage.png')
    plt.show()

    plt.figure()
    title_string = f"Perplexity score (the lower the better)"
    bar_plot = sns.barplot(x=labels, y=strings["spelling_percentage"], palette="crest").set_title(title_string)
    plt.ylabel("Perplexity Score") 
    plt.xlabel(xlabel)
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_perplexity.png')

    plt.figure()
    title_string = f"TTR score (the higher the better)"
    bar_plot = sns.barplot(x=labels, y=strings["spelling_percentage"],palette="crest").set_title(title_string)
    plt.ylabel("TTR Score")
    plt.xlabel(xlabel)
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_TTR.png')

    plt.figure()
    title_string = f"Bartscore (the higher the better)"
    bar_plot = sns.barplot(x=labels, y=strings["spelling_percentage"], palette="crest").set_title(title_string)
    plt.ylabel("Bartscore")
    plt.xlabel(xlabel)
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bartscore.png')


    l = ["bleu1" for i in range(3)]
    # print('l:', l)
    # penguins = sns.load_dataset("penguins")
    # print(penguins.head())
    bleu1 = strings[["label", "bleu1"]].copy() 
    bleu1["bleu type"] = ["bleu1" for i in range(3)]
    bleu1 = bleu1.rename(columns={"bleu1":"bleu"})

    bleu2 = strings[["label", "bleu2"]].copy() 
    bleu2["bleu type"] = ["bleu2" for i in range(3)]
    bleu2 = bleu2.rename(columns={"bleu2":"bleu"})

    bleu3 = strings[["label", "bleu3"]].copy() 
    bleu3["bleu type"] = ["bleu3" for i in range(3)]
    bleu3 = bleu3.rename(columns={"bleu3":"bleu"})

    bleu4 = strings[["label", "bleu4"]].copy() 
    bleu4["bleu type"] = ["bleu4" for i in range(3)]
    bleu4 = bleu4.rename(columns={"bleu4":"bleu"})

    bleu_df = pd.concat([bleu1, bleu2, bleu3, bleu4])
    print(bleu_df)

    plt.figure()
    title_string = f"BLEU Score (the higher the better)"
    bar_plot = sns.barplot(x=bleu_df.label, y=bleu_df.bleu, data=bleu_df, hue=bleu_df["bleu type"], palette="crest").set_title(title_string)
    plt.ylabel("BLEU Score")
    plt.xlabel(xlabel)
    plt.legend(bbox_to_anchor=(0.93, 1), loc='upper left', borderaxespad=0)
    plt.show()
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bleu.png')

    bertscore = strings[["label", "bertscore"]].copy() 
    bertscore["bertscore type"] = "bertscore"
    bertscore = bertscore.rename(columns={"bertscore":"bert"})

    bertscore_precision = strings[["label", "bertscore_precision"]].copy() 
    bertscore_precision["bertscore type"] = "bertscore_precision"
    bertscore_precision = bertscore_precision.rename(columns={"bertscore_precision":"bert"})

    bertscore_recall = strings[["label", "bertscore_recall"]].copy() 
    bertscore_recall["bertscore type"] = "bertscore_recall"
    bertscore_recall = bertscore_recall.rename(columns={"bertscore_recall":"bert"})

    bertscore_f1 = strings[["label", "bertscore_f1"]].copy() 
    bertscore_f1["bertscore type"] = "bertscore_f1"
    bertscore_f1 = bertscore_f1.rename(columns={"bertscore_f1":"bert"})

    bertscore = pd.concat([bertscore, bertscore_precision, bertscore_recall, bertscore_f1])

    plt.figure()
    title_string = f"BertScore, temperature (the higher the better)"
    bar_plot = sns.barplot(x=bertscore.label, y=bertscore.bert, data=bertscore, hue="bertscore type", palette="crest").set_title(title_string)
    plt.legend(bbox_to_anchor=(0.75, 0.3), loc='upper left', borderaxespad=0)
    plt.ylabel("Bert Score")
    plt.xlabel(xlabel)
    plt.show()
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bert_types_temp.png')

if __name__ == '__main__':
    #plot_metrics(0.7)
    plot_string()
