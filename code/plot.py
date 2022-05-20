import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics():
    import pandas as pd
    import matplotlib.pyplot as plt
    this_temp = 0.7
    temp_check = pd.read_csv("../results/score_check/metric_check.csv")
    temp_check = temp_check[temp_check["temperature"]==this_temp]
    SAVE_DIR = "../results/score_check/plots"
    sns.set_style("whitegrid")
    """
    Get spelling percentage
    """
    plt.figure()
    title_string = f"Spelling pecetange (the higher the better), temperature: {this_temp}"
    bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.spelling_percentage, palette=["r","b","g","orange"]).set_title(title_string)
    plt.ylabel("Spelling Percentage")
    plt.xlabel("Hidden Size")
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_spelling_pecetange_temp_{this_temp}.png')

    plt.figure()
    title_string = f"Perplexity score (the lower the better), temperature: {this_temp}"
    bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.perplexity, palette=["r","b","g","orange"]).set_title(title_string)
    plt.ylabel("Perplexity Score")
    plt.xlabel("Hidden Size")
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_perplexity_temp_{this_temp}.png')

    plt.figure()
    title_string = f"TTR score (the higher the better), temperature: {this_temp}"
    bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.TTR, palette=["r","b","g","orange"]).set_title(title_string)
    plt.ylabel("TTR Score")
    plt.xlabel("Hidden Size")
    bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_TTR_temp_{this_temp}.png')

    plt.figure()
    title_string = f"Bartscore (the higher the better), temperature: {this_temp}"
    bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.bartscore, palette=["r","b","g","orange"]).set_title(title_string)
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
    


# if __name__ == '__main__':
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     this_temp = 0.7
#     temp_check = pd.read_csv("../results/score_check/metric_check.csv")
#     temp_check = temp_check[temp_check["temperature"]==this_temp]
#     SAVE_DIR = "../results/score_check/plots"
#     sns.set_style("whitegrid")
#     """
#     Get spelling percentage
#     """
#     plt.figure()
#     title_string = f"Spelling pecetange (the higher the better), temperature: {this_temp}"
#     bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.spelling_percentage, palette=["r","b","g","orange"]).set_title(title_string)
#     plt.ylabel("Spelling Percentage")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_spelling_pecetange_temp_{this_temp}.png')

#     plt.figure()
#     title_string = f"Perplexity score (the lower the better), temperature: {this_temp}"
#     bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.perplexity, palette=["r","b","g","orange"]).set_title(title_string)
#     plt.ylabel("Perplexity Score")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_perplexity_temp_{this_temp}.png')

#     plt.figure()
#     title_string = f"TTR score (the lower the better), temperature: {this_temp}"
#     bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.TTR, palette=["r","b","g","orange"]).set_title(title_string)
#     plt.ylabel("TTR Score")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_plot_TTR_temp_{this_temp}.png')

#     plt.figure()
#     title_string = f"Bartscore (the higher the better), temperature: {this_temp}"
#     bar_plot = sns.barplot(x=temp_check.hidden_size, y=temp_check.bartscore, palette=["r","b","g","orange"]).set_title(title_string)
#     plt.ylabel("Bartscore")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bartscore_temp_{this_temp}.png')

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
#     title_string = f"BLEU Score, temperature (the higher the better): {this_temp}"
#     bar_plot = sns.barplot(x="hidden_size", y="bleu", data=bleu_df, hue="bleu type", palette="crest").set_title(title_string)
#     plt.ylabel("BLEU Score")
#     plt.xlabel("Hidden Size")
#     plt.legend(bbox_to_anchor=(0.93, 1), loc='upper left', borderaxespad=0)
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bleu_temp_{this_temp}.png')

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
#     title_string = f"BertScore, temperature (the higher the better): {this_temp}"
#     bar_plot = sns.barplot(x="hidden_size", y="bert", data=bertscore, hue="bertscore type", palette="crest").set_title(title_string)
#     plt.legend(bbox_to_anchor=(0.75, 0.3), loc='upper left', borderaxespad=0)
#     plt.ylabel("Bert Score")
#     plt.xlabel("Hidden Size")
#     bar_plot.get_figure().savefig(f'{SAVE_DIR}/bar_bert_types_temp_{this_temp}.png')
    
