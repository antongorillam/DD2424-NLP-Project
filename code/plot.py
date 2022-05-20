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
    
def plot_string(this_temp):
    import matplotlib.pyplot as plt
    import pandas as pd

    string_1 = pd.read_csv("../results/score_check/original_str_benchmark.csv")
    string_1 = pd.read_csv("../results/score_check/metric_check.csv")
    string_2 = pd.read_csv("../results/score_check/metric_check.csv")
    string_3 = pd.read_csv("../results/score_check/metric_check.csv")
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

def lineplot_metrics(metric="temperature"):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("../results/score_check/metric_check.csv")
    SAVE_DIR = f"../results/score_check/plots/{metric}_lineplot"

    sns.set_style("whitegrid")
    """
    Get spelling percentage
    """
    plt.figure()
    title_string = f"Spelling pecetange (the higher the better)"
    lineplot = sns.lineplot(x=metric, y="spelling_percentage", data=df, hue=None, palette="crest", legend=True).set_title(title_string)
    plt.ylabel("Spelling Percentage")
    plt.xlabel(metric)
    lineplot.get_figure().savefig(f'{SAVE_DIR}/lineplot_spelling_pecetange_{metric}.png')

    plt.figure()
    title_string = f"Perplexity score (the lower the better)"
    lineplot = sns.lineplot(x=df[metric], y=df.perplexity, hue=None, palette="crest").set_title(title_string)
    plt.ylabel("Perplexity Score")
    plt.xlabel(metric)
    lineplot.get_figure().savefig(f'{SAVE_DIR}/lineplot_perplexity_{metric}.png')

    plt.figure()
    title_string = f"TTR score (the higher the better)"
    lineplot = sns.lineplot(x=df[metric], y=df.TTR, hue=None, palette="crest").set_title(title_string)
    plt.ylabel("TTR Score")
    plt.xlabel(metric)
    lineplot.get_figure().savefig(f'{SAVE_DIR}/lineplot_TTR_{metric}.png')

    plt.figure()
    title_string = f"Bartscore (the higher the better)"
    lineplot = sns.lineplot(x=df[metric], y=df.bartscore, hue=None, palette="crest").set_title(title_string)
    plt.ylabel("Bartscore")
    plt.xlabel(metric)
    lineplot.get_figure().savefig(f'{SAVE_DIR}/lineplot_bartscore_{metric}.png')

    bleu1 = df[[metric, "hidden_size", "bleu1"]] 
    bleu1["bleu type"] = "bleu1"
    bleu1 = bleu1.rename(columns={"bleu1":"bleu"})

    bleu2 = df[[metric, "hidden_size", "bleu2"]] 
    bleu2["bleu type"] = "bleu2"
    bleu2 = bleu2.rename(columns={"bleu2":"bleu"})

    bleu3 = df[[metric, "hidden_size", "bleu3"]] 
    bleu3["bleu type"] = "bleu3"
    bleu3 = bleu3.rename(columns={"bleu3":"bleu"})

    bleu4 = df[[metric, "hidden_size", "bleu4"]] 
    bleu4["bleu type"] = "bleu4"
    bleu4 = bleu4.rename(columns={"bleu4":"bleu"})
    bleu_df = pd.concat([bleu1, bleu2, bleu3, bleu4])

    plt.figure()
    title_string = f"BLEU Score, {metric} (the higher the better)"
    line_bleu1 = sns.lineplot(x=metric, y="bleu", data=bleu1, hue=None, palette="crest", legend=True).set_title(title_string)
    line_bleu2 = sns.lineplot(x=metric, y="bleu", data=bleu2, hue=None, palette="crest", legend=True).set_title(title_string)
    line_bleu3 = sns.lineplot(x=metric, y="bleu", data=bleu3, hue=None, palette="crest", legend=True).set_title(title_string)
    line_bleu4 = sns.lineplot(x=metric, y="bleu", data=bleu4, hue=None, palette="crest", legend=True).set_title(title_string)
    plt.ylabel("BLEU Score")
    plt.xlabel(metric)
    plt.legend(["bleu1", '_nolegend_', "bleu2", '_nolegend_', "bleu3", '_nolegend_', "bleu4", '_nolegend_',])
    plt.savefig(f'{SAVE_DIR}/lineplot_bleu_{metric}.png')
    

    plt.figure()
    bertscore = df[["temperature", "hidden_size", "bertscore"]] 
    bertscore["bertscore type"] = "bertscore"
    bertscore = bertscore.rename(columns={"bertscore":"bert"})

    bertscore_precision = df[["temperature", "hidden_size", "bertscore_precision"]] 
    bertscore_precision["bertscore type"] = "bertscore_precision"
    bertscore_precision = bertscore_precision.rename(columns={"bertscore_precision":"bert"})

    bertscore_recall = df[["temperature", "hidden_size", "bertscore_recall"]] 
    bertscore_recall["bertscore type"] = "bertscore_recall"
    bertscore_recall = bertscore_recall.rename(columns={"bertscore_recall":"bert"})

    bertscore_f1 = df[["temperature", "hidden_size", "bertscore_f1"]] 
    bertscore_f1["bertscore type"] = "bertscore_f1"
    bertscore_f1 = bertscore_f1.rename(columns={"bertscore_f1":"bert"})

        
    # bertscore = pd.concat([bertscore, bertscore_precision, bertscore_recall, bertscore_f1])
    plt.figure()
    title_string = f"BertScore (the higher the better)"
    sns.lineplot(x=metric, y="bert", data=bertscore, hue=None, palette="crest").set_title(title_string)
    # sns.lineplot(x=metric, y="bert", data=bertscore_precision, hue=None, palette="crest").set_title(title_string)
    # sns.lineplot(x=metric, y="bert", data=bertscore_recall, hue=None, palette="crest").set_title(title_string)
    # sns.lineplot(x=metric, y="bert", data=bertscore_f1, hue=None, palette="crest").set_title(title_string)
    
    plt.ylabel("Bert Score")
    plt.xlabel(metric)
    plt.savefig(f"{SAVE_DIR}/lineplot_bert_{metric}.png")
    # lineplot.get_figure().savefig(f'{SAVE_DIR}/lineplot_bert_types_{this_temp}.png')

if __name__ == '__main__':
    lineplot_metrics()
