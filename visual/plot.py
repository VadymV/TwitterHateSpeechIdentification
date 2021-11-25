import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix

def plot_cm(conf_matrix, plot_file_name, title):
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix,
                                    figsize=(3.5, 3.5),
                                    show_absolute=True,
                                    show_normed=True,
                                    colorbar=True)
    plt.xlabel('Predicted class', fontsize=14)
    plt.ylabel('True class', fontsize=14)
    plt.title(title, fontsize=14)
    plt.savefig("./output/{}.pdf".format(plot_file_name))
    plt.close()


def create_train_val_plot(data, plot_file_name, title):

    sns.set(font_scale=1.4)
    sns.set(rc={'figure.figsize': (4, 4)})
    l_plot = sns.lineplot(data=data, x="epoch", y="loss", hue="set_id", style="set_id", ci=95)
    l_plot.axes.set_title(title, fontsize=20)
    l_plot.set_xlabel("Epoch", fontsize=20)
    l_plot.set_ylabel("Loss", fontsize=20)
    l_plot.tick_params(labelsize=14)
    l_plot.legend_.set_title(None)
    #l_plot.set(xticks=[1,2,3,4,5,6,7,8,9,10])
    plt.gcf().subplots_adjust(bottom=0.2, left=0.25)
    plt.savefig("./output/{}.pdf".format(plot_file_name))
    plt.close()


def plot_pdf_y(data, file_name):
    sns.set(rc={'figure.figsize': (4, 4)})
    data = data.rename(columns={"actuals_y": "Tweet label"})
    ax = sns.displot(data, x="preds_y", hue="Tweet label", kind="kde", fill=True)
    plt.xlabel('Predictions', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axvline(0, color='red')
    plt.savefig("./output/{}.pdf".format(file_name))
    plt.close()


def plot_pdf_x(data, file_name):
    sns.set(rc={'figure.figsize': (4, 4)})
    data = data.rename(columns={"actuals_x": "Tweet label"})
    ax = sns.displot(data, x="preds_x", hue="Tweet label", kind="kde", fill=True)
    plt.xlabel('Predictions', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axvline(0, color='red')
    plt.savefig("./output/{}.pdf".format(file_name))
    plt.close()

