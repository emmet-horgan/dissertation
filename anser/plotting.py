import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import re
import numpy as np
from sklearn.metrics import auc


def argparse_setup():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--show", "-s",
        help="Show the plot",
        action="store_true",
    )
    parser.add_argument(
        "--path", "-p",
        help="Path to the folder",
        action="store",
        type=str,
        default="."
    )
    parser.add_argument(
        "--save",
        help="Save the plot as a png",
        action="store_true",
    )
    parser.add_argument(
        "--window-size",
        help="Choose window size for moving average filter",
        action="store",
        type=int,
        default=10
    )
    parser.add_argument(
        "--type",
        help="Choose the type of plot",
        action="store",
        type=str,
        default="single"
    )
    parser.add_argument(
        "--average",
        help="Choose the type of plot",
        action="store",
        type=int,
        default=-1
    )
    return parser


def read(path, window=20):
    data = pd.read_csv(path, delimiter=",", dtype=float, header=None)
    average = data.iloc[:, 1].rolling(window=window).mean()
    return data, average


def plot_kfolds(path, window=20, titlespec=None, axesspec=None, epochonly=False, average_index=-1):
    if titlespec is None:
        titlespec = {
            "fontname": "DejaVu Sans",
            "fontsize": 14,
        }
    if axesspec is None:
        axesspec = {
            "fontname": "DejaVu Sans",
            "fontstyle": "oblique",
            "fontsize": 12,
        }

    regex = re.compile(r"(?P<set>[a-zA-z0-9]+)_(?P<y>[a-zA-Z0-9]+)_(?P<x>[a-zA-Z0-9]+)\.(?P<type>[a-zA-Z0-9]+)")
    figs = []
    figs_iter = None
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    accuracymean = []
    aucmean = []

    with os.scandir(path) as folds:
        for i, fold in enumerate(folds):
            if os.path.isfile(fold.path):
                continue
            here = os.getcwd()
            os.chdir(fold.path)
            with os.scandir("data") as d:
                for j, f in enumerate(d):
                    parsed = regex.search(f.name)
                    dataset = parsed.group("set").capitalize()
                    ylabel = parsed.group("y").capitalize()

                    if ylabel.upper() == "AUC":
                        ylabel = ylabel.upper()

                    xlabel = parsed.group("x").capitalize()
                    filetype = parsed.group("type")

                    if f.is_file() and filetype == "csv":
                        data = pd.read_csv(f.path, delimiter=",", dtype=float, header=None)
                        average = data.iloc[:, 1].rolling(window=window).mean()

                        title = " ".join([dataset, ylabel])
                        img_name = "_".join([dataset, ylabel, xlabel + ".png"]).lower()
                        label = "FOLD " + str(i + 1)

                        if i != 0:
                            plt.figure(next(figs_iter))
                        else:
                            fig = plt.figure()

                        plt.plot(data.iloc[:, 0], data.iloc[:, 1], color=colors[i], label=label)
                        #plt.plot(data.iloc[:, 0], average, color=colors[i], label=label)
                        plt.title(title, **titlespec)
                        if epochonly:
                            plt.xlabel("Epoch", **axesspec)
                        else:
                            plt.xlabel(xlabel, **axesspec)

                        plt.ylabel(ylabel, **axesspec)
                        if i == 0:
                            figs.append(fig)
                        if ylabel.upper() == "ACCURACY" or ylabel.upper() == "AUC":
                            mean = data.iloc[average_index, 1]
                            if dataset.upper() == "TEST":
                                if ylabel.upper() == "ACCURACY":
                                    accuracymean.append(mean)
                                else:
                                    aucmean.append(mean)
                            plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                                        0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
                            plt.ylim([0.0, 1.0])
                            plt.plot(data.iloc[:, 0], [0.5] * len(data.iloc[:, 0]), '--', color="black")
                            plt.axvline(linestyle="--", x=50, color="black")
                        plt.close()

            figs_iter = iter(figs)
            os.chdir(here)
        print("AUC MEANS:", aucmean)
        print(f"AUC MEAN: ", np.array(aucmean).mean())
        print(f"AUC STDDEV: ", np.array(aucmean).std())

        print(f"ACCURACY MEANS: ", accuracymean)
        print(f"ACCURACY MEAN: ", np.array(accuracymean).mean())
        print(f"ACCURACY STDDEV: ", np.array(accuracymean).std())
        for fig in figs:
            plt.figure(fig)
            plt.grid(True)
            plt.legend(prop={"size": 6})
            plt.show()


def plot_roc(fpr, tpr, aucscores, eval_data, path, titlespec=None, axesspec=None, show=False):
    if titlespec is None:
        titlespec = {
            "fontname": "DejaVu Sans",
            "fontsize": 14,
        }
    if axesspec is None:
        axesspec = {
            "fontname": "DejaVu Sans",
            "fontstyle": "oblique",
            "fontsize": 12,
        }
    print("AUC AVERAGED = ", aucscores.mean().item())
    print("AUC ROC AVERAGE = ", auc(fpr, tpr))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.plot(fpr, tpr, label=f"AUC = {aucscores.mean().item():.3f}\nSTDEV = {aucscores.std().item():.3f}")
    plt.legend(loc="best")
    plt.plot([0, 1], [0, 1], 'b--')
    for i, (f, t, thresh) in enumerate(eval_data):
        plt.plot(f, t, alpha=0.2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("Receiver Operating Curve", **titlespec)
    plt.ylabel('True Positive Rate', **axesspec)
    plt.xlabel('False Positive Rate', **axesspec)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "roc.png"))
    if show:
        plt.show()


def plot_comp(path1, path2, label1, label2, window=20, titlespec=None, axesspec=None):
    if titlespec is None:
        titlespec = {
            "fontname": "DejaVu Sans",
            "fontsize": 14,
        }
    if axesspec is None:
        axesspec = {
            "fontname": "DejaVu Sans",
            "fontstyle": "oblique",
            "fontsize": 12,
        }

    regex = re.compile(r"(?P<set>[a-zA-z0-9]+)_(?P<y>[a-zA-Z0-9]+)_(?P<x>[a-zA-Z0-9]+)\.(?P<type>[a-zA-Z0-9]+)")
    figs = []
    figs_iter = None
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for index, path in enumerate([path1, path2]):
        here = os.getcwd()
        os.chdir(path)
        with os.scandir("data") as d:
            for i, f in enumerate(d):
                parsed = regex.search(f.name)
                dataset = parsed.group("set").capitalize()
                ylabel = parsed.group("y").capitalize()
                xlabel = parsed.group("x").capitalize()
                filetype = parsed.group("type")

                if f.is_file() and filetype == "csv":
                    data = pd.read_csv(f.path, delimiter=",", dtype=float, header=None)
                    average = data.iloc[:, 1].rolling(window=window).mean()

                    title = " ".join([dataset, ylabel])
                    img_name = "_".join([dataset, ylabel, xlabel + ".png"]).lower()
                    if ylabel.upper() == "LOSS":
                        ylabel = r"$\mathcal{L}$"
                    if index != 0:
                        label = label2
                        plt.figure(next(figs_iter))
                    else:
                        label = label1
                        fig = plt.figure()

                    plt.plot(data.iloc[:, 0], data.iloc[:, 1], color=colors[index], alpha=0.2)
                    plt.plot(data.iloc[:, 0], average, color=colors[index], label=label)

                    if index != 0:
                        plt.grid(True)
                        plt.title(title, **titlespec)
                        plt.xlabel(xlabel, **axesspec)
                        plt.ylabel(ylabel, **axesspec)
                        plt.legend()
                    else:
                        figs.append(fig)
                    plt.close()
        os.chdir(here)
        figs_iter = iter(figs)

    for fig in figs:
        plt.figure(fig)
        plt.show()


def plot(path=".", show=False, save=False, window=20, titlespec=None, axesspec=None):
    if titlespec is None:
        titlespec = {
            "fontname": "DejaVu Sans",
            "fontsize": 14,
        }
    if axesspec is None:
        axesspec = {
            "fontname": "DejaVu Sans",
            "fontstyle": "oblique",
            "fontsize": 12,
        }
    here = os.getcwd()
    os.chdir(path)

    regex = re.compile(r"(?P<set>[a-zA-z0-9]+)_(?P<y>[a-zA-Z0-9]+)_(?P<x>[a-zA-Z0-9]+)\.(?P<type>[a-zA-Z0-9]+)")
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    with os.scandir("data") as d:
        for i, f in enumerate(d):
            parsed = regex.search(f.name)
            dataset = parsed.group("set").capitalize()
            ylabel = parsed.group("y").capitalize()
            xlabel = parsed.group("x").capitalize()
            filetype = parsed.group("type")

            if f.is_file() and filetype == "csv":
                data = pd.read_csv(f.path, delimiter=",", dtype=float, header=None)
                average = data.iloc[:, 1].rolling(window=window).mean()

                title = " ".join([dataset, ylabel])
                img_name = "_".join([dataset, ylabel, xlabel + ".png"]).lower()
                if ylabel.upper() == "LOSS":
                    ylabel = r"$\mathcal{L}$"

                plt.plot(data.iloc[:, 0], data.iloc[:, 1], color=colors[i], label=title, alpha=0.2)
                plt.plot(data.iloc[:, 0], average, color=colors[i], label="Moving Average")

                plt.grid(True)
                plt.title(title, **titlespec)
                plt.xlabel(xlabel, **axesspec)
                plt.ylabel(ylabel, **axesspec)
                plt.legend()
                if save:
                    plt.savefig(os.path.join("data", img_name))

                if show:
                    plt.show()
                plt.close()

    os.chdir(here)


if __name__ == "__main__":
    parser = argparse_setup()
    args = parser.parse_args()
    plot(path=args.path, show=args.show, save=args.save, window=args.window_size)
