"""ROC plotting for attack reports."""
import matplotlib.pyplot as plt
import numpy as np


def plot_roc(fpr_list, tpr_list, roc_auc, path, log=False):
    range01 = np.linspace(0, 1)
    plt.fill_between(fpr_list, tpr_list, alpha=0.15)
    plt.plot(range01, range01, "--", label="Random guess")
    plt.plot(fpr_list, tpr_list, label="ROC curve")
    if log:
        plt.xlim([10e-6, 1])
        plt.ylim([10e-6, 1])
        plt.xscale("log")
        plt.yscale("log")
    else:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("ROC curve")
    plt.text(
        0.7,
        0.3,
        f"AUC = {roc_auc:.03f}",
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.savefig(
        fname=path,
        dpi=1000,
    )
    plt.clf()
