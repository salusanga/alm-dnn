"""
    Function for final ensembling of results
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def validation_ensembling(val_matrix_logits, validation_labels):
    """
    Ensembling of validation.
    """
    val_ave_logits = np.average(val_matrix_logits, axis=1)
    fpr, tpr, _ = roc_curve(validation_labels, val_ave_logits, pos_label=1)
    val_AUC_ensembled = auc(fpr, tpr)
    print("### Final results ###")
    print("AUC of ensembled validation logits:", val_AUC_ensembled)


def final_ensembling_results(
    subfolder_name,
    model_dir,
    model_name,
    matrix_logits,
    test_auc_array,
    max_validation_auc_runs,
):
    """
    Function to calculate final results and print them
    """
    print(
        "Validation AUC average: {}, standard deviation: {}".format(
            np.average(max_validation_auc_runs), np.std(max_validation_auc_runs)
        )
    )

    # Analyse mean and average of results on test set
    df_fv = pd.read_csv(os.path.join(model_dir, model_name + "_LOGITS_and_GT_fv.csv"))
    gt_labels = df_fv["GT"].to_numpy()
    ave_logits = np.average(matrix_logits, axis=1)
    fpr, tpr, _ = roc_curve(gt_labels, ave_logits, pos_label=1)
    roc_auc = auc(fpr, tpr)

    print("AUC of averaged logits of final test: {}".format(roc_auc * 100))
    tpr_int_list = [0.98, 0.95, 0.9]
    for tpr_int in tpr_int_list:
        if fpr[tpr == tpr_int].size != 0:
            avg_interesting_fpr = (fpr[tpr == tpr_int])[0]
        elif fpr[np.where(np.around(tpr, decimals=2) == tpr_int)].size != 0:
            avg_interesting_fpr = (
                fpr[np.where(np.around(tpr, decimals=2) == tpr_int)]
            )[0]
        elif (
            fpr[np.where(np.around(tpr, decimals=2) == tpr_int - 0.01)].size != 0
            and fpr[np.where(np.around(tpr, decimals=2) == tpr_int + 0.01)].size != 0
        ):
            avg_interesting_fpr = (
                (fpr[np.where(np.around(tpr, decimals=2) == tpr_int - 0.01)])[0]
                + (fpr[np.where(np.around(tpr, decimals=2) == tpr_int + 0.01)])[0]
            ) / 2
        else:
            avg_interesting_fpr = 0
        print("TPR = {}, FPR = {}.".format(tpr_int * 100, avg_interesting_fpr * 100))

    print(
        "Test averaged AUC: {}, std {}.".format(
            np.mean(test_auc_array), np.std(test_auc_array)
        )
    )
    # Save avg logits and ROC plot of avg logits
    np.savetxt(
        os.path.join(
            "checkpoint/", subfolder_name, model_name, model_name + "_avg_logits_fv.csv"
        ),
        np.c_[gt_labels, ave_logits],
        delimiter=",",
        header="GT,AvgPred",
        comments="",
    )
    plt.figure()
    LW = 2
    roc_auc = roc_auc * 100
    plt.plot(
        fpr, tpr, color="darkorange", lw=LW, label="ROC curve (area = %.1f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=LW, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title("Receiver operating characteristic of averaged test logits")
    plt.legend(loc="lower right")
    plt.grid(ls="--", lw=0.5, markevery=0.1)
    plt.savefig(
        os.path.join(
            "checkpoint/", subfolder_name, model_name, model_name + "_ROC_avg_fv.pdf"
        )
    )
