"""Main to run training and test"""

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import warnings
import torch
from torchvision import transforms
from models.resnet import ResNet10
from utils.train_test_CIFAR import train, test
from utils.CIFAR100_dataset import BinaryUnbalancedCIFAR100, BinaryBalancedCIFAR100
from utils.final_ensembling_results import (
    final_ensembling_results,
    validation_ensembling,
)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: ", device)

    ########################################
    # Set parameters
    ########################################

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=int, default=1, help="whether to run also train"
    )
    parser.add_argument(
        "--name_file_bs", type=str, default="", help="json file of the configuration"
    )
    parser.add_argument(
        "--name_file_alm", type=str, default="", help="json file of the configuration"
    )
    parser.add_argument("--model_name", type=str, default="prova", help="model name")
    parser.add_argument(
        "--subfolder_name", type=str, default="prova", help="subfolder name"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="cifar10", help="subfolder name"
    )
    parser.add_argument("--bsize", type=int, default=64, help="set batch size")
    parser.add_argument(
        "--lr_decrease", type=int, default=0, help="decrease learning rate"
    )
    parser.add_argument(
        "--patience_lr",
        type=int,
        default=10,
        help="change lr at AUC plateau after patience",
    )
    parser.add_argument(
        "--patience_early_stopping",
        type=int,
        default=30,
        help="after this patience do early stopping",
    )
    parser.add_argument(
        "--continue_train",
        type=int,
        default=0,
        help="continue training from a pretrained model",
    )
    parser.add_argument(
        "--continue_train_epochs",
        type=int,
        default=50,
        help="how long continuing training for",
    )
    parser.add_argument(
        "--baseline_cont_train",
        type=str,
        default="prova",
        help="name of the baseline to continue the training",
    )
    parser.add_argument(
        "--ratio_pos_train",
        type=int,
        default=100,
        help="percentage of class1 samples in training set",
    )
    parser.add_argument("--hot_enc", type=int, default=0, help="hot enc set")
    parser.add_argument("--num_epochs", type=int, default=100, help="num epochs")
    parser.add_argument(
        "--resnet_type", type=str, default="resnet10", help="type of renset to be used"
    )
    # Objective function parameters
    parser.add_argument(
        "--training_type",
        type=str,
        help="Loss function type. \
                        Options: 'ASYM_LM', 'SYM_LM', 'ASYM_FL', 'SYM_FL', 'WBCE', \
                        'cb_BCE', 'MBAUC', otherwise 'BCE'",
    )
    parser.add_argument("--gamma", help="set gamma hyperparam for FL")
    parser.add_argument("--m", type=float, help="set m hyperparam for LM")
    parser.add_argument("--c", type=float, help="set c hyperparam for WBCE")
    parser.add_argument("--beta", type=float, help="set c hyperparam for WBCE")
    # ALM parameters
    parser.add_argument("--use_ALM", type=int, default=0, help="use ALM")
    parser.add_argument(
        "--introd_ALM_from", type=int, default=0, help="introduce ALM at this epoch"
    )
    parser.add_argument("--p2_norm", type=float, default=1, help="2p norm for ALM")
    parser.add_argument("--mu", type=float, default=0.1, help="set initial mu")
    parser.add_argument(
        "--mu_limit", type=float, default=0.06, help="maximum mu allowed"
    )
    parser.add_argument(
        "--rho", type=int, default=2, help="coefficient in ALM for squared penalty term"
    )
    parser.add_argument("--delta", type=float, default=0, help="margin to improve ALM")
    parser.add_argument("--cap_mu", type=int, default=1, help="set capping of mu")
    parser.add_argument(
        "--weight_dec", type=float, default=0, help="weight decay of optimizer"
    )
    args = parser.parse_args()
    args = vars(args)

    ########################################
    # Initialize all the useful variables
    ########################################
    # Dataset settings
    path_to_dataset = "datasets/" + args["dataset_name"] + "/"
    print(
        "You have selected dataset {} and network {}. ".format(
            args["dataset_name"], args["resnet_type"]
        )
    )
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_val_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    if args["dataset_name"] == "cifar100":
        new_majority_labels = np.array(
            [72, 4, 95, 30, 55]
        )  # Superclass fishes - to reproduce the paper
        new_minority_labels = np.array([73])  # Subclass sharks - to reproduce the paper
        val_dataset = BinaryBalancedCIFAR100(
            root=path_to_dataset,
            first_class_labels=new_majority_labels,
            second_class_labels=new_minority_labels,
            val_samples_per_cls=50,
            train=True,
            download=False,
            transform=transform_val_test,
        )
        test_dataset = BinaryBalancedCIFAR100(
            root=path_to_dataset,
            first_class_labels=new_majority_labels,
            second_class_labels=new_minority_labels,
            val_samples_per_cls=100,
            train=False,
            download=False,
            transform=transform_val_test,
        )
    else:
        warnings.warn("Dataset is not listed")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize parameters
    seeds = np.arange(1, 11)
    max_validation_auc_runs = np.zeros(len(seeds))
    val_matrix_logits = np.zeros((len(val_loader.dataset), len(seeds)))
    test_matrix_logits = np.zeros((len(test_loader.dataset), len(seeds)))
    test_auc_array = np.zeros(10)
    num_classes = 2

    # Read config. files
    with open(os.path.join("configs/baseline/", args["name_file_bs"] + ".json")) as f:
        args_json_bs = json.load(f)
    print("JSON file baseline:", args["name_file_bs"])
    args.update(args_json_bs)

    if args["name_file_alm"] != "":
        with open(os.path.join("configs/ALM/", args["name_file_alm"] + ".json")) as f:
            args_json_alm = json.load(f)
        print("JSON file ALM:", args["name_file_alm"])
        args.update(args_json_alm)
        if args["mu"] < 1e-3:
            args["mu_limit"] = args["mu"] * args["rho"] ** 5
        else:
            args["mu_limit"] = args["mu"] * args["rho"] ** 3

    MODEL_NAME = args["model_name"]
    print(
        "Model name {}, ratio classes {} training, epochs {}, LR decrease {}.".format(
            MODEL_NAME, args["ratio_pos_train"], args["num_epochs"], args["lr_decrease"]
        )
    )

    for seed in seeds:
        print("Training model: {}. Run num. {}.".format(MODEL_NAME, seed))
        print("Training type", args["training_type"])
        if args["use_ALM"] == 1:
            print("ALM selected.")
            print(
                "ALM params: rho={}, mu={}, delta={}".format(
                    args["rho"], args["mu"], args["delta"]
                )
            )
        else:
            print("ALM not selected.")
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        MODEL_DIR = os.path.join(
            "checkpoint", args["subfolder_name"], MODEL_NAME, SEED_STR
        )
        SEED_STR = "seed" + str(seed)
        os.makedirs(MODEL_DIR, exist_ok=True)

        if args["dataset_name"] == "cifar100":
            train_dataset = BinaryUnbalancedCIFAR100(
                root=path_to_dataset,
                imb_factor=1 / args["ratio_pos_train"],
                seed=seed,
                new_majority_labels=new_majority_labels,
                new_minority_labels=new_minority_labels,
                val_samples_per_cls=50,
                perc_per_run=0.9,
                train=True,
                download=False,
                transform=transform_train,
            )
            lr = 0.001
        else:
            warnings.warn("Dataset is not listed")

        per_class_samples = train_dataset.get_tot_classes_list(tot_classes=2)
        num_maj = len(train_dataset.targets) - sum(
            train_dataset.targets
        )  # class 0 majority
        use_norm = True if args["training_type"] == "LDAM" else False
        net = ResNet10(num_classes, use_norm=use_norm)
        net.to(device)

        #     ######################################
        #     #Train, Validate, Test
        #     ######################################
        if args["train"] == 1:
            max_validation_auc_runs[seed - 1], val_matrix_logits[:, seed - 1] = train(
                net,
                args["num_epochs"],
                args["subfolder_name"],
                MODEL_NAME,
                MODEL_DIR,
                seed,
                per_class_samples,
                args["bsize"],
                lr,
                args["lr_decrease"],
                args["patience_lr"],
                args["patience_early_stopping"],
                num_maj,
                args["ratio_pos_train"],
                args["training_type"],
                args["gamma"],
                args["m"],
                args["c"],
                args["beta"],
                args["continue_train"],
                args["continue_train_epochs"],
                args["baseline_cont_train"],
                args["use_ALM"],
                args["introd_ALM_from"],
                args["p2_norm"],
                train_dataset,
                val_loader,
                args["rho"],
                args["mu"],
                args["mu_limit"],
                args["cap_mu"],
                args["delta"],
                device,
            )

        test_auc_array[seed - 1] = test(
            net,
            test_loader,
            MODEL_DIR,
            MODEL_NAME,
            args["training_type"],
            device,
        )

        # To keep track of the results on test set either unbalanced in the same way as training or balanced
        df_fv = pd.read_csv(
            os.path.join(MODEL_DIR, MODEL_NAME + "_LOGITS_and_GT_fv.csv")
        )
        logits = df_fv["Z"].to_numpy()
        test_matrix_logits[:, seed - 1] = logits

    # Calculate and print final results
    validation_ensembling(val_matrix_logits, val_loader.dataset.targets)

    final_ensembling_results(
        args["subfolder_name"],
        MODEL_DIR,
        MODEL_NAME,
        test_matrix_logits,
        test_auc_array,
        max_validation_auc_runs,
    )


if __name__ == "__main__":
    main()
