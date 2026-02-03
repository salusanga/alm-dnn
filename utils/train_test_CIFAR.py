"""Functions to train/test models on CIFAR datasets"""

import os
import csv
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import torch
import torch.optim as optim
import torch.nn as nn
from .custom_losses import select_objective_function, ALMTermsClass


def train(
    net,
    num_epochs,
    subfolder_name,
    model_name,
    model_dir,
    seed,
    per_class_samples,
    bsize,
    lr,
    lr_decrease,
    patience_lr,
    patience_early_stopping,
    num_maj,
    ratio_pos_train,
    training_type,
    gamma,
    m,
    c,
    beta,
    continue_train,
    continue_train_epochs,
    cont_tr_bs_name,
    use_ALM,
    introd_ALM_from,
    p2_norm,
    trainset,
    val_loader,
    rho,
    mu,
    mu_limit,
    cap_mu,
    delta,
    device,
):
    """
    Train a neural network model on CIFAR datasets with various optimization settings.

    This function trains a neural network model on CIFAR datasets with options for different
    training strategies and optimization techniques.

    Parameters:
        net (nn.Module): The neural network model to be trained.
        num_epochs (int): The total number of training epochs.
        subfolder_name (str): Subfolder name for storing checkpoints and logs.
        model_name (str): Name of the model.
        model_dir (str): Directory where model checkpoints and logs will be saved.
        seed (int): Random seed for reproducibility.
        per_class_samples (int): Number of samples per class.
        bsize (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        lr_decrease (int): Whether to decrease learning rate on a plateau (0 or 1).
        patience_lr (int): Patience for learning rate scheduler.
        patience_early_stopping (int): Patience for early stopping.
        num_maj (int): Number of majority class samples.
        ratio_pos_train (float): Ratio of positive class samples in the training set.
        training_type (str): Type of training loss function (e.g., 'LDAM', 'MBAUC').
        gamma (float): Gamma parameter for the objective function.
        m (float): Margin parameter for the objective function.
        c (float): Constant for the objective function.
        beta (float): Beta parameter for the objective function.
        continue_train (int): Whether to continue training from a checkpoint (0 or 1).
        continue_train_epochs (int): Number of additional epochs for continued training.
        cont_tr_bs_name (str): Name of the baseline to continue the triaing from.
        use_ALM (int): Whether to use Augmented Lagrangian Method (ALM) for optimization (0 or 1).
        introd_ALM_from (int): Epoch from which to introduce ALM.
        p2_norm (float): Power parameter for the ALM penalty term.
        trainset (torch.utils.data.Dataset): Training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        rho (float): Rho parameter for updating the ALM multiplier.
        mu (float): Initial value of the ALM quadratic penalty term.
        mu_limit (float): Upper limit for the ALM multiplier.
        cap_mu (int): Whether to cap the ALM multiplier (0 or 1).
        delta (float): Delta parameter for ALM.

    Returns:
        float: Maximum validation AUC achieved during training.
        list: Logits of the validation set for the checkpointed model.
    """
    # Set objective function
    loss_function = select_objective_function(
        training_type,
        gamma,
        m,
        c,
        ratio_pos_train,
        num_maj,
        beta,
        delta,
        per_class_samples,
    )
    # Setup ALM class if needed
    if use_ALM == 1:
        ALM_terms_module = ALMTermsClass(delta, p2_norm, device)

    # Init variables useful to implement ALM
    lambdas = np.zeros(len(trainset))
    pos_train_samples = np.arange(len(trainset))
    stop_update_mu = 0
    buffer_batch_pos = []
    buffer_batch_neg = []
    lambdas_index_buffer = []
    # Other parameters
    start_epoch = 0
    avg_loss_train = 0
    val_loss_list = []
    val_penalty_list = []
    val_auc_list = []
    train_auc_list = []

    # Optimizer settings
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=2e-4, momentum=0.9)
    if lr_decrease == 1:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=patience_lr, verbose=True
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=1000
        )

    # Start from pre-trained model
    if continue_train:
        bs_losses_file = (
            "checkpoint/"
            + subfolder_name
            + cont_tr_bs_name
            + "/seed"
            + str(seed)
            + "/"
            + cont_tr_bs_name
            + "_train_loss_AUC_values.csv"
        )
        losses = pd.read_csv(bs_losses_file, sep=",")
        val_loss_list = losses["Val Loss"].tolist()
        val_auc_list = losses["Val AUC"].tolist()
        train_auc_list = losses["Train AUC"].tolist()
        net, optimizer, start_epoch, scheduler = load_checkpoint(
            net,
            optimizer,
            scheduler,
            +"/checkpoint/"
            + subfolder_name
            + cont_tr_bs_name
            + "/seed"
            + str(seed)
            + "/"
            + cont_tr_bs_name
            + ".pth",
        )
        num_epochs = start_epoch + continue_train_epochs
        introd_ALM_from = start_epoch - 1

    # Iterate over epochs
    for epoch in range(start_epoch, num_epochs):
        lambdas, trainset = shuffle_dataset(trainset, lambdas)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=bsize, shuffle=False
        )

        net.train()

        avg_loss_train = 0
        train_logits = []
        train_gt_labels = []
        for batch_idx, train_data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            bias_index = int(batch_idx * bsize)
            data, target = train_data
            data, target = data.to(device), target.to(device)
            train_out = net(data)

            # Get the tuples of positive and negative samples' indeces within a batch
            tuple_positive_index = (target).nonzero()
            tuple_negative_index = (target == 0).nonzero()
            # Get the tuples of positive and negative samples within a batch
            buffer_batch_pos = train_out[tuple_positive_index]
            buffer_batch_neg = train_out[tuple_negative_index]

            # Get the absolute positions of positive samples at that batch
            lambdas_index_buffer = pos_train_samples[
                bias_index + (tuple_positive_index).detach().cpu().numpy()
            ]
            lambdas_index_buffer = (
                torch.from_numpy(lambdas_index_buffer).float().to(device)
            )
            loss = loss_function(train_out, target.float())

            # If ALM is selected, calculate penalty and multipliers' terms
            if len(buffer_batch_pos) != 0 and len(buffer_batch_neg) != 0:
                if training_type == "LDAM":
                    buffer_batch_pos = buffer_batch_pos[:, :, 1]
                    buffer_batch_neg = buffer_batch_neg[:, :, 1]
            if use_ALM ==1:
                lambdas, loss_ALM_terms = ALM_terms_module(
                    buffer_batch_pos,
                    buffer_batch_neg,
                    lambdas_index_buffer,
                    lambdas,
                    mu,
                )
                loss_total = loss + loss_ALM_terms
            loss_total.backward()
            avg_loss_train = avg_loss_train + loss_total.item()

            # Re-initialize for next training batch
            buffer_batch_pos = []
            buffer_batch_neg = []
            lambdas_index_buffer = []
            optimizer.step()
            optimizer.zero_grad()
            if training_type == "LDAM":
                train_logits = (
                    train_logits + (train_out[:, 1].detach().cpu().numpy()).tolist()
                )
            else:
                train_logits = (
                    train_logits + (train_out.detach().cpu().numpy()).tolist()
                )
            train_gt_labels = train_gt_labels + (target.detach().cpu().numpy()).tolist()

        # At the end of each epoch calculate train AUC
        fpr, tpr, _ = roc_curve(
            (np.array(train_gt_labels)).astype(int), (train_logits), pos_label=1
        )
        train_roc_auc = auc(fpr, tpr)
        train_auc_list.append(train_roc_auc)

        # VALIDATION #
        val_loss = 0
        val_penalty = 0
        val_gt_labels = []
        val_logits = []
        val_buffer_batch_pos = []
        val_buffer_batch_neg = []

        net.eval()
        with torch.no_grad():
            print("Validation")
            for val_data in val_loader:
                val_data, val_target = val_data
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_out = net(val_data)

                if training_type != "LDAM":
                    if val_target.detach().cpu().numpy() == 1.0:
                        val_buffer_batch_pos.append(val_out.detach().cpu().numpy())
                    else:
                        val_buffer_batch_neg.append(val_out.detach().cpu().numpy())
                else:
                    if val_target.detach().cpu().numpy() == 1.0:
                        val_buffer_batch_pos.append(val_out[1].detach().cpu().numpy())
                    else:
                        val_buffer_batch_neg.append(val_out[1].detach().cpu().numpy())

                # Calculate validation loss
                if training_type != "MBAUC":
                    val_loss = (
                        val_loss
                        + loss_function(torch.unsqueeze(val_out, 0), val_target.float())
                        .detach()
                        .cpu()
                        .numpy()
                    )

                val_gt_labels.append(
                    np.squeeze(val_target.detach().cpu().numpy().astype(bool))
                )
                val_logits.append(np.squeeze(val_out.detach().cpu().numpy()))

            if training_type == "MBAUC":
                val_loss = (
                    loss_function(
                        torch.Tensor(np.array(val_logits)),
                        torch.Tensor(np.array(val_gt_labels)),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

            # Calculate ALM terms
            for pos_sample in val_buffer_batch_pos:
                val_q_pos = np.zeros([1, 1])
                for neg_sample in val_buffer_batch_neg:
                    val_q_pos += np.maximum(0, -(pos_sample - neg_sample) + delta)
                val_penalty += np.power(val_q_pos, 2)

            val_penalty = val_penalty / (
                len(val_buffer_batch_pos) * len(val_buffer_batch_neg)
            )
            if training_type != "MBAUC":
                val_loss = val_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)
            val_penalty_list.append(val_penalty)
            if training_type == "LDAM":
                for i, el in enumerate(val_logits):
                    val_logits[i] = el[1]
            fpr, tpr, _ = roc_curve(
                (np.array(val_gt_labels)).astype(int), (val_logits), pos_label=1
            )
            val_roc_auc = auc(fpr, tpr)
            val_auc_list.append(val_roc_auc)
            print(
                "Epoch: {} >> train. loss: {:0.4f}, train AUC {:0.4f}".format(
                    epoch, avg_loss_train, train_roc_auc
                )
            )
            print("          >> val AUC {:0.4f}".format(val_roc_auc))

            # Model checkpoint based on the minimum validation AUC
            # Always checkpoint at the first epoch
            if (continue_train == 0 and epoch == 0) or (
                continue_train == 1 and epoch == start_epoch
            ):
                max_val_auc = val_roc_auc
                val_logits_ckp_model = val_logits
                max_train_auc = train_roc_auc
                epoch_last_auc_max = epoch
                val_loss_at_max_auc = val_loss
                print("First chepoint.")
                state = {
                    "epoch": epoch + 1,
                    "net_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(state, os.path.join(model_dir, model_name + ".pth"))
            # For the next epochs checkpoint at best validation AUC
            else:
                if val_roc_auc > max_val_auc:
                    max_val_auc = val_roc_auc
                    val_logits_ckp_model = val_logits
                    max_train_auc = train_roc_auc
                    epoch_last_auc_max = epoch
                    val_loss_at_max_auc = val_loss
                    # Model checkpoint
                    state = {
                        "epoch": epoch + 1,
                        "net_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }
                    torch.save(state, os.path.join(model_dir, model_name + ".pth"))
                    print(
                        "New maximum AUC value {:0.4f} at epoch {}. Checkpointing the model.".format(
                            max_val_auc, epoch
                        )
                    )

                elif (
                    val_roc_auc == max_val_auc
                    and val_loss < val_loss_at_max_auc
                    and train_roc_auc > max_train_auc - 0.05
                ):
                    max_val_auc = val_roc_auc
                    val_logits_ckp_model = val_logits
                    max_train_auc = train_roc_auc
                    epoch_last_auc_max = epoch
                    val_loss_at_max_auc = val_loss
                    # Model checkpoint
                    state = {
                        "epoch": epoch + 1,
                        "net_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }
                    torch.save(state, os.path.join(model_dir, model_name + ".pth"))
                    print(
                        "Same maximum AUC value {:0.4f} with lower val loss {} at epoch {}. Checkpointing the model.".format(
                            max_val_auc, val_loss, epoch
                        )
                    )

        scheduler.step(val_roc_auc)

        # Update mu based on val_auc, up to mu limit
        if stop_update_mu == 0 and (epoch > introd_ALM_from) and (use_ALM == 1):
            if epoch == introd_ALM_from + 1:
                previous_auc = val_roc_auc
            else:
                if val_roc_auc == 1.0 and cap_mu == 1:
                    stop_update_mu = 1
                    print(
                        "Validation AUC reached 100%. No further change in mu. Mu remains: ",
                        mu,
                    )
                else:
                    mu_old = mu
                    mu = mu * rho
                    if mu > mu_limit:
                        mu = mu_limit
                        stop_update_mu = 1
                        print(
                            "Mu reached the limit value. No further change in mu. Mu remains: ",
                            mu,
                        )
                    else:
                        if epoch > introd_ALM_from + 2 or (
                            continue_train == 1 and epoch > 0
                        ):
                            if val_roc_auc < previous_auc - 0.05:
                                print("Mu changed from", mu_old, "to", mu)
                            else:
                                mu = mu_old
                        previous_auc = val_roc_auc

        # Early stopping to avoid overfitting
        if epoch - epoch_last_auc_max > patience_early_stopping:
            print(
                "No improvement since {} epochs. Early stopping.".format(
                    patience_early_stopping
                )
            )
            print(
                "Maximum validation AUC value {:0.4f} at epoch {}.".format(
                    max_val_auc, epoch_last_auc_max
                )
            )
            # Plot validation loss and AUC and store values
            with open(
                os.path.join(model_dir, model_name + "_loss_AUC_values.csv"),
                "w+",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["Val Loss", "Val AUC"])
                writer.writerows(zip(val_loss_list, val_auc_list))
                return max_val_auc, val_logits_ckp_model

    print(
        "Maximum validation AUC value {} at epoch {}.".format(
            max_val_auc, epoch_last_auc_max
        )
    )
    # Plot validation loss and AUC and store values when training is complete
    with open(
        os.path.join(model_dir, model_name + "_val_loss_AUC_values.csv"),
        "w+",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Val Loss", "Val AUC"])
        writer.writerows(zip(val_loss_list, val_auc_list))
    return max_val_auc, val_logits_ckp_model


def test(net, test_loader, model_dir, model_name, training_type, device):
    """
    Evaluate the performance of a trained neural network model on a test dataset.

    This function evaluates a trained neural network model on a test dataset, calculates
    the area under the ROC curve (AUC).

    Parameters:
        net (nn.Module): The trained neural network model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        model_dir (str): Directory where the trained model checkpoint is located.
        model_name (str): Name of the trained model to retrieve the checkpoint file.
        training_type (str): Type of loss (e.g., 'LDAM', 'MBAUC') used during training.
        device (str): Device (CPU or GPU) on which to perform the evaluation.

    Returns:
        float: Area under the receiver operating characteristic (ROC) curve (AUC) for the test dataset.
    """
    print("Starting test.")
    checkpoint = torch.load(os.path.join(model_dir, model_name + ".pth"))
    net.load_state_dict(checkpoint["net_state_dict"])
    net.eval()
    test_logits = []
    test_gt_labels = []
    test_labels_predicted = []

    with torch.no_grad():
        for test_data in test_loader:
            data, target = test_data
            data = data.to(device)
            test_out = net(data)

            test_out_max = (test_out.detach().cpu().numpy() > 0.0).astype(float)
            test_gt_labels.append(np.squeeze(target.detach().cpu().numpy().astype(int)))
            test_logits.append(np.squeeze(test_out.detach().cpu().numpy()))
            test_labels_predicted.append(np.squeeze(test_out_max.astype(int)))

        file_labels_pred = os.path.join(
            model_dir, model_name + "_labels_pred_and_GT_fv.csv"
        )
        file_logits_gt = os.path.join(model_dir, model_name + "_LOGITS_and_GT_fv.csv")

        if training_type == "LDAM":
            for i, el in enumerate(test_logits):
                test_logits[i] = el[1]
        np.savetxt(
            file_labels_pred,
            np.c_[test_gt_labels, test_labels_predicted],
            delimiter=",",
            header="GT,Pred",
            comments="",
        )
        np.savetxt(
            file_logits_gt,
            np.c_[test_gt_labels, test_logits],
            delimiter=",",
            header="GT,Z",
            comments="",
        )

        fpr, tpr, _ = roc_curve(test_gt_labels, test_logits, pos_label=1)
        roc_auc = auc(fpr, tpr)

        return roc_auc


def shuffle_dataset(dataset_train, lambdas):
    """
    Shuffle dataset samples along with lambdas
    """
    temp_dataset_lists = list(zip(dataset_train.targets, dataset_train.data))
    x = list(enumerate(temp_dataset_lists))
    random.shuffle(x)
    indeces, dataset_lists = zip(*x)
    dataset_train.targets, dataset_train.data = zip(*dataset_lists)
    lambdas = lambdas[np.asarray(indeces)]
    return lambdas, dataset_train


def load_checkpoint(net, optimizer, scheduler, filename):
    """
    Load pre-trained checkpoint
    """
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"])
        )
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return net, optimizer, start_epoch, scheduler


if __name__ == "__main__":
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    print(a, b, c)
    temp_dataset_lists = list(zip(a, b, c))
    x = list(enumerate(temp_dataset_lists))
    random.shuffle(x)
    indeces, dataset_lists = zip(*x)
    a, b, c = zip(*dataset_lists)
    print(a, b, c)
