import torchvision
import numpy as np
import warnings


class BinaryUnbalancedCIFAR100(torchvision.datasets.CIFAR100):
    """
    Custom dataset class for creating a binary unbalanced version of CIFAR-100 dataset.

    This class extends the torchvision CIFAR100 dataset and allows you to create a binary
    classification task with imbalanced classes. It generates a new dataset with only two
    classes: majority and minority, while preserving the original CIFAR-100 images.

    Parameters:
        root (str): Root directory of the CIFAR-100 dataset.
        imb_factor (float): Imbalance factor, representing the ratio of minority class samples
            to majority class samples.
        seed (int): Seed for randomization to ensure reproducibility.
        tot_classes (int): Total number of classes in the binary dataset (usually 2).
        new_majority_labels (list): List of class labels to be assigned to the majority class.
        new_minority_labels (list): List of class labels to be assigned to the minority class.
        val_samples_per_cls (int): Number of validation samples to reserve for each class.
        perc_per_run (float): Percentage of the majority class samples to use for training.

    Attributes:
        num_per_cls_dict_original (dict): A dictionary containing the original number of samples
            per class before creating the binary dataset.
        num_per_cls_dict (dict): A dictionary containing the number of samples per class in the
            binary dataset after balancing.

    Methods:
        gen_imbalanced_data(seed, original_num_classes, new_majority_labels, new_minority_labels,
                            imb_factor, val_samples_per_cls, perc_per_run):
            Generates the imbalanced binary dataset based on the provided parameters.
        get_tot_classes_list(tot_classes):
            Returns a list of counts for each class in the binary dataset.

    Note:
        - The original CIFAR-100 dataset has 100 classes, but this class allows you to create a
          binary classification task by specifying two target classes: majority and minority.
        - The generated dataset is imbalanced, with the specified imbalance factor.
    """

    def __init__(
        self,
        root,
        imb_factor,
        seed,
        new_majority_labels,
        new_minority_labels,
        tot_classes=2,
        val_samples_per_cls=50,
        perc_per_run=0.9,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        original_num_classes = 100
        super(BinaryUnbalancedCIFAR100, self).__init__(
            root, train, transform, target_transform, download
        )

        self.gen_imbalanced_data(
            seed,
            original_num_classes,
            new_majority_labels,
            new_minority_labels,
            imb_factor,
            val_samples_per_cls,
            perc_per_run,
        )

    def gen_imbalanced_data(
        self,
        seed,
        original_num_classes,
        new_majority_labels,
        new_minority_labels,
        imb_factor,
        val_samples_per_cls,
        perc_per_run,
    ):
        img_max = int(
            ((len(self.data) / original_num_classes) - val_samples_per_cls)
            * perc_per_run
        )
        img_min = int(
            (img_max * np.size(new_majority_labels) * imb_factor)
            / np.size(new_minority_labels)
        )
        img_num_per_cls = [img_max] * np.size(new_majority_labels) + [
            img_min
        ] * np.size(new_minority_labels)
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.concatenate((new_majority_labels, new_minority_labels))
        # np.random.shuffle(classes) # to test on randomly imbalanced test set
        self.num_per_cls_dict_original = dict()
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict_original[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.seed(seed)
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend(
                [
                    the_class,
                ]
                * the_img_num
            )
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        for i, target in enumerate(self.targets):
            if target in new_majority_labels:
                self.targets[i] = 0
            elif target in new_minority_labels:
                self.targets[i] = 1
            else:
                warnings.warn("Original class is not listed")

    def get_tot_classes_list(self, tot_classes):
        tot_classes_list = []
        for i in range(tot_classes):
            tot_classes_list.append(self.targets.count(i))
        return tot_classes_list


class BinaryBalancedCIFAR100(torchvision.datasets.CIFAR100):
    """
    Custom dataset class for creating a binary balanced version of the CIFAR-100 dataset.

    This class extends the torchvision CIFAR100 dataset to facilitate the creation of a
    binary classification task with balanced classes. It generates a new dataset with only two
    classes: majority and minority, while preserving the original CIFAR-100 images.

    Parameters:
        root (str): Root directory of the CIFAR-100 dataset.
        tot_classes (int): Total number of classes in the binary dataset (usually 2).
        first_class_labels (list): List of class labels to be assigned to the majority class.
        second_class_labels (list): List of class labels to be assigned to the minority class.
        val_samples_per_cls (int): Number of validation samples to reserve for each class.
        train (bool): If True, loads the training dataset; otherwise, loads the test dataset.
        transform (callable, optional): A function/transform to apply to the dataset's samples.
        target_transform (callable, optional): A function/transform to apply to the dataset's targets.
        download (bool): If True, downloads the dataset from the internet and places it in `root`
            if it doesn't already exist.

    Attributes:
        num_per_cls_dict_original (dict): A dictionary containing the original number of samples
            per class before creating the binary dataset.
        num_per_cls_dict (dict): A dictionary containing the number of samples per class in the
            binary dataset after balancing.

    Methods:
        gen_balanced_two_classes(new_majority_labels, second_class_labels, val_samples_per_cls):
            Generates a binary balanced dataset based on the provided parameters.
        get_tot_classes_list(tot_classes):
            Returns a list of counts for each class in the binary dataset.

    Note:
        - The original CIFAR-100 dataset has 100 classes, but this class allows you to create a
          binary classification task with two specified classes.
        - The generated dataset is balanced with an equal number of samples for each class.
    """

    def __init__(
        self,
        root,
        tot_classes,
        first_class_labels,
        second_class_labels,
        val_samples_per_cls=50,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(BinaryBalancedCIFAR100, self).__init__(
            root, train, transform, target_transform, download
        )

        self.gen_balanced_two_classes(
            first_class_labels, second_class_labels, val_samples_per_cls
        )

    def gen_balanced_two_classes(
        self, first_class_labels, second_class_labels, val_samples_per_cls
    ):
        img_num = val_samples_per_cls
        img_num_per_cls = [img_num] * np.size(first_class_labels) + [img_num] * np.size(
            second_class_labels
        )
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.concatenate((first_class_labels, second_class_labels))
        # np.random.shuffle(classes) # to test on randomly imbalanced test set
        self.num_per_cls_dict_original = dict()
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict_original[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.seed(42)
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend(
                [
                    the_class,
                ]
                * the_img_num
            )
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        for i, target in enumerate(self.targets):
            if target in first_class_labels:
                self.targets[i] = 0
            elif target in second_class_labels:
                self.targets[i] = 1
            else:
                warnings.warn("Original class is not listed")

    def get_tot_classes_list(self, tot_classes):
        tot_classes_list = []
        for i in range(tot_classes):
            tot_classes_list.append(self.targets.count(i))
        return tot_classes_list
