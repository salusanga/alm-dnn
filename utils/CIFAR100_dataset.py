import torchvision
import numpy as np
import warnings

class BinaryUnbalancedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, imb_factor, seed, tot_classes,
                new_majority_labels, new_minority_labels,
                val_samples_per_cls = 50, perc_per_run = 0.9,
                train=True, transform=None, target_transform=None,
                download=False):
        original_num_classes = 100
        super(BinaryUnbalancedCIFAR100, self).__init__(root, train, transform,
                                                target_transform, download)

        self.gen_imbalanced_data(seed, original_num_classes,
                        new_majority_labels, new_minority_labels,
                        imb_factor, val_samples_per_cls, perc_per_run)

    def gen_imbalanced_data(self, seed, original_num_classes,
                        new_majority_labels, new_minority_labels,
                        imb_factor, val_samples_per_cls, perc_per_run):
    
        img_max = int(((len(self.data) / original_num_classes) \
                                - val_samples_per_cls) * perc_per_run)
        img_min = int((img_max * np.size(new_majority_labels) * imb_factor) \
                            / np.size(new_minority_labels))
        img_num_per_cls = [img_max] * np.size(new_majority_labels) + \
                            [img_min] * np.size(new_minority_labels)
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
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        for i, target in enumerate(self.targets):
            if target in new_majority_labels:
                self.targets[i] = 0
            elif target in new_minority_labels:
                self.targets[i] = 1
            else:
                warnings.warn('Original class is not listed')
    
    def get_tot_classes_list(self, tot_classes):
        tot_classes_list = []
        for i in range(tot_classes):
            tot_classes_list.append(self.targets.count(i))
        return tot_classes_list





class BinaryBalancedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, tot_classes,
                new_majority_labels, new_minority_labels,
                val_samples_per_cls = 50,
                train=True, transform=None, target_transform=None,
                download=False):
        super(BinaryBalancedCIFAR100, self).__init__(root, train, transform,
                                                target_transform, download)

        self.gen_imbalanced_data(new_majority_labels, new_minority_labels, val_samples_per_cls)

    def gen_imbalanced_data(self,
                            new_majority_labels, new_minority_labels,
                            val_samples_per_cls):
    
        img_num = val_samples_per_cls
        img_num_per_cls = [img_num] * np.size(new_majority_labels) + \
                            [img_num] * np.size(new_minority_labels)
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
            np.random.seed(42)
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        for i, target in enumerate(self.targets):
            if target in new_majority_labels:
                self.targets[i] = 0
            elif target in new_minority_labels:
                self.targets[i] = 1
            else:
                warnings.warn('Original class is not listed')


    def get_tot_classes_list(self, tot_classes):
        tot_classes_list = []
        for i in range(tot_classes):
            tot_classes_list.append(self.targets.count(i))
        return tot_classes_list
