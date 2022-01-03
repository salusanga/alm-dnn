# Constrained Optimization to Train Neural Networks on Critical and Under-Represented Classes

This repository is the official implementation of [Constrained Optimization to Train Neural Networks on Critical and Under-Represented Classes](https://arxiv.org/abs/2102.12894).

## Overview
Deep neural networks (DNNs) are notorious for making more mistakes for the classes that have substantially fewer samples than the others during training. Such class imbalance is ubiquitous in clinical applications and very crucial to handle because the classes with fewer samples most often correspond to critical cases (e.g., cancer) where misclassifications can have severe consequences. Not to miss such cases, binary classifiers need to be operated at high True Positive Rates (TPRs) by setting a higher threshold, but this comes at the cost of very high False Positive Rates (FPRs) for problems with class imbalance. Existing methods for learning under class imbalance most often do not take this into account. We argue that prediction accuracy should be improved by emphasizing reducing FPRs at high TPRs for problems where misclassification of the positive, i.e. critical, class samples are associated with higher cost. To this end, we pose the training of a DNN for binary classification as a constrained optimization problem and introduce a novel constraint that can be used with existing loss functions to enforce maximal area under the ROC curve (AUC) through prioritizing FPR reduction at high TPR. We solve the resulting constrained optimization problem using an Augmented Lagrangian method (ALM). Going beyond binary, we also propose two possible extensions of the proposed constraint for multi-class classification problems. We present experimental results for image-based binary and multi-class classification applications using an in-house medical imaging dataset, CIFAR10, and CIFAR100. Our results demonstrate that the proposed method improves the baselines in majority of the cases by attaining higher accuracy on critical classes while reducing the misclassification rate for the non-critical class samples.

The repo contains the code required to train/evaluate a model on a binary task. The multi-class version will be made avsailable.

If you find this code useful for your project, please consider citing our paper:
```
@inproceedings{sangalli2021constrained,
  title={Constrained Optimization to Train Neural Networks on Critical and Under-Represented Classes},
  author={Sara Sangalli and Ertunc Erdil and Andeas H{\"o}tker and Olivio F Donati and Ender Konukoglu},
  booktitle={Advances in Neural Information Processing Systems},
  editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
  year={2021},
  url={https://openreview.net/forum?id=vERYhbX_6Y}
}
```

## Installation

To create a conda environment with all the requirements:

```setup1
conda env create -f environment.yml
```

Activate the environment via:
```setup2
conda activate alm-dnn
```

Create a folder 'datasets/' to store the data.


## Important arguments
These are the most important arguments to set for the experiments.
___
Parameter | Explaination |
--- | --- |
name_file_bs | config file to set the objective function (loss function) and its required hyperparameters, chosen from [configs/baseline/](https://gitlab.ethz.ch/sansara/alm-dnn/-/tree/master/configs/baseline/) |
name_file_alm | config file to set the ALM parameters (mu, rho), chosen from [configs/ALM/](https://gitlab.ethz.ch/sansara/alm-dnn/-/tree/master/configs/ALM/) |
delta | value of delta for ALM (or set to 0 if not used) |
model_name | name of the model to be trained from scratch or resumed |
ratio_pos_train | ratio between classes 1:x, for binary case|
dataset_name | either 'cifar10' or 'cifar100' |
___


## Training from scratch and evaluate

Run the file [main.py](https://gitlab.ethz.ch/sansara/alm-dnn/-/blob/master/main.py)  to train and test a desired model, with a specific configuration of the parameters specified in the previous section. For example, in order to train from scratch a ResNet (whose base parameters are set, as in the paper), with BCE + ALM with a certain (mu, rho), on CIFAR10 with an injected ratio of 1:200:

```train
python main.py --name_file_bs baseline_bce --name_file_alm ALM_14_beg --model_name alm_14  --ratio_pos_train 200 --dataset_name cifar10 --train 1
```

## Evaluation of a pre-trained model

To evaluate a pre-trained model, ofirst create a folder called 'checkpoint/' where to put the pre-trained model, and assign its name to the argument '--model_name'. For example, to train an already existing model 'alm_14':

```eval
python main.py --name_file_bs baseline_bce --name_file_alm ALM_14_beg --model_name alm_14  --ratio_pos_train 200 --dataset_name cifar10 --train 0
```

## Configurations of hyperparameters
Here we report the configurations used in the main paper for the binary experiments.

___
**CIFAR10, CLASS RATIO 1:100**
| Model | Name config. file objective function | Name config. file ALM | Delta |
| --- | --- | --- | --- |
BCE | baseline_BCE | - | - |
S-LM | baseline_s_LM_m05 | - | - |
S-FL | baseline_s_FL_g05 | - | - |
A-LM | baseline_a_LM_m05 | - | - |
A-FL |  baseline_a_FL_g1 | - | - |
CB-BCE | baseline_cb_BCE_99 | - | - |
W-BCE |  baseline_WBCE_1_3 | - | - |
LDAM | baseline_LDAM | - | - |
MBAUC | baseline_MBAUC | - | - |
  |  |  |  |
BCE + ALM | baseline_BCE | ALM_17_beg | 0.25 |
S-LM + ALM | baseline_s_LM_m05 | ALM_16_beg | 0.1 |
S-FL + ALM | baseline_s_FL_g05 | ALM_16_beg | 0.5 |
A-LM + ALM | baseline_a_LM_m05 | ALM_17_beg | 0.25 |
A-FL + ALM | baseline_a_FL_g1 | ALM_15_beg | 0.25 |
CB-BCE + ALM | baseline_cb_BCE_99 | ALM_14_beg | 0.5 |
W-BCE + ALM | baseline_WBCE_1_3 | ALM_17_beg | 1.0 |
LDAM + ALM | baseline_LDAM | ALM_16_beg | 1.0 |
___


___
**CIFAR10, CLASS RATIO 1:200**
| Model | Name config. file objective function | Name config. file ALM | Delta |
| --- | --- | --- | --- |
BCE | baseline_BCE | - | - |
S-LM | baseline_s_LM_m05 | - | - |
S-FL | baseline_s_FL_g05 | - | - |
A-LM | baseline_a_LM_m05 | - | - |
A-FL | baseline_a_FL_g05 | - | - |
CB-BCE | baseline_cb_BCE_99 | - | - |
W-BCE | baseline_WBCE_1_3 | - | - |
LDAM | baseline_LDAM | - | - |
MBAUC | baseline_MBAUC | - | - |
  |  |  |  |
BCE + ALM | baseline_BCE | ALM_15_beg | 0.5 |
S-LM + ALM | baseline_s_LM_m05 | ALM_19_beg | 0.0 |
S-FL + ALM | baseline_s_FL_g05 | ALM_17_beg | 0.0 |
A-LM + ALM | baseline_a_LM_m05 | ALM_19_beg | 0.0 |
A-FL + ALM |  baseline_a_FL_g05 | ALM_17_beg | 1.0 |
CB-BCE + ALM | baseline_cb_BCE_99 | ALM_14_beg | 0.1 |
W-BCE + ALM |  baseline_WBCE_1_3| ALM_17_beg | 0.1 |
LDAM + ALM | baseline_LDAM | ALM_18_beg | 0.25 |
___


___
**CIFAR100, CLASS RATIO 1:100**
| Model | Name config. file objective function | Name config. file ALM | Delta |
| --- | --- | --- | --- |
BCE | baseline_BCE | - | - |
S-LM | baseline_s_LM_m05 | - | - |
S-FL | baseline_s_FL_g2 | - | - |
A-LM | baseline_a_LM_m2 | - | - |
A-FL |  baseline_a_FL_g2 | - | - |
CB-BCE | baseline_cb_BCE_99 | - | - |
W-BCE |  baseline_WBCE_1 | - | - |
LDAM | baseline_LDAM | - | - |
MBAUC | baseline_MBAUC | - | - |
  |  |  |  |
BCE + ALM | baseline_BCE | ALM_13_beg | 0.1 |
S-LM + ALM | baseline_s_LM_m05 | ALM_13_beg | 1.0 |
S-FL + ALM | baseline_s_FL_g2 | ALM_15_beg | 0.25 |
A-LM + ALM | baseline_a_LM_m2 | ALM_13_beg | 0.5 |
A-FL + ALM | baseline_a_FL_g2 | ALM_14_beg | 0.1 |
CB-BCE + ALM | baseline_cb_BCE_99 | ALM_12_beg | 0.25 |
W-BCE + ALM | baseline_WBCE_1 | ALM_17_beg | 1.0 |
LDAM + ALM | baseline_LDAM | ALM_17_beg | 0.5 |
___


___
**CIFAR100, CLASS RATIO 1:200**
| Model | Name config. file objective function | Name config. file ALM | Delta |
| --- | --- | --- | --- |
BCE | baseline_BCE | - | - |
S-LM | baseline_s_LM_m2 | - | - |
S-FL | baseline_s_FL_g2 | - | - |
A-LM | baseline_a_LM_m4 | - | - |
A-FL | baseline_a_FL_g2 | - | - |
CB-BCE | baseline_cb_BCE_99 | - | - |
W-BCE | baseline_WBCE_1 | - | - |
LDAM | baseline_LDAM | - | - |
MBAUC | baseline_MBAUC | - | - |
  |  |  |  |
BCE + ALM | baseline_BCE | ALM_13_beg | 0.1 |
S-LM + ALM | baseline_s_LM_m2 | ALM_14_beg | 0.5 |
S-FL + ALM | baseline_s_FL_g2 | ALM_13_beg | 0.1 |
A-LM + ALM | baseline_a_LM_m4 | ALM_13_beg | 0.5 |
A-FL + ALM |  baseline_a_FL_g2 | ALM_13_beg | 0.25 |
CB-BCE + ALM | baseline_cb_BCE_99 | ALM_13_beg | 1.0 |
W-BCE + ALM |  baseline_WBCE_1| ALM_16_beg | 1.0 |
LDAM + ALM | baseline_LDAM | ALM_13_beg | 0.25 |
___
