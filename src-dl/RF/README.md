# USENIX Security 2023: Subverting Website Fingerprinting Defenses with Robust Traffic Representation (https://github.com/robust-fingerprinting/RF)

## The following code was obtained from: [https://github.com/robust-fingerprinting/RF](https://github.com/robust-fingerprinting/RF)  

This is a Pytorch implementation of [Subverting Website Fingerprinting Defenses with Robust Traffic Representation](https://www.usenix.org/conference/usenixsecurity23/presentation/shenmeng). 
 
   
The diagram of TAM extraction:

<div align="center">
<img src="img/TAM_overview.png" width="600px">
</div>

Architecture of the code:
```
ROBUST-FINGERPRINTING
└─  RF
    │  const_rf.py (parameters of the RF)
    │  extract-all.py (extract all traces from the dataset into a .npy file for 10-fold validation)
    │  extract-list.py (extract traces according to the training and testing indices and save them into .npy files)
    │  pre_recall.py (file for evaluating functions)
    │  test.py (test file for the dataset)
    │  train.py (train file for the dataset)
    │  train_10fold.py (10-fold validation)
    ├─ dataset (folder for .npy dataset)
    ├─ FeatureExtraction (folder for feature extraction functions)
    │     packets_per_slot.py (the extraction function of TAM)
    ├─ list (training and testing trace indices, also including the fastest&slowest loaded trace indices)
    ├─ models
    │     RF.py (RF network)
    ├─ pretrained (trained model folder)
    └─ result (folder for evaluation results)
```
:warning: The code is intended for RESEARCH PURPOSES ONLY! :warning:

## Preparation

### Environments
* Python Version: 3.6  

Use this command to install the required packages.
```commandline
pip3 install -r requirements.txt
```

## How to Run
There are several data paths you should change in the code, and we have marked them with `#TODO`.

### Robust Fingerprinting
#### Parameter Setting
You can find the parameter config information in `RF/const_rf.py`.
#### Feature Extraction
The following command can extract a randomly split training and testing dataset.
```commandline
python extract-list.py
```
or use the following command to extract all traces for 10-fold validation.
```commandline
python extract-all.py
```
The extracted dataset will be saved in `RF/dataset`.
#### Training
If you want to train the model on the dataset with the given training indices, you can use this command.   
```commandline
python train.py
```
If you want to train the model with 10-fold validation, you can use this command.
```commandline
python train_10fold.py
```
#### Evaluate
If you want to evaluate RF, you can use this command. And you need to change the trained model path to load different models and the test dataset path.
```commandline
python test.py
```


## Main Results

### Closed World
<div align="center">
<img src="img/closed_world.png" width="800px">
</div>

### Open World
<div align="center">
<img src="img/open_world.png" width="800px">
</div>

## Citation
If you find this work useful for your research, please cite the following:
```
M. Shen, K. Ji, Z. Gao, Q. Li, L. Zhu, and K. Xu, "Subverting website fingerprinting defenses with robust traffic representation," in 32nd USENIX Security Symposium (USENIX Security 23). Anaheim, CA: USENIX Association, Aug. 2023, pp. 607–624. [Online]. Available: https://www.usenix.org/conference/usenixsecurity23/presentation/shen-meng
```
The BibTeX is as follows:
```BibTeX
@inproceedings {287396,
    author = {Meng Shen and Kexin Ji and Zhenbo Gao and Qi Li and Liehuang Zhu and Ke Xu},
    title = {Subverting Website Fingerprinting Defenses with Robust Traffic Representation},
    booktitle = {32nd USENIX Security Symposium (USENIX Security 23)},
    year = {2023},
    isbn = {978-1-939133-37-3},
    address = {Anaheim, CA},
    pages = {607--624},
    url = {https://www.usenix.org/conference/usenixsecurity23/presentation/shen-meng},
    publisher = {USENIX Association},
    month = aug
}
```

## Contact
If you have any questions, please get in touch with us.
* Prof. Meng Shen ([shenmeng@bit.edu.cn](shenmeng@bit.edu.cn))
* Kexin Ji ([jikexin@bit.edu.cn](jikexin@bit.edu.cn))

More detailed information about the research of Meng Shen Lab can be found here (https://mengshen-office.github.io/).
