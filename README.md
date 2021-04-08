## SSS-AE: Anomaly Detection using Self-Attention based Sequence-to-Sequence Auto-Encoder in SMD Assembly Machine Sound
[SSS-AE: Anomaly Detection using Self-Attention based Sequence-to-Sequence Auto-Encoder in SMD Assembly Machine Sound]

[Kihyun Nam](https://github.com/DevKiHyun)<sup>1*</sup>, YoungJong Song<sup>1*</sup>, IlDong Yun<sup>1,2</sup> 

A Surface-Mounted Device (SMD) assembly machine continuously assembles various products in real field. Unwanted situations such as assembly failure and device breakdown can occur at any time during the assembly process and result in costly losses. Anomaly detection techniques using deep learning are effective in detecting such abnormal situations. Two training scenarios, single-product learning and multi-product learning, can be considered for SMD anomaly detection workflows. Since there are not many products in previous studies, single-product learning is sufficient. However, multi-product learning is required when the number of products increases gradually. Successful performance of multi-product learning requires effective learning methods for various assembly sound data. In this paper, we propose robust model and effective data preprocessing method, Self-Attention based Sequence-to-Sequence Auto-Encoder (SSS-AE) and Temporal Adaptive Average Pooling (TAAP). For more accurate evaluation compared with the previous SMD anomaly detection studies, a new large-scale SMD dataset containing observed real abnormal products were collected and evaluated. As a result, we show that SSS-AE and TAAP are powerful and practical approaches for both single-product learning and multi-product learning.

<sup>1</sup> Department of Computer Engineering, Hankuk University on Foreign Studies <p>
<sup>2</sup> Corresponding Author <p>
<sup>*</sup> Both authors equally contributed to this work.
  

## Table of contents 
* [1. Dataset](#1-dataset)
    + [LSMD](#lsmd)
    + [Setting](#setting)
* [2. Dependency](#2-dependency)
* [3. Training and Evaluation](#3-training-and-evaluation)
    + [Run train](#run-train)
    + [Run test](#run-test)

## 1. Dataset

### LSMD
```
We plan to release the large-scale SMD (LSMD) dataset with additional information in follow-up study.
```

#### Audio examples

| Name | Normal 1 | Normal 2 | Error level 1 | Error level 2|
| :---: | :-----: | :------: | :------------: | :-----------: |
| GT-4118 | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/GT-4118/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/GT-4118/002.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/GT-4118-1/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/GT-4118-2/001.wav) |
| ST-3214 | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/ST-3214/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/ST-3214/002.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/ST-3214-1/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/ST-3214-2/001.wav) |
| ST-3708 | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/ST-3708/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/ST-3708/002.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/ST-3708-1/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/LSMD/ST-3708-2/001.wav) |


### Setting
We should follow the structure of the directory and manifests of the dataset as below:

```
Tensorflow-SSAE/
└──run.py
...
└──dataset/ # Important!!
   └──LSMD/
      └──GT-4118/
         └──001.wav
         ...
         └──00N.wav
      └──ST-3214/
         └──001.wav
         ...
         └──00N.wav
   └──other_dataset/ # Just Example 
      └──class1/
         └──sample1.wav
         ...
         └──sample2.wav
└──manifets/ # Manifests of target data. Also Important!!
   └──GT-4118.json
   └──GT-4118-1.json
   └──GT-4118-2.json
   └──ST-3214.json
   └──ST-3214-1.json
   └──ST-3214-2.json
   └──ST-3708.json
   └──ST-3708-1.json
   └──ST-3708-2.json
```

We should be make manfiests of the target data(to train and test) into `manifests/` (e.g. GT-4118, ST-3214)
```
GT-4118.json
[
    {
        "wav": "LSMD/GT-4118/001",
        "sr": 192000,
        "item": "GT-4118",
        "type": "NORMAL"
    },
    {
        "wav": "LSMD/GT-4118/002",
        "sr": 192000,
        "item": "GT-4118",
        "type": "NORMAL"
    }
]
---------------------
ST-3214.json
[
    {
        "wav": "LSMD/ST-3214/001",
        "sr": 192000,
        "item": "ST-3214",
        "type": "NORMAL"
    },
    {
        "wav": "LSMD/ST-3214/002",
        "sr": 192000,
        "item": "ST-3214",
        "type": "NORMAL"
    }
]
```

## 2. Dependency
```
numpy==1.16.4
matplotlib==3.2.1
librosa==0.7.0
scipy==1.3.1
tensorflow==1.14 (Available= 1.10 <= x <=1.14)
```

## 3. Training and Evaluation

### Run train

```
cd script/

./run_ssae_trainer.sh
```

### Run test
```
cd script/

./run_ssae_test.sh
```
