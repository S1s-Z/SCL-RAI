# SCL-RAI
Hi, this is the code of our paper "SCL-RAI: Span-based Contrastive Learning with Retrieval Augmented Inference for Unlabeled Entity Problem in NER" accepted by COLING 2022. Our paper is available [here](https://arxiv.org/pdf/2209.01646.pdf).

News:

Accepted by COLING 2022. 2022.08.16

Code released at Github. 2022.08.16

## Preparation
1. Download pretrained LM: bert-base-chinese [model.pt](https://drive.google.com/drive/folders/1gZHmImlXDX6-uJaMvcYJf24ZwMCeq1P-?usp=sharing) and put it into resource/bert-base-chinese
2. Use requirements.txt to get the right environments.


## Reproduce results
For EC: 
>sh train_EC.sh

For NEWS: 
>sh train_NEWS.sh


We got our results in single A40.
