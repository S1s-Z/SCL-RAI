# SCL-RAI
Hi, this is the code of our paper "SCL-RAI: Span-based Contrastive Learning with Retrieval Augmented Inference for Unlabeled Entity Problem in NER" accepted by COLING 2022.

## Preparation
1. download pretrained LM: bert-base-chinese [model.pt](https://drive.google.com/file/d/1dh7yH6YeZNuBCY9-aS3HBf0FFcsfi4AG/view?usp=sharing) and put it into resource/bert-base-chinese
2. use requirements.txt to get the right environments.


## Reproduce results
For EC: 
>sh train_EC.sh

For NEWS: 
>sh train_NEWS.sh


We get our results in single A40.
