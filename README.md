# ROSE

Code for our paper ROSE: Retrieval Augmented Contrastive Learning for Unlabeled Entity Problem in Named Entity Recognition

## Preparation
1. download pretrained LM: bert-base-chinese [model.pt](https://drive.google.com/file/d/1dh7yH6YeZNuBCY9-aS3HBf0FFcsfi4AG/view?usp=sharing) and put it into resource/bert-base-chinese
2. use requirements.txt to get the right environments.


## Reproduce results
For EC: 
>sh train_EC.sh

For NEWS: 
>sh train_NEWS.sh


We get our results in single A40.
