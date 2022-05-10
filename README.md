# ROSE

Code for our paper ROSE: Retrieval Augmented Contrastive Learning for Unlabeled Entity Problem in Named Entity Recognition

## Preparation

1. you need to download pretrained LM: bert-base-chinese [model.pt](https://drive.google.com/file/d/1dh7yH6YeZNuBCY9-aS3HBf0FFcsfi4AG/view?usp=sharing) and put it into resource/bert-base-cased/
2. use requirements.txt to get the right environment.


## Reproduce results
For EC: CUDA_VISIBLE_DEVICES=0 python main.py -dd dataset/EC -cd save -rd resource
For NEWS: CUDA_VISIBLE_DEVICES=0 python main.py -dd dataset/MSRA -cd save -rd resource -cs 1000 -ud True


We get our results in single A40.
