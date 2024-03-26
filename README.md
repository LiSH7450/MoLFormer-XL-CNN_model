File DESCRIPTION
====================
## Introduction
In this study, we employ MoLFormer-XL, a pre-trained model trained on large-scale molecular datasets, to encode molecules. This is coupled with convolutional neural networks (CNN) to predict the risk of drug-induced QT prolongation (DIQT), drug-induced teratogenicity (DIT), and drug-induced rhabdomyolysis (DIR).
## Configuration
The environment of our model is athe same as "MoLFormer"(https://github.com/IBM/molformer/blob/main/environment.md)
## Datasets
The data presented in this study: DIQTA,DITD,DIR.
## linear_attention_rotary
The code of attention maps.
## MoLFormer-XL-CNN
The Python codes for MoLFormer-XL-CNN model. 
## Supplementary File
Supplementary File 1: The attention maps of five drugs with a high-risk of DIQT. Supplementary File 2: The attention maps of two antiepileptic drugs with a high-risk of DIT. Supplementary File 3: The attention maps of seven statins.
