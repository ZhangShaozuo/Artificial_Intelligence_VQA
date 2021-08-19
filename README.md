# Artificial_Intelligence_VQA
50.021 Artificial Intelligence course project on VQA

## Dataset Visualization

```Dataset_visualization.ipynb```. It includes two different Class distribution bar charts and three word frequency bar charts for training, validation and test set.

## Training and testing

```BERT_AlexNet_Mul.ipynb, BERT_ResNet152.ipynb, GRU_Alexnet_Attention.ipynb```. They includes experiments on model architectures. 

The question processor opts:  GRU, pretrained BERT, 

the image processor opts from AlexNet and ResNet, 

Fusion schema opts from element-wise multiplication, element-wise concatenation.

General architectures opts to have question-guided image attention or not.

## Demonstration

```/ui``` . It includes frontend and backend implementation. Instructions for running is allocated at ```/ui/frontend```

The model weights and model architecture used for demo is allocated at ```/models```

## Utils

```/utils/vocab.py``` Vocab class is for building customized question vocabulary and answer vocabulary. 

```/utils/data.py``` VQA2Dataset extracts the information from public VQA 2.0 Balanced Real Images Dataset and reconstruct the customized dataset for training and testing

```/utils/train.py```  Methods include training and validation function, as well as model save and model load. Top-k accuracy is supported. 

```/utils/helper.py``` Customized process bar for training

