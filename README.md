# Artificial_Intelligence_VQA
50.021 Artificial Intelligence course project on VQA

## Dataset Visualization

```Dataset_visualization.ipynb```. It includes two different Class distribution bar charts and three word frequency bar charts for training, validation and test set.

## Training and Testing

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

## Report

Experiment settings, result discussion see the report ```50.021 AI-VQA-Project.pdf```

## Run Instructions

1. Install [PyTorch 1.8](https://pytorch.org/get-started/locally/), [transformers](https://huggingface.co/transformers/installation.html) and [flask](https://flask.palletsprojects.com/en/2.0.x/installation/).

For example:
```
conda create -n vqa python=3.8
conda activate vqa
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install transformers==4.8.2 flask==2.0.1
```

2. Clone this repository:

```
git clone https://github.com/ZhangShaozuo/Artificial_Intelligence_VQA.git
cd Artificial_Intelligence_VQA
```

### Models
1. To train a new model, run `BERT_ResNet152_Mul.ipynb` (skipping cell `Load
pre-trained weight`) to create the model, train it on the VQA v2.0 dataset, and
evaluate the model on the test set.
2. For reproducibility of our results, download the model weights from this link and save
it to the `./models` folder:
https://sutdapac-my.sharepoint.com/:u:/g/personal/yuhang_he_mymail_sutd_edu_sg/EbuC0ops4GFDu-Cp6QKKSMMBWBPk0X8a_ICa9YlPxzejww?e=tEvw3a
3. Run `BERT_ResNet152_Mul.ipynb` (skipping cell `Train model`) to load the
pre-trained weights of our best model before evaluating the model.

### Dataset Visualization
1. Run `Dataset_Visualization.ipynb` to print the sizes of the dataset and visualize
the dataset by getting graphs for class distribution and top 50 most frequent words in
train, val, and test dataset.

### User Interface
1. Download the model weights from this link and save it to the `./models` folder:
https://sutdapac-my.sharepoint.com/:u:/g/personal/yuhang_he_mymail_sutd_edu_sg/EbuC0ops4GFDu-Cp6QKKSMMBWBPk0X8a_ICa9YlPxzejww?e=tEvw3a
2. Run `python -m ui.backend` to launch the backend service.
3. Open frontend folder: `cd ui/frontend`
4. Install frontend dependencies: `yarn install`
5. Start frontend: `yarn start`
6. Go to [http://localhost:3000](http://localhost:3000) to see the demo web app.


