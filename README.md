以下是将上述内容翻译成英文的版本：

---

## U-Net: Convolutional Networks for Biomedical Image Segmentation Target Detection Model Implementation in Pytorch

---

### Table of Contents
1. [Top News](#Top News)
2. [Related Repositories](#Related Repositories)
3. [Performance](#Performance)
4. [Required Environment](#Required Environment)
5. [Download](#Download)
6. [Training Steps](#Training Steps)
7. [Prediction Steps](#Prediction Steps)
8. [Evaluation Steps](#Evaluation Steps)
9. [References](#References)

## Top News
**`March 2022`**: **Significant updates have been made, including support for step and cosine learning rate decay, support for Adam and SGD optimizers, and adaptive learning rate adjustment based on batch size.**  
The original repository address in the BiliBili video is: https://github.com/bubbliiiing/unet-pytorch/tree/bilibili 

**`August 2020`**: **The repository was created, supporting multiple backbones, data miou evaluation, annotation data processing, extensive comments, etc.**  

## Related Repositories
| Model | Path |
| :----- | :----- |
| Unet | https://github.com/bubbliiiing/unet-pytorch   
| PSPnet | https://github.com/bubbliiiing/pspnet-pytorch 
| Deeplabv3+ | https://github.com/bubbliiiing/deeplabv3-plus-pytorch 

### Performance
**Unet is not suitable for datasets like VOC, it is more suitable for medical datasets with fewer features that require shallow features.**
| Training Dataset | Weights File Name | Testing Dataset | Input Image Size | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| VOC12+SBD | [unet_vgg_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_vgg_voc.pth)  | VOC-Val12 | 512x512| 58.78 | 
| VOC12+SBD | [unet_resnet_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_resnet_voc.pth)  | VOC-Val12 | 512x512| 67.53 | 

### Required Environment
torch==1.2.0    
torchvision==0.4.0   

### Download
The weights required for training can be downloaded from Baidu Pan.    
Link: https://pan.baidu.com/s/1A22fC5cPRb74gqrpq7O9-A     
Extraction Code: 6n2c   

The VOC expanded dataset on Baidu Pan is as follows:   
Link: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng     
Extraction Code: 44mk   

### Training Steps
#### I. Training VOC Dataset
1. Place the provided VOC dataset into the VOCdevkit (no need to run voc_annotation.py).  
2. Run train.py to train, the default parameters are already corresponding to the parameters needed for the VOC dataset.  

#### II. Training Your Own Dataset
1. This article uses the VOC format for training.  
2. Before training, place the label files in the SegmentationClass folder under the VOC2007 folder in the VOCdevkit folder.    
3. Before training, place the image files in the JPEGImages folder under the VOC2007 folder in the VOCdevkit folder.    
4. Use the voc_annotation.py file to generate the corresponding txt before training.    
5. Note to modify the num_classes to the number of classes + 1 in train.py.    
6. Run train.py to start training.  

#### III. Training Medical Dataset
1. Download the VGG pre-trained weights to the model_data folder.  
2. Run train_medical.py with the default parameters to start training.

### Prediction Steps
#### I. Using Pre-trained Weights
##### a. VOC Pre-trained Weights
1. After downloading and unzipping the library, if you want to use the VOC trained weights for prediction, download the weights from Baidu Pan or release and place them in model_data, and you can run the prediction.  
```python
img/street.jpg
```    
2. In predict.py, you can set to perform fps testing and video detection.    
##### b. Medical Pre-trained Weights
1. After downloading and unzipping the library, if you want to use the medical dataset trained weights for prediction, download the weights from Baidu Pan or release and place them in model_data, modify model_path and num_classes in unet.py;
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path points to the weight file in the logs folder
    #   After training, there are multiple weight files in the logs folder, choose the one with lower validation loss.
    #   The lower validation loss does not mean higher miou, it only means that the weight has better generalization performance on the validation set.
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_medical.pth',
    #--------------------------------#
    #   The number of classes to be distinguished + 1
    #--------------------------------#
    "num_classes"   : 2,
    #--------------------------------#
    #   The main network used: vgg, resnet50   
    #--------------------------------#
    "backbone"      : "vgg",
    #--------------------------------#
    #   Input image size
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   The blend parameter is used to control whether
    #   to mix the recognition result with the original image
    #--------------------------------#
    "blend"         : True,
    #--------------------------------#
    #   Whether to use Cuda
    #   You can set it to False if there is no GPU
    #--------------------------------#
    "cuda"          : True,
}
```

2. Run to predict.  
```python
img/cell.png
```

#### II. Using Your Own Trained Weights
1. Train according to the training steps.    
2. In the unet.py file, modify model_path, backbone, and num_classes in the following part to correspond to the trained file; **model_path corresponds to the weight file under the logs folder**.    
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path points to the weight file in the logs folder
    #   After training, there are multiple weight files in the logs folder, choose the one with lower validation loss.
    #   The lower validation loss does not mean higher miou, it only means that the weight has better generalization performance on the validation set.
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_voc.pth',
    #--------------------------------#
    #   The number of classes to be distinguished + 1
    #--------------------------------#
    "num_classes"   : 21,
    #--------------------------------#
    #   The main network used: vgg, resnet50   
    #--------------------------------#
    "backbone"      : "vgg",
    #--------------------------------#
    #   Input image size
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   The blend parameter is used to control whether
    #   to mix the recognition result with the original image
    #--------------------------------#
    "blend"         : True,
    #--------------------------------#
    #   Whether to use Cuda
    #   You can set it to False if there is no GPU
    #--------------------------------#
    "cuda"          : True,
}
```

3. Run predict.py, input    
```python
img/street.jpg
```   
4. In predict.py, you can set to perform fps testing and video detection.    

### Evaluation Steps
1. Set num_classes in get_miou.py to the number of predicted classes plus 1.  
2. Set name_classes in get_miou.py to the categories that need to be distinguished.  
3. Run get_miou.py to get the miou size.  

## References
https://github.com/ggyyzm/pytorch_segmentation   
https://github.com/bonlime/keras-deeplab-v3-plus   

---
