# Faster-RCNN-pytorch
implement Faster R-CNN by Pytorch
This code Learned from [https://github.com/chenyuntc/simple-faster-rcnn-pytorch.git]()，And I personally made relevant optimizations, such as adding a video target detection algorithm.

If you need the pretrained model of Vgg16_caffe.pth or resnet101_caffe.pth, you can send an email to me.

The method of Faster R-CNN is as follows
![alt text](image.png)

The resultof my Faster R-CNN is as follows:

| Pretrained Model  | Map(%) | Paper Map(%) |
| ----------------- | ------ | ------------ |
| VGG16_torchvision | 68.76  | /            |
| VGG16_caffe       | 69.85  | 69.9         |

Discussion on training set size

| Dataset                    | Map(%) | Paper Map(%) |
| -------------------------- | ------ | ------------ |
| PASCAL VOC 2007            | 69.46  | 69.9         |
| PASCAL VOC 2007 + VOC 2012 | 76.37  | 73.2         |

