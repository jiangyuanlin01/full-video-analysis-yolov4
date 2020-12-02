# full-video-analysis（视频全量目标分析与建模）

#### 项目介绍

人工智能结合视觉分析，极大推动各行业视觉应用，人脸识别、车辆测别、车辆智能驾驶等。

本项目，基于TensorFlow+yolov4，针对1080P视频，视频内容街景（行车记录仪、电影等拍摄）内容，利用视觉分析技术，对高分辨率视频进行视频图像，对街景或高楼的高清视频进行目标检测和统计，致力于创新深度学习算法模型解决方案。

#### 技术架构
| 基础环境          | 核心算法 |
| :---------------- | -------- |
| Python 3.6        | YoloV4   |
| TensorFlow 1.13.1 | DeepSort |
| CUDA 10.0         |          |

#### 算法解析

- **YOLOV4**

  ​       Yolo-V4的主要目的在于设计一个能够应用于实际工作环境中的快速目标检测系统，且能够被并行优化，并没有很刻意的去追求理论上的低计算量（BFLOP）。同时，Yolo-V4的作者希望算法能够很轻易的被训练，也就是说拥有一块常规的GTX-2080ti或者Titan-XP GPU就能够训练Yolo-V4，同时能够得到一个较好的结果。

  ![](http://jerusalem01.gitee.io/images-bed/images/full-video-analysis/image-20201202161918594.png)

  - **Related work**

  ![image-20201202162208026](http://jerusalem01.gitee.io/images-bed/images/full-video-analysis/image-20201202162208026.png)

  **Input**：算法的输入。包括整个图像，一个patch，或者image pyramid。

  **Backbone**：提取图像特征的部分。由于图像中的浅层特征比较类似，例如提取边缘，颜色，纹理等。因此这部分可以很好的借鉴一些设计好并且已经训练好的网络，例如（VGG16,19，ResNet-50, ResNeXt-101, Darknet53）, 同时还有一些轻量级的backbone（MobilenetV1,2,3 ShuffleNet1,2）。

  **Neck**：特征增强模块。前面的backbone已经提取到了一些相关的浅层特征，由这部分对backbone提取到的浅层特征进行加工，增强，从而使得模型学到的特征是想要的特征。这部分典型的有（SPP，ASPP in deeplabV3+，RFB，SAM），还有一些（FPN, PAN, NAS-FPN, BiFPN, ASFF, SFAM）。

  **Head**：检测头。算法最关键的部分，就是来输出想要的结果。例如想得到一个heatmap，那就增加一些反卷积层来一层一层反卷积回去。如果想直接得到bbox，那就可以接conv来输出结果，例如Yolo，ssd。亦或是想输出多任务（mask-RCNN），那就输出三个head：classification，regression，segmentation。

  因此，一个检测算法可以理解为：**Object Detection = Backbone + Neck + Head**

  

  - **网路结构选择**

  ​       网络结构选择是为了在输入分辨率、网络层数、参数量、输出滤波器数之间寻求折中。研究表明：**CSPResNeXt50在分类方面优于CSPDarkNet53，而在检测方面反而表现要差**。

  ​      网络主要结构确定后，下一个目标是选择额外的模块以提升感受野、更好的特征汇聚模块（如FPN、PAN、ASFF、BiFPN）。对于分类而言最好的模型可能并不适合于检测，相反，检测模型需要具有以下特性：

   1.更高的输入分辨率，为了更好的检测小目标；

   2.更多的层，为了具有更大的感受野；

   3.更多的参数，更大的模型可以同时检测不同大小的目标。

  一句话就是：**选择具有更大感受野、更大参数的模型作为backbone**。

  下图给出了不同backbone的信息对比。

  ![image-20201202162329025](http://jerusalem01.gitee.io/images-bed/images/full-video-analysis/image-20201202162329025.png)

  从中可以看到：CSPResNeXt50仅仅包含16个卷积层，其感受野为425x425，包含20.6M参数；而CSPDarkNet53包含29个卷积层，725x725的感受野，27.6M参数。

  从理论与实验角度表明：CSPDarkNet53更适合作为检测模型的Backbone。在CSPDarkNet53基础上，添加SPP模块，因其可以提升模型的感受野、分离更重要的上下文信息、不会导致模型推理速度的下降；与此同时，采用PANet中的不同backbone级的参数汇聚方法替代FPN。

  最终的模型为：**CSPDarkNet53+SPP+PANet(path-aggregation neck)+YOLOv3-head = YOLOv4**

  

  - **Tricks选择**

    为更好的训练目标检测模型，CNN模型通常具有以下模块：

     Activations：ReLU、Leaky-ReLU、PReLU、ReLU6、SELU、Swish or Mish

     Bounding box regression Loss：MSE、IoU、GIoU、CIoU、DIoU

     Data Augmentation：CutOut、MixUp、CutMix

     Regularization：DropOut、DropPath、Spatial DropOut、DropBlock

     Normalization：BN、SyncBn、FRN、CBN

     Skip-connections： Residual connections, weighted residual connections, Cross stage partial connections

      作者从上述模块中选择如下：

     激活函数方面选择Mish；正则化方面选择DropBlock；由于聚焦在单GPU，故而未考虑SyncBN。

  

  - **YOLOV总结**

    Backbone：CSPDarkNet53

     Neck：SPP，PAN

     Head：YOLOv3

     Tricks（backbone）：CutMix、Mosaic、DropBlock、Label Smoothing

     Modified（backbone）: Mish、CSP、MiWRC

     Tricks（detector）：CIoU、CMBN、DropBlock、Mosaic、SAT、Eliminate grid sensitivity、Multiple    Anchor、Cosine Annealing scheduler、Random training shape

     Modified（tector）：Mish、SPP、SAM、PAN、DIoU-NMS

  

- **DeepSort**

​       DeepSort是在Sort目标追踪基础上的改进。引入了在行人重识别数据集上离线训练的深度学习模型，在实时目标追踪过程中，提取目标的表观特征进行最近邻匹配，可以改善有遮挡情况下的目标追踪效果。同时，也减少了目标ID跳变的问题。

**核心思想：**

1. 轨迹处理和状态估计
2. 指派问题
3. 级联匹配
4. 深度特征描述器

**卡尔曼滤波**

卡尔曼滤波是用来对目标的轨迹进行预测，并且使用确信度较高的跟踪结果进行预测结果的修正

**匈牙利算法**

匈牙利算法是一种寻找二分图的最大匹配的算法，在多目标检测跟踪问题中可以理解为寻找前后两帧的若干目标的匹配最优解的方法。

**工作过程：**

1.读取当前帧目标检测框的位置以及各检测框图像块的深度特征；

2.根据置信度对检测框过滤，对置信度不够高的检测框和特征予以删除；

3.对检测框进行非极大抑制，消除一个目标多个框；

4.预测：使用kalman滤波预测目标在当前帧的位置。

#### 创新点

- 引入一种新的数据增广方法：Mosaic与自对抗训练；
- 通过GA算法选择最优超参数；

- 对现有方法进行改进以更适合高效训练和推理：改进SAM、改进PAN，CmBN；

- 优化 DeepSort 方法，实现**多类别**多目标实时跟踪检测；

- 对于检测到的目标进行分割，并按类别输出小图到指定目录。

#### 效果演示

![效果图](http://jerusalem01.gitee.io/images-bed/images/full-video-analysis/效果图.png)

#### 使用步骤

##### 1、使用预训练权重

1.1 下载完库后解压，在百度网盘下载yolo4_weights.pth或者yolo4_voc_weights.pth，放入model_data，运行predict.py，输入

```
img/street.jpg
```

可完成预测。
1.2 利用video.py可进行摄像头检测。

##### 2、使用自己训练的权重

2.1 按照训练步骤训练。
2.2 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类。

```
_defaults = {
    "model_path": 'model_data/yolo4_weights.pth',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt',
    "model_image_size" : (416, 416, 3),
    "confidence": 0.5,
    "cuda": True
}
```

2.3 运行predict.py，输入

```
img/street.jpg
```

可完成预测。
2.4 利用video.py可进行摄像头检测。

##### 3、训练步骤

1、本文使用VOC格式进行训练。
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。
4、在训练前利用voc2yolo4.py文件生成对应的txt。
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**

```
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```

6、此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。
7、**在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：

```
classes_path = 'model_data/new_classes.txt'    
```

model_data/new_classes.txt文件内容为：

```
cat
···
```

8、运行train.py即可开始训练。

#### 友情链接

- **[个人博客](http://www.nm83.com)**

- **[码云](https://gitee.com/jerusalem01)**

- **[GitHub](https://github.com/Jerusalem01)**
=======
# full-video-analysis
视频全量分析与建模
>>>>>>> 0a167c9789d594da0339e382f8b4de43b24b8e3d
