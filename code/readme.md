# Physical Attack

### 运行前依赖

- 数据集
- 3d obj 文件，和对应的 mtl纹理文件
- 需要被训练的face的id列表（保存成txt）
- 被训练的face对应的初始内容纹理content，以及内容纹理对应的边缘mask纹理 canny

### 训练

```shell
python train.py [参数]
```

- 训练参数
  - epoch
  - lr
  - batchsize
  - lamb  内容loss的系数lambda
  - d1  对边缘区域的保护系数
  - d2  对非边缘区域的保护系数
  - t  平滑loss的系数
- 路径参数
  - obj  3d obj文件路径
  - faces  可训练faces的文件路径
  - datapath  数据集路径
  - content  内容纹理路径
  - canny  边缘mask纹理路径

训练结果存储在`logs/`目录下相应的文件夹内，其中包括：

- 训练过程中的一些图片输出
  - cam.jpg    gradcam图
  - cam_b.jpg
  - cam_p.jpg
  - mask.png  
  - test_total.jpg  模型输入图片展示
  - texture2.png
- loss.txt   训练过程中的loss变化
- texture.npy   训练好的纹理文件

### 测试

```shell
python test.py --texture=path_to_texture
```

测试结果存储在本目录下的`acc.txt`文件中，为使用resnet152模型测试的正确率acc的结果