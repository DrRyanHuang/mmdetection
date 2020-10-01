## 安装教程

### 安装依赖

- `Linux` or `macOS` (`Windows`暂时还不被官方支持)
- `Python 3.6+`
- `PyTorch 1.3+`
- `CUDA 9.2+` (如果`PyTorch`是由源码编译的，那么`CUDA 9.0`也是兼容的)
- `GCC 5+`
- [`mmcv`](https://github.com/open-mmlab/mmcv)



### 安装`mmdetection`

##### a. 创建一个 conda 虚拟环境，并激活它

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```



##### b. 从[官方指南](https://pytorch.org/)中安装 `PyTorch` 和`torchvision` ，例如：

```shell
conda install pytorch torchvision -c pytorch
```

**注意**：确保你的 `compilation CUDA`版本和`runtime CUDA`版本相匹配
您可以在[PyTorch网站](https://pytorch.org/)上查看受支持的CUDA版本的预编译软件包。



`例1` 如果你在`/usr/local/cuda`安装了`CUDA 10.1`，同时想安装`PyTorch 1.5`，那么你需要安装与`CUDA 10.1`相对应的`PyTorch`

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`例2` 如果你在`/usr/local/cuda`安装了`CUDA 9.2`，同时想安装`PyTorch 1.3.1`，那么你需要安装与`CUDA 9.2`相对应的`PyTorch`

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

如果你从源代码编译了`PyTorch`，而不是安装预构建的软件包，那么你可以使用更多的`CUDA`版本例如`9.0`



##### c. 安装`mmcv`, 建议直接安装如下所示预编译的`mmcv`，(吐槽注：对`win10`支持真不好, 只有版本`1.1.3`有)

```shell
pip install mmcv-full==latest+torch1.6.0+cu102 -f https://download.openmmlab.com/mmcv/dist/index.html
```

可以参考[这里](https://github.com/open-mmlab/mmcv#install-with-pip)去查看`MMCV`与不同版本的`PyTorch`和`CUDA`的兼容对应关系
(可选) 当然可以选择通过以下命令从源代码编译`mmcv`

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
cd ..
```

或者直接运行hhh

```shell
pip install mmcv-full
```

**值得注意的点**:

1. 从`MMDetection 2.0`开始，不同版本的`MMDetection`所需的`MMCV`版本如下。 请安装正确版本的`MMCV`，以避免安装问题。

| `MMDetection version` |    `MMCV version`    |
|:-------------------:|:-------------------:|
| `master`            | `mmcv-full>=1.1.1, <=1.2` |
| `2.4.0`             | `mmcv-full>=1.1.1, <=1.2` |
| `2.3.0`             | `mmcv-full==1.0.5` |
| `2.3.0rc0`          | `mmcv-full>=1.0.2`  |
| `2.2.1`             | `mmcv==0.6.2`       |
| `2.2.0`             | `mmcv==0.6.2`       |
| `2.1.0`             | `mmcv>=0.5.9, <=0.6.1` |
| `2.0.0`             | `mmcv>=0.5.1, <=0.5.8` |

2. 如果你之前安装过`mmcv`，那么需要先运行 `pip uninstall mmcv` 去写在`mmcv`

  如果`mmcv` 和`mmcv-full`都被安装了，那么将会报错： `ModuleNotFoundError`



##### d. 克隆`mmdetection`仓库.

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```



##### e. 安装`requirements`，然后安装`mmdetection`。

(为了更好地与我们的仓库兼容，我们通过`github`仓库安装`pycocotools`，而不是通过`pypi`安装)

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

如果你在`macOS`上编译`mmdetection`环境，用这一句代替最后一个命令：

```shell
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

**注意**:

1. 在上述**步骤d**中，`git commit id`将被写入版本号中，例如：`0.6.0+2e7045c`. 版本号将被保存在训练模型中.

2. 建议您每次从`github`提取一些更新时都运行**步骤d**。 如果修改了`C ++/CUDA`内核，则此步骤是**必须的**。

   > 重要：如果你用不同的`CUDA/PyTorch`版本重新安装了`mmdet`，一定要移除`./build`文件夹

   ```shell
   pip uninstall mmdet
   rm -rf ./build
   find . -name "*.so" | xargs rm
   ```

3. 按照上述说明，`mmdetection`将以`dev`模式安装，对代码进行的任何本地修改都将生效而无需重新安装 (除非您提交了一些`commits`并希望更新版本号)。

4. 如果你想要的使用`opencv-python-headless` 而不是`opencv-python`，你可以在安装`MMCV`之前安装它

5. 一些依赖是可选的，简单运行`pip install -v -e .`仅安装最低运行需求依赖。

   要使用诸如`albumentations`和`imagecorruptions`之类的可选依赖项，请使用`pip install -r requirements/optional.txt`手动安装，或者在调用`pip`时指定所需的附加项 (例如`pip install -v -e .[optional] `)。额外的有效字段为： `all`, `tests`, `build`和`optional`.

### 只安装CPU版本

这些代码用于在仅仅使用CPU的环境(CUDA不可用的环境)运行

想看例子的话，在CPU模式下运行`demo/webcam_demo.py`试试
但是在CPU模式下，这些功能不可以用：

- Deformable Convolution
- Deformable ROI pooling
- CARAFE: Content-Aware ReAssembly of FEatures
- nms_cuda
- sigmoid_focal_loss_cuda

所以当你使用一个包含`deformable convolutiond`的模型进行推理时，你会遇到一个`error`
**注意**: 在CPU模式下，我们为`RoIPool`和`RoIAlign`实时设置了`use_torchvision = True`。



### 其他可选项: Docker镜像

我们提供了 [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) 去构建这个镜像.

```shell
# build an image with PyTorch 1.5, CUDA 10.1
docker build -t mmdetection docker/
```

运行脚本：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```



### 安装`mmdetection`的全部脚本

这里是使用`conda`安装 `mmdetection`的全部脚本

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install the latest mmcv
pip install mmcv-full

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```



### 如何使用多个版本的`MMDetection`

训练和测试脚本已经修改了`PYTHONPATH`，以确保脚本使用当前目录中的 `MMDetection`

要使用安装在指定环境中而不是正在使用的默认`MMDetection`，则可以在这些脚本中删除以下行：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
