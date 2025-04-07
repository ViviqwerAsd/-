# 三层神经网络实现的 CIFAR-10 图像分类器
本项目基于 NumPy 手工实现了一个三层神经网络，用于在 CIFAR-10 数据集上完成图像分类任务。不依赖 PyTorch / TensorFlow 等自动微分框架。
模型权重下载地址：https://pan.baidu.com/s/1UhLgHyFwjUsqxrpv5gvHiw?pwd=i46g
## 🚀 使用方法
### 1️⃣ 安装依赖
本项目依赖仅为标准库与 numpy、matplotlib、pandas：

``` bash
pip install numpy matplotlib pandas
``` 
### 2️⃣ 数据准备
请从 CIFAR-10官网 下载并解压数据集，将以下文件放入项目根目录下的 ./cifar-10-batches-py/ 文件夹中：

data_batch_1 ~ data_batch_5
test_batch
batches.meta
或者你也可以使用我已经准备好的 data_loader.py 直接加载本地数据。

### 3️⃣ 训练模型
你可以直接运行以下命令开始训练：

``` bash
python main.py --train
``` 
或使用提供的 run.sh 脚本在后台运行，并自动保存训练日志：

``` bash
bash run.sh
``` 
日志文件默认保存为 train_时间戳.log。

### 4️⃣ 进行测试
使用训练过程中保存的最佳模型权重（默认保存为 cv_best_model.npz），在测试集上评估性能：

``` bash
python main.py --test
``` 
### 5️⃣ 超参数搜索（可选）
自动遍历一系列超参数组合，记录验证准确率并保存最优模型，搜索结果自动保存在 param_search_results.csv 中，并生成多张图像分析不同超参数的影响：

``` bash
python main.py --param_search
``` 
你也可以根据 param_search.py 文件中的 params 字典自定义搜索空间。

## 🔧 可调参数说明（训练/搜索）
hidden_size：隐藏层神经元个数，默认 128 可选 [128, 256, 512, 1024]

learning_rate：学习率，默认 0.01 可选 [0.1, 0.01, 0.001]

reg：L2正则化系数，默认 0.01

dropout_rate：Dropout比例，默认 0

activation：激活函数类型，可选 relu 或 sigmoid

## ✅ 输出内容
最优模型保存路径：cv_best_model.npz

参数搜索记录：param_search_results.csv

参数可视化图像：

param_search_learning_rate_vs_hidden_size.png

param_search_reg_vs_hidden_size.png

param_search_activation_vs_hidden_size.png
