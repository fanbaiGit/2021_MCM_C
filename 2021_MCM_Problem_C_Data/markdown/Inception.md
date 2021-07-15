# 图像识别

> 对于题目提供的4000多张照片，我们采用了**机器学习**的方法对图像进行识别并且进行分类，由于时间有限再加上从零搭建机器学习模型的难度不小，所以采用了**Tensorflow**的已有的**Inception模型**，将4000多张照片分成训练集和测试集，利用**迁移学习**的方法在**Inception模型**的基础上进行训练和测试，训练出自己的图像识别AI。

## 1 迁移学习

迁移学习(Transfer Learning)试图使用已经训练好的模型参数帮助新模型训练新的数据集。当数个训练任务及数据有一定程度的相关性，使用迁移学习可以加快新模型的学习而不用花费大量时间和样本从头开始训练。我们的任务是一个图像识别任务，这与Google训练好的Inception模型有较大的相关性，而我们的样本又较少，因此基于Inception模型进行迁移学习是个不错的选择。

![TransferLearning](https://zhaomenghuan.js.org/assets/img/tensors_flowing.4a67e129.gif"迁移学习")

## 2 Inception模型

Inception模型是一种用于图像分类的卷积神经网络模型。这是一个多层、有着极其复杂结构的卷积神经网络。该模型可以识别超过1000种物品，但并不包括我们想要的动漫人物。Github上有提到Inception模型训练是在一台有128GB RAM和8块Tesla K40的电脑上进行的(We targeted a desktop with 128GB of CPU ram connected to 8 NVIDIA Tesla K40 GPU)。如果你想在你的个人电脑上尝试从零开始训练一个这种量级的神经网络，你可能需要数个星期才能完成训练，还有很大的可能性出现run out of GPU memory或run out of CPU memory导致训练失败。在这里我们将尝试使用Tensorflow提供的retrain.py训练整个神经网络的最后一层，即决策层/分类层，而倒数第二层被称为Bottlenecks(瓶颈层)。我们将利用Bottlenecks产生的有效数据，供给最后的决策层/分类层做出最后的分类预测。

