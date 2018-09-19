jupyter addkernel:  python3.6 -m ipykernel install --user --name=cs231n
jupyter deletekernel: jupyter kernelspec remove cs231n_env

## Assignment1: Image Classification, kNN, SVM, Softmax, Neural Network ##

**KNN**
k-cross validated超参数
best k值设定为10
dist[i][j]保存test和train之间的distance  test序号作为rows。
可以用无循环的方式计算dist距离

**SVM**
1. 数据损失 + 正则化损失
2. SGD batch_size是怎么训练的？   
相当于很多条数据一起训练分别得到对应的梯度 再求平均 *learning-rate作为梯度的更新值 减少了dw的更新次数
Weight相当于是每一种分类的权重 或者理解为特征？
将Wextend到0~255 大概可以看出其图像轮廓 线性Svm分类器需要调整W 使得正确的分类和错误的分类差距大于delta
3.SVM分类器使用的是折叶损失(hinge loss)，有时候又被称为最大边界损失(max-margin loss) Softmax分类器使用的是交叉熵损失(corss-entropy loss)

**SoftMax分类器**
Softmax的输出（归一化的分类概率）更加直观，并且从概率上可以解释
类似于SVM流程
1.1为Softmax分类器实现完全向量化的损失函数
1.2为其分析梯度实现完全向量化的表达式
1.3用数值梯度检查你的实现
1.4使用验证集来调整学习率和正则化强度
1.5使用SGD优化损失函数
1.6可视化最终学习的重量

**TwolayerNet**
先验知识:
第一层Hidden 经过W1 b1的线性变化后  使用RELU的激活函数得到H_output
输出层是没有激活函数的  直接w2 b2的线性变化
代码:
1. f = lambda W: net.loss(X, y, reg=0.1)[0]
FIXME: 关于变量W的函数; [0]怎么理解呢？
2.超参数:
hidden-size设置  learning-rate设置  lamda参数设置
hs 100 lr 1.100000e-03 reg 7.500000e-01

注: hidden-size相当于是第一层的神经元数量  
W0的大小=inputSize * hidddenSize


参考:
https://github.com/lightaime/cs231n
