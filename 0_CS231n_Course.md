*0_图像基础*
1.k-Nearest-Neighber: 最相似的前k个元素中 找到属于同一类别最多的label
2.高维数据的PCA降维 再使用KNN
3.参数K值  向量距离的L1(差值Sum) or L2(差值平方和)定义

*1_线性分类器*
1.1 f(xi,w,b) = W * xi + b; weights_vector(矩阵) and bias_vector(偏差向量,列向量)
1.2 weights_vector每一行就是一个类的分类器
1.3 Xi是列向量(抽象为高维空间的点)
1.4 权重W & 偏差b的合并为一个矩阵的处理
1.5 normalization减去平均值来中心化数据 + 归一化
2.1 loss function or cost funciotn用来调整Weight矩阵,调整到最后的 评分列向量中的正确分类的评分比其它的类别的评分至少高△
2.2 正则化惩罚 避免大权重的W的出现
完整Loss = 1/N * Sum(Li)这一项是dataLoss + λ * 正则化损失(所有参数的平方和)  
2.3 data_train做出准确分类预测和让Min loss_funciton这两件事是等价的
3.1 SVM损失函数 关键idea: 正确分类和错误分类之间的差值大于某个阈值
3.2 softMax损失函数 关键idea: 交叉熵; 得到每一个结果分类正确的概率
如何高效的计算Min_loss的参数呢？

*2_最优化 optimization*
寻找最优的Weights矩阵
2.1 随机搜索  随机本地搜索  跟随梯度(偏导最大的方向)
2.2 有限差值近似计算 、可导微分计算 学习率 learning-rate 10^-5感觉效果比较好
2.3 batchs训练权重weights
evaluate_gradient函数 对于batch究竟是如何处理的？
2.3 核心idea: 计算损失函数关于权重的梯度(=》Min Loss)

*3_反向传播 Backprop*
3.1 核心Idea: 分段计算偏导数   损失函数可以拆分成加法门  取最大值门 乘法门等,通过链式法则计算局部偏导数  
3.2 例如:给定的X_train,y_train可以通过链式法则计算(保存前向传播的中间变量)  x的偏导数即dfdx = dfdq * dqdx  (存在多分支则进行累加)
fixme: 如果局部梯度很小或者很大  会影响相乘的结果
3.3 tips: 矩阵or向量的偏导数  维度和原来的size一定是一样的
3.4 通过BackProp 来计算NeuralNetWork中的各个节点的损失函数的偏导数了

*4_NN基础I_SetUp Architecture*
4.1 例如3层NN s=W3 * max(0,W2 * max(0,W1x))其中W1,,W2,W3是需要学习的参数。中间隐藏层的尺寸size是超参数  
```python
class Neuron(object):
  # ... 
  def forward(inputs):
    """ 假设输入和权重是1-D的numpy数组，偏差是一个数字 """
    cell_body_sum = np.sum(inputs * self.weights) + self.bias
    firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) # sigmoid激活函数
    return firing_rate
```
4.2 forward中激活函数 非线性函数。sigmod的饱和使得梯度消失,不建议用。学习率较大时,RELU可能遇到较多的死亡神经单元。
tips:设置好LearningRate or 使用Leakly Relu or 使用Maxout效果更好一些
4.3 NN本质是通用的函数近似器理论上可以用来模拟所有连续函数
4.4 确定网络尺寸 weights bias_vector
4.5 更大的网络效果一般更好 ;大网络的overfitting通过正则化强度等等手段加以控制

*5_NN基础II_Data and Loss*
1.1 数据预处理:均值减法得到零中心化 =》归一化数据{-1,1}  常用操作
1.2 PCA;白化Whitening  不常用了
1.3 randn(n)零均值的高斯分布  n是input数据的数量。 bias一般设置为0
目前认为 RELU的最佳Weights初始化为np.random.randn(n) * sqrt(2.0/n)
含义: 标准差为2/n的高斯分布 n是输入层的神经元数目
1.4 batch Normalization批量归一化 全连接层和激活函数之间添加BatchNorm层用来归一化预处理
Regularization损失 通过调整Weights避免overfitting
2.1 L2正则化 w += -lambda * W
2.2 Max norm Constraints最大范式约束
2.3 Dropout神经元随机失活 or DropConnect权重向量被随机设为零
```python
""" 
反向随机失活: 推荐实现方式.
在训练的时候drop和调整数值范围，测试时不做任何事.
"""
p = 0.5 # 激活神经元的概率. p值更高 = 随机失活更弱
def train_step(X):
  # 3层neural network的前向传播
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # 第一个随机失活遮罩. 注意/p!
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # 第二个随机失活遮罩. 注意/p!
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  # 反向传播:计算梯度... (略)
  # 进行参数更新... (略)
def predict(X):
  # 前向传播时模型集成
  H1 = np.maximum(0, np.dot(W1, X) + b1) # 不用数值范围调整了
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3
 ```
 dataLoss
 3.1分类问题和回归问题的Loss; L2损失;softMax损失
 预处理数据+初始化模型 下一节进行算法的学习过程
  
*6_NN基础III_Learning and Evaluation*
1.1 梯度检查 相对误差;不要让正则化Loss吞没数据;调试梯度记得关闭dropout随机失活
1.2 epochs周期
1.3 训练集准确率和验证集准确率中间的空隙指明了模型过拟合的程度。 Lr设置不恰当;或者正则化惩罚权重较低
2.1 Weight更新方式  
*TODO* 已经通过反向传播计算得到了所有unit的梯度
```
负梯度方向最小化LossFunc
2.1.1 普通更新 W += - learning_rate * dw
2.1.2 普通动量更新 Momentum用于深度网络;  v一般初始化为0   mu为0.9
v = mu * v - learning_rate * dw # 与速度融合
w += v # 与位置融合
2.13 Nesterov动量
v_prev = v # 存储备份
v = mu * v - learning_rate * dx # 速度更新保持不变
x += -mu * v_prev + (1 + mu) * v # 位置更新变了形式
```
2.2 动态调整learning-rate方法-常用Adam用来更新Weights
```
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
推荐的参数值eps=1e-8, beta1=0.9, beta2=0.999
```
3. 超参数调优 
3.1 learning-rate初始化 & 周期衰减
3.2 正则化Loss的惩罚(e.g. DropOut)
3.3 随机搜索寻找最优超参数优于网格搜索;其他一些情形的技巧详细请看Notes

*7_CNN基础*
目的: 大尺寸的图像NN全连接层导致Weight过多
1.1 卷积层conv receptive_filed=>计算卷积层中每个神经元的Weights区域大小。同一深度方向的感受野相同。输出数据体超参数: depth stride zero-padding
1.2 同一个深度切片的使用相同的Weight  kernel.
1.3 输出数据体的空间尺寸:  正方形宽=高=filter滤波器的F,切片的深度=(W-F+2P)/S+1
Winput图像尺寸  F滤波器尺寸  P零填充长度 Stride步长
*FIXME:到底怎么算Filter  怎么算Weight size*
```
卷积层超参数:
滤波器数量K      //TODO: 输出actionMap的个数  
滤波器空间尺寸F  
步长Stride  
零填充数量P  
```
2.1 pooling减少网络间的传递参数
downsample 降采样
一般采用F=2 S=2的滤波器进行Max_pool  相当于丢弃了75%数据(深度切片的个数不变)
2.2 其他思路=抛弃汇聚层:通过在卷积层中使用更大的步长来降低数据体的尺寸
3. 全连接层 和 卷积层的相互转换  这里我没看
4. 卷积层小尺寸filter3 * 3 stride=1 padding=1  或者 filter5 * 5 stride=1,p=2
5. leNet AlexNet ZFNet GoogleNet VGGNet  ResNet等等卷积网络
·

**参考**
http://cs231n.github.io/
https://zhuanlan.zhihu.com/p/21930884
