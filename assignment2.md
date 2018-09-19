参考:
https://github.com/wjbKimberly/cs231n_spring_2017_assignment/blob/master/assignment2/PyTorch.ipynb

tuple  读作too-pull

## Assignment2: Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets ##

**1.Fully connected NN**
模块化层 从而可以实现任意深度的NeuralNetWorks
x.shape[0] == 矩阵X的行数目
先验知识:
1.inputShape转换为一维Row e.g. (4,5,6) = (,120)
2.神经元num=3 则bias=3  同时这一层的Weight=(120*3)  ?是不是这样？
3.Weight matrix=(上一层神经元数量,下一层神经元数量)
e.g. Weight的matrix  = weight_scale * np.random.randn(layer_dim[0],layer_dim[1])
*trick*
函数:RELUForward  RELUBackward
函数:SGD_momentum 动量式梯度下降 更新Weight matrix

**2.批量标准化** 
网络每一层的数据进行标准化 
这部分简单做了   具体使用还不太懂   放在后面再说吧  

**3.Dropout**
随机失活 *mask 提高泛化能力
注意: predict函数中不进行随机失活，但是对于两个隐层的输出都要乘以p，调整其数值范围。、
 p值更高 = 随机失活更弱
3.1 开启随机失活后   train全部数据中的准确率下降的值  < 从train中抽取样本中的准确率    ==> 说明随机失活确实是有效的

**4.CNN**
这部分实现我直接跳过了  

**5.pytorch**
参考:
https://github.com/haofeixu/standford-cs231n-2018/blob/master/assignment2/PyTorch.ipynb
1.函数torch.mm(tensor1,tensor2) 矩阵tensor相乘
2.conv2d函数中in_channel out_channel含义？？

```python
# With square kernels and equal stride
#保证filter是in_channel的倍数
filters = torch.randn(6filter的数量,3in_channel通道,3filter高,3filter宽)
inputs = torch.randn(minibatch,3数据通道,5,5)
F.conv2d(inputs, filters, padding=1)
```
3.卷积后的全连接层Weight size如何计算？
CN通道数*图片size   例如: 9 * 32 *32

4.计算   手动写张量(wx计算scores  容易出错)
two-fully-connected-layer
three-convnet-layer(conv2d函数的要熟悉  + 计算Fully-connected-layer Weights)
注:  这些参数的计算一定要特别熟悉
conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b

5.Pytorch-API  
实现虚基类nn.Module
demo==> two-fully-connected-layer & three-convnet-layer
5.1 注意: nn.Moudle中的nn.Conv2d(in_channels, out_channels, 
    kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    kernel_size如果是int则high=weight 也可以是tuple
*convNet - FullyConNet之间需要调用flatten处理*
```python
def forward(self, x):
       scores = None
       relu1 = F.relu(self.conv1(x))
       relu2 = F.relu(self.conv2(relu1))
       #TODO: 将activity-map激活图 转化成二维matrix  在进行FC的linear计算
       scores = self.fc(flatten(relu2))
```

5.2 优化器
```python
# you can use Nesterov momentum in optim.SGD
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                     momentum=0.9, nesterov=True)
```

5.3 批量标准化 (对于out_channel处理)
nn.BatchNorm2d(32)
BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
