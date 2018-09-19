
## Assignment3: Image Captioning with Vanilla RNNs, Image Captioning with LSTMs, Network Visualization, Style Transfer, Generative Adversarial Networks ##

**RecurrentNeuralNetwork -RNN** 
image capation将图片对应一段文字
数据集:MS-COCO
*RNN先验知识*
目的: 解决普通神经网络结果无法处理数据之间的关联性
gradient-vanishing梯度消失: 反向传递的loss不断乘以小于1值对于初始节点造成接近于0的数值
==> 对应:  梯度爆炸
==> 这也是普通RNN无法使用过于久远的信息
==> 解决: LSTM   还没看后面再说吧
这部分的RNN 我没有进行实现  跳过吧

**LSTM**
完成长短期记忆  long short term memory
未完成  跳过

**NetWorkVisualization**
网络可视化 依赖于pytorch
明显的特征提取   做了一点  跳过

**Style Transfer**
风格画像的转递  squeezeNet

**Generative Adversarial Network -GAN**
生成式对抗网络
==》 maximize the probability of the discriminator making incorrect choice
(discriminator is confident)
注意:   关键是loss的计算

deeply-convolutional-GAN 效果好看
