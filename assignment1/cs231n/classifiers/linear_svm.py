import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)          #计算得到给定W下的线性liner  预测值
    correct_class_score = scores[y[i]] #正确label对应的分数
    for j in range(num_classes):
      if j == y[i]:
        continue
      #真实值 预测值之间的差距
      #为什么margin大于0的情况下才进行更新dW
      #解释: svm线性分类器只关心正确分类和预测分类差距在某个阈值之间的数据  同时开始计算损失值
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # 添加dw的变化   这里我不明白梯度是如何更新的？  为什么使用了X[i]
        #解释:  因为当前关于W的梯度是 xi   因为是batch训练  加起来  后面再求平均  再乘以learningrate更新Weights(基类的train方法)
        #这里都是分类出错的数据  我们就朝着梯度上升的方向更新？？
        dW[:,j] += X[i].T
        #print( X[i].T)  #len=3073
        #print( len(X[i].T))  #len=3073
        # 分类的正确的数据  也需要更新W  朝着梯度下降的方向更新
        dW[:,y[i]] += -X[i].T 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  
      # Add regularization to the loss.
    #特别注意乘以 1/2
    #data loss 以及正则化损失(限制Weight的权重)来作为loss函数
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
 
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
  margins = np.maximum(0, scores - correct_class_scores +1)
  margins[range(num_train), list(y)] = 0

#data loss 以及正则化损失(限制Weight的权重)来作为loss函数
  loss = np.sum(margins)  / num_train + 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # pass
  # 不明白这里的Weights的梯度是如何计算
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + reg*W
    
    
 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
