
#encoding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Perceptron(object):
    """
        eta:学习率
        n_iter:权重向量训练的次数
        w_：神经分叉权重向量
        errors_：用于记录神经元判断出错的次数
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        pass


    def net_input(self, X):
        """
            做点积运算
            z = W0*1 + W1*X1 +.... Wn*Xn
        """
        """
              计算净输入（加权和）

              参数:
                  X (numpy.ndarray): 输入特征向量。

              返回:
                  float: 净输入值，即 W0*1 + W1*X1 + ... + Wn*Xn
              """
        # np.dot(X, self.w_[1:]) 计算输入向量与权重向量的点积（不包括偏置）
        # self.w_[0] 是偏置项，加到点积结果上
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass

    def predict(self, X):
        """
               预测类别标签

               参数:
                   X (numpy.ndarray): 输入特征向量。

               返回:
                   numpy.ndarray: 预测的类别标签（1 或 -1）。
               """
        # 如果净输入 >= 0，则预测为1，否则为-1
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        pass

    """
        输入训练数据，培训神经元
        x 输入样本向量， y对应的样本分类
        x shape[n_samples, n_features]
        x:[[1, 2, 3], [4, 5, 6]]
        x.shape[1] = 2; x.shape[2] = 3
    """
    def fit(self, X, y):
        """
                训练感知器模型

                参数:
                    X (numpy.ndarray): 训练样本特征矩阵，形状为 [n_samples, n_features]。
                    y (numpy.ndarray): 训练样本的目标值，形状为 [n_samples]。

                返回:
                    self: 训练后的感知器模型。
                """
        # 初始化权重向量，包含偏置，初始值全为0
        # 初始化权重为0 加一是因为步调函数阈值    # 初始化权重向量，包含偏置，初始值全为0
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []   # 初始化错误记录列表

        # 进行n_iter次迭代训练
        for _ in range(self.n_iter):
            errors = 0
            # 遍历每一个训练样本
            for xi, target in zip(X, y):
                # 计算预测值与真实值的差异，并乘以学习率
                # update =η * (y - y')
                update = self.eta * (target - self.predict(xi))

                """
                xi是一个向量
                update * xi 等价：
                [▽W(1) = X[1]*update, ▽w(2) = X[2]*update, ▽w(3) = X[3]*update]
                """
                 # 更新权重向量（不包括偏置）
                self.w_[1:] += update * xi
                # 如果有更新，错误计数加1
                self.w_[0] += update
                errors += int(update != 0.0)
                # 记录每次迭代中的错误次数
                self.errors_.append(errors)

                pass
            pass
        pass

from matplotlib.colors import ListedColormap
def plot_decision_regions(x, y, classifier, resolution = 0.02):
    """
       绘制决策区域

       参数:
           x (numpy.ndarray): 特征矩阵，形状为 [n_samples, 2]。
           y (numpy.ndarray): 类别标签。
           classifier (object): 训练好的分类器，需实现predict方法。
           resolution (float): 网格细分的步长。
       """
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # 获取特征1和特征2的最小值和最大值，并扩展边界
    x1_min, x1_max = x[: , 0].min() - 1, x[: , 0].max()
    x2_min, x2_max = x[: , 1].min() - 1, x[: , 1].max()
    #print x1_min, x1_max, x2_min, x2_max
    # 185 从3.3- 6.98 每隔0.02
    # 255 从0  - 5.08 每隔0.02
    # xx1   从3.3 - 6.98 为一行  有185行相同的数据
    # xx2   从0   - 5.08 为一列  第一行全为0 第二行全1 (255, 185)
    # 创建网格点
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # print np.arange(x1_min, x1_max, resolution).shape
    # print np.arange(x1_min, x1_max, resolution)
    # print np.arange(x2_min, x2_max, resolution).shape
    # print np.arange(x2_min, x2_max, resolution)
    # print xx2.shape
    # print xx2
    # 相当于 np.arange(x1_min, x1_max, resolution) np.arange(x2_min, x2_max, resolution)
    # 已经在分类了站如果是3.3 0 则为1 6.94 5.08 则-1
    # 使用分类器对网格点进行预测
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(xx1.ravel())
    print(xx2.ravel())
    print(z)
    # 将预测结果重新塑形为网格的形状
    z = z.reshape(xx1.shape)
    print(z)
    # 在两个分类之间画分界线
    plt.contourf(xx1, xx2, z, alpha = 0.4, cmap = cmap)
    plt.xlim(x1_min, x1_max)
    print("xx1.min()", x1_min)
    plt.ylim(xx2.min(), xx2.max())
    plt.xlabel('length of the huajing')
    plt.ylabel('length of the huaban')
    plt.legend(loc='upper right')
    plt.show()

def main():

    ## X = np.array([[1, 2, 3], [4, 5, 6]])
    # print X.shape
    ## y = [1, -1]
    # 从UCI机器学习库加载鸢尾花数据集
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
    ## print df.head(10)
    # 选择前100个样本，分别为两类鸢尾花（Setosa和Versicolor）
    y = df.loc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)
    # 选择花径长度和花瓣长度作为特征
    x = df.iloc[0:100, [0, 2]].values
    plt.scatter(x[:50, 0], x[:50, 1], color = 'red', marker = 'o', label = 'setosa')
    plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='*', label='versicolor')
    plt.xlabel('length of the huajing')
    plt.ylabel('length of the huaban')
    plt.legend(loc = 'upper right')
    # plt.show()
    p1 = Perceptron(eta = 0.1)
    p1.fit(x, y)
    # plt.plot(range(1, len(p1.errors_) + 1), p1.errors_, marker = 'o')
    # plt.xlabel('Epochs')
    # plt.ylabel('error sort')
    # plt.show()
    plot_decision_regions(x, y, p1)
    pass


if __name__ == '__main__':
    main()
