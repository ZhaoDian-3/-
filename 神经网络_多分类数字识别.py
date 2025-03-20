from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 测试集，画图对预测值和实际值进行比较
def test_validate(x_test, y_test, y_predict, classifier):
    """
    可视化测试集的真实值与预测值，并展示分类器的准确率。

    参数:
        x_test (numpy.ndarray): 测试集的特征数据。
        y_test (numpy.ndarray): 测试集的真实标签。
        y_predict (numpy.ndarray): 测试集的预测标签。
        classifier (object): 训练好的分类器，需实现 score 方法。
    """
    # 创建一个范围与测试样本数量相同的x轴
    x = range(len(y_test))

    # 绘制真实值，使用红色圆点标记
    plt.plot(x, y_test, "ro", markersize=5, zorder=3, label=u"True Values")

    # 绘制预测值，使用绿色圆点标记，并显示分类器的准确率
    plt.plot(
        x,
        y_predict,
        "go",
        markersize=8,
        zorder=2,
        label=u"Predicted Values, $R$=%.3f" % classifier.score(x_test, y_test)
    )

    # 添加图例，位于左上角
    plt.legend(loc="upper left")

    # 设置x轴标签
    plt.xlabel("Sample Index")

    # 设置y轴标签
    plt.ylabel("Digit Label")

    # 显示图形
    plt.show()


# 神经网络数字分类
def multi_class_nn():
    """
      使用多层感知器（MLP）分类器进行多分类任务（如数字分类）
      过程:
          1. 加载数字数据集
          2. 对数据进行标准化处理
          3. 将数据集划分为训练集和测试集
          4. 使用MLP神经网络进行训练
          5. 测试模型的准确率并进行预测
          6. 绘制真实值与预测值的比较图
      """
    # 加载手写数字数据集
    digits = datasets.load_digits()

    # 提取特征数据（每个样本的像素值）和目标标签（数字0-9）
    x = digits['data']
    y = digits['target']

    # 对特征数据进行标准化处理，使其均值为0，方差为1
    ss = StandardScaler()
    x_regular = ss.fit_transform(x)

    # 将数据集划分为训练集和测试集，测试集占10%
    x_train, x_test, y_train, y_test = train_test_split(
        x_regular, y, test_size=0.1, random_state=42
    )

    # 创建多层感知器分类器实例
    # 参数说明:
    # - solver='lbfgs': 使用拟牛顿法优化器，适用于小型数据集
    # - alpha=1e-5: L2正则化参数，防止过拟合
    # - hidden_layer_sizes=(5,): 隐藏层包含5个神经元
    # - random_state=1: 设置随机种子，保证结果可重复
    clf = MLPClassifier(
        solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(5,),
        random_state=1
    )

    # 训练多层感知器分类器
    clf.fit(x_train, y_train)

    # 获取并打印训练集上的准确率
    r = clf.score(x_train, y_train)
    print("Training Accuracy (R值):", r)

    # 使用训练好的模型对测试集进行预测
    y_predict = clf.predict(x_test)

    # 打印预测结果和真实标签
    print("Predicted Labels:", y_predict)
    print("True Labels:", y_test)

    # 可视化测试集的预测结果与真实结果的比较
    test_validate(x_test=x_test, y_test=y_test, y_predict=y_predict, classifier=clf)


# 调用多类神经网络分类函数
multi_class_nn()
