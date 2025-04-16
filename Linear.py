class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weight = 0  # 权重 (w)
        self.bias = 0    # 偏置 (b)

    def fit(self, X, y):

        n_samples = len(X)

        for _ in range(self.n_iterations):
            # 计算预测值
            y_predicted = [self.weight * x + self.bias for x in X]

            # 计算损失函数的梯度
            dw = -(2 / n_samples) * sum([X[i] * (y[i] - y_predicted[i]) for i in range(n_samples)])
            db = -(2 / n_samples) * sum([y[i] - y_predicted[i] for i in range(n_samples)])

            # 更新权重和偏置
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):

        return [self.weight * x + self.bias for x in X]


X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# 创建并训练模型
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# 输出训练后的参数
print(f"训练后的权重 (w): {model.weight}")
print(f"训练后的偏置 (b): {model.bias}")

# 测试模型
test_X = [11, 12, 13]
predictions = model.predict(test_X)
print(f"测试数据 {test_X} 的预测值: {predictions}")