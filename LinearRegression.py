class LinearRegression:
    def __init__(self):
        # Initialize the weight(s) & bias from a random distribution (e.g. normal distribution).
        self.weight: float = 0  # 权重 (w)
        self.bias: float = 0    # 偏置 (b)

    def fit(self, x, y, learning_rate: float = 0.01, n_iterations: int = 1000):
        n_samples: int = len(x)

        for _ in range(n_iterations):
            # 计算预测值
            y_predicted = self.predict(x)

            # 计算损失函数的梯度
            dw = -(1 / n_samples) * sum([x[i] * (y[i] - y_predicted[i]) for i in range(n_samples)])
            db = -(1 / n_samples) * sum([y[i] - y_predicted[i] for i in range(n_samples)])

            # 更新权重和偏置
            self.weight -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, x):
        return [self.weight * a + self.bias for a in x]


def sample_function(x: int) -> int:
    return 2 * x # + a random value.

if __name__ == '__main__':
    x: list[int] = list(range(20))
    y: list[int] = [sample_function(a) for a in x]

    # 创建并训练模型
    # Default argument.
    model = LinearRegression()
    model.fit(x, y, learning_rate=0.1, n_iterations=1000)

    # 输出训练后的参数
    # Print the expected weight(s) & bias.
    print(f"训练后的权重 (w): {model.weight}")
    print(f"训练后的偏置 (b): {model.bias}")
    # Print the expected expression (y = 2 * x).
    # Print the fit expression (y = model.weight * x + model.bias).

    # 测试模型
    test_X = [11, 12, 13]
    predictions = model.predict(test_X)
    print(f"测试数据 {test_X} 的预测值: {predictions}") # And the real value (calculated from the expected expression).

    # Calculate accuracy here.

    # Show the plot here.