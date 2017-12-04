import pandas
import math
import matplotlib.pyplot as plot
import numpy

class Perceptron(object):
    '''
    iteration_limit: int
    eta: float: learning rate [0:1]
    '''
    
    def __init__(self, eta = 0.01, iteration=10):
        self.eta = eta
        self.iteration_limit = iteration

    def training(self, X, y):
        self.weight = numpy.zeros(1 + X.shape[1])
        self.errors_list = []

        for _ in range(self.iteration_limit):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weight[1:] += update * xi
                self.weight[0] += update
                errors += int(update != 0.0)
            self.errors_list.append(errors)
        return self

    def predict(self, X):
        return numpy.where(self.sign(self.get_input(X)), 1, -1)

    def sign(self, value):
        return value >= 0.0

    def get_input(self, X):
        return numpy.dot(X, self.weight[1:]) + self.weight[0]

def normalize(dataframe):
    result = dataframe.copy()
    for feature_name in dataframe.columns[1:]:
        max_value = dataframe[feature_name].max()
        min_value = dataframe[feature_name].min()
        result[feature_name] = (dataframe[feature_name] - min_value) / (max_value - min_value)
    return result

''' 
import data 
datasource: https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
and shuffle order
''' 
data = pandas.read_csv('wine.data').sample(frac=1)
# normalize data by each column's max and min
data = normalize(data)

y = data.iloc[:, 0].values
y = numpy.where(y == 1, 1, -1) # label class = 1
X = data.iloc[:, 1:].values

perceptron = Perceptron(eta = 0.1, iteration=20)
perceptron.training(X, y)
print(perceptron.weight)

plot.plot(range(0, len(perceptron.errors_list)), perceptron.errors_list, marker='.')
plot.ylabel('errors count')
plot.xlabel('epochs')
plot.title('w/ normolization')
plot.show()