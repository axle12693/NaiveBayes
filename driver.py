import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from NaiveBayes import NaiveBayesClassifier


def one_hot(feature):
    classes = []
    out = []
    for feature_class in feature:
        if feature_class not in classes:
            classes.append(feature_class)
        out.append(classes.index(feature_class))
    onehot_out = [[] for i in range(len(out))]
    for i in range(len(onehot_out)):
        onehot_out[i] = [0] * len(classes)
        onehot_out[i][out[i]] = 1
    return np.array(onehot_out)


def load_car():
    car = np.ndarray.tolist(np.transpose(pd.read_csv("car.data").values))
    targets = car[6]
    for i in range(len(car)):
        car[i] = one_hot(car[i])
    car = car[:6]
    # np can't handle transposing this array for some reason... so I'll transpose it myself
    new_car = []

    for row_index in range(len(car)):
        for el_index in range(len(car[0])):
            if len(new_car) < el_index + 1:
                new_car.append([])
            new_car[el_index].append(np.ndarray.tolist(car[row_index][el_index]))

    for row_index in range(len(new_car)):
        new_row = []
        for feature_index in range(len(new_car[row_index])):
            new_row += new_car[row_index][feature_index]
        new_car[row_index] = new_row
    return new_car, targets


print("Car")

car_data, car_targets = load_car()

x_train, x_test, y_train, y_test = train_test_split(car_data, car_targets, test_size=0.3)

classifier = GaussianNB()
classifier.fit(x_train, y_train)
predicted = classifier.predict(x_test)

total_correct = 0
total = 0
for i in range(len(predicted)):
    if np.array_equal(predicted[i], y_test[i]):
        total_correct += 1
    total += 1
print("Accuracy of existing classifier: " + str(total_correct/total))


classifier = NaiveBayesClassifier()
# x_train = [["comedy", "deep", "yes"],
#            ["comedy", "shallow", "yes"],
#            ["drama", "deep", "yes"],
#            ["drama", "shallow", "no"],
#            ["comedy", "deep", "no"],
#            ["comedy", "shallow", "no"],
#            ["drama", "deep", "no"]]
# y_train = ["low", "high", "high", "low", "high", "high", "low"]

classifier.fit(x_train, y_train)
predicted = classifier.predict(x_test)

total_correct = 0
total = 0
for i in range(len(predicted)):
    if predicted[i] == y_test[i]:
        total_correct += 1
    total += 1
print("Accuracy of my classifier: " + str(total_correct/total))
