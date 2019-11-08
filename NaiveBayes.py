class NaiveBayesClassifier:
    def __init__(self):
        self.data = []
        self.labels = []
        self.p = {}
        self.feature_class_count = {}
        self.target_class_count = {}

    def fit(self, data_train, data_labels):
        self.data = data_train
        self.labels = data_labels

        # Get the probability and count of the labels. Probability here is just count/total
        for label in self.labels:
            if label not in self.target_class_count:
                self.target_class_count[label] = 1
            else:
                self.target_class_count[label] += 1
        for key in self.target_class_count.keys():
            self.p["target=" + str(key)] = self.target_class_count[key] / len(self.labels)
            self.feature_class_count["target=" + str(key)] = self.target_class_count[key]

        # Get the count of all of the features and their classes, given the labels and their classes
        for feature_index in range(len(self.data[0])):
            for datapoint_index in range(len(self.data)):
                datapoint = self.data[datapoint_index]
                for target_class in self.target_class_count.keys():
                    index_string = "feature" + str(feature_index) + "=" + str(
                        datapoint[feature_index]) + "|" + "target=" + str(target_class)
                    if index_string not in self.feature_class_count.keys():
                        self.feature_class_count[index_string] = 0
                    if self.labels[datapoint_index] == target_class:
                        self.feature_class_count[index_string] += 1
                    self.p[index_string] = self.feature_class_count[index_string] \
                                         / self.target_class_count[target_class]

    def predict(self, data):
        out = []
        for datapoint in data:
            probs = []
            for target_class in self.target_class_count.keys():
                prob = [target_class, self.p["target=" + str(target_class)]]
                for feature_index in range(len(datapoint)):
                    index_string_1 = "feature" + str(feature_index) + "=" + str(datapoint[feature_index])
                    index_string_2 = "target=" + str(target_class)
                    prob[1] *= self.p[index_string_1 + "|" + index_string_2]
                probs.append(prob)
            probs = sorted(probs, key=lambda a: -a[1])
            out.append(probs[0][0])
        return out
