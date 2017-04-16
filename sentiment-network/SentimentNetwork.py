# TODO: -Copy the SentimentNetwork class from Project 4 lesson
#       -Modify it according to the above instructions

import time
import sys

import numpy as np
import itertools


# Encapsulate our neural network in a class
# noinspection PyAttributeOutsideInit,PyMethodMayBeStatic,PyShadowingNames
# noinspection PyTypeChecker,PyRedundantParentheses,PyPep8
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1):
        """Create a SentimentNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        np.random.seed(1)
        self.pre_process_data(reviews, labels)
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        review_vocab = set(itertools.chain.from_iterable(r.split(' ') for r in reviews))
        self.review_vocab = list(review_vocab)
        label_vocab = set(labels)
        self.label_vocab = list(label_vocab)

        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        self.word2index = {word: idx for idx, word in enumerate(review_vocab)}
        self.label2index = {label: idx for idx, label in enumerate(label_vocab)}

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 注意: 不用能np.random.rand,否则是(0,1)的正态；
        # 第1/2层之间必须用zeros初始化,否则训练不动,原因不明
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        # 注意这里不能用rand,原因同上
        self.weights_1_2 = np.random.normal(0, self.output_nodes ** -0.5,
                                            (self.hidden_nodes, self.output_nodes))

        self.layer_1 = np.zeros((1, hidden_nodes))

    def get_target_for_label(self, label):
        return 1 if label == 'POSITIVE' else 0
        # 注意后面的run函数输出,以>=0.5为POSITIVE, self.label2index实际上没用到
        # return self.label2index[label]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):
        assert (len(training_reviews_raw) == len(training_labels))
        training_reviews = []
        for r in training_reviews_raw:
            training_reviews.append(np.unique([self.word2index[w] for w in r.split(' ') if w in self.word2index]))
        correct_so_far = 0
        start = time.time()

        for i in range(len(training_reviews)):
            review, label = training_reviews[i], training_labels[i]
            self.layer_1 = np.sum([self.weights_0_1[wordIdx] for wordIdx in review],
                                  axis=0, keepdims=True)
            layer_2 = self.sigmoid(np.dot(self.layer_1, self.weights_1_2))

            # 注意 layer2_error是loss关于layer2的偏导,而不是L本身
            # $L = \frac{1}{2} (y - layer2) ^ 2$
            # $ \frac{\partial{L}}{\partial{layer2}} = (y - layer2) * (-1) = layer2 - y $
            # 不要这样写: error = (self.layer_2_output - self.label2index[label]) ** 2
            layer2_error = layer_2 - self.get_target_for_label(label)
            layer2_delta = layer2_error * self.sigmoid_output_2_derivative(layer_2)

            layer1_error = np.dot(layer2_delta, self.weights_1_2.T)
            layer1_delta = layer1_error

            self.weights_1_2 -= self.learning_rate * np.dot(self.layer_1.T, layer2_delta)
            # self.weights_0_1 -= self.learning_rate * np.dot(self.layer_0.T, layer1_delta)
            # 注意 layer1_delta[0] 里面的下标0不能省,否则维度不对
            for wordIdx in review:
                self.weights_0_1[wordIdx] -= self.learning_rate * layer1_delta[0]

            if np.abs(layer2_error) < 0.5:
                correct_so_far += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i + 1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")
            if (i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        correct = 0
        start = time.time()

        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if (pred == testing_labels[i]):
                correct += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i + 1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        review = np.unique([self.word2index[w] for w in review.split(' ') if w in self.word2index])

        self.layer_1 = np.sum(self.weights_0_1[wordIdx] for wordIdx in review)
        self.layer_2 = self.sigmoid(np.dot(self.layer_1, self.weights_1_2))

        return 'POSITIVE' if self.layer_2 >= 0.5 else 'NEGATIVE'


if __name__ == '__main__':
    g = open('reviews.txt', 'r')  # What we know!
    reviews = list(map(lambda x: x[:-1], g.readlines()))
    g.close()

    g = open('labels.txt', 'r')  # What we WANT to know!
    labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
    g.close()

    # learning_rate 要慢慢试,刚开始用0.1/0.001 都不收敛
    mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.001)
    # mlp.test(reviews[-1000:], labels[-1000:])
    mlp.train(reviews[:-1000], labels[:-1000])
