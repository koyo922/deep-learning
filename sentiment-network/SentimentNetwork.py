# %pdb

# TODO: -Copy the SentimentNetwork class from Project 4 lesson
#       -Modify it according to the above instructions

import time
import sys

import numpy as np
import itertools
from collections import defaultdict, Counter


# Encapsulate our neural network in a class
# noinspection PyAttributeOutsideInit,PyMethodMayBeStatic,PyShadowingNames
# noinspection PyTypeChecker,PyRedundantParentheses,PyPep8
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1,
                 min_count=50, polarity_cutoff=0.4):
        """Create a SentimentNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        np.random.seed(1)
        self.pre_process_data(reviews, labels, min_count, polarity_cutoff)
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, min_count, polarity_cutoff):
        positive_counter = Counter()
        negative_counter = Counter()
        total_counter = Counter()
        for review, label in zip(reviews, labels):
            if label == 'POSITIVE':
                for w in review.split(' '): positive_counter[w] += 1
            else:
                for w in review.split(' '): negative_counter[w] += 1
            for w in review.split(' '): total_counter[w] += 1
        pos_neg_ratio = defaultdict(float)
        for w, c in total_counter.items():
            r = positive_counter[w] / (negative_counter[w] + 1)
            pos_neg_ratio[w] = np.log(r) if r > 1 else -np.log(1 / (r + 0.001))

        review_vocab = set(itertools.chain.from_iterable(r.split(' ') for r in reviews))
        self.review_vocab = [w for w in review_vocab \
                             if total_counter[w] >= min_count \
                             and abs(pos_neg_ratio[w]) >= polarity_cutoff]

        label_vocab = set(labels)
        self.label_vocab = list(label_vocab)

        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        # 注意: 现在的self.review_vocab 跟直接的review_vocab已经不同了，滤过了一些词
        self.word2index = {word: idx for idx, word in enumerate(self.review_vocab)}
        self.label2index = {label: idx for idx, label in enumerate(self.label_vocab)}

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
            wordIdxes = [self.word2index[w] for w in r.split(' ') if w in self.word2index]
            training_reviews.append(np.unique(wordIdxes))
        correct_so_far = 0
        start = time.time()

        for i in range(len(training_reviews)):
            review, label = training_reviews[i], training_labels[i]
            # 注意,这种写法兼容空的 review
            # np.unique()出来的结果,默认还是np.float64类型的dtype
            # 而在ndarray的下标中,必须用int类型的dtype；例如
            # self.weights_0_1[review.astype(int)]
            # 有值的还好,自动识别成int；而当array([])时,就用默认的float64了,所以必须astype(int)
            self.layer_1 = self.weights_0_1[review.astype(int)].sum(axis=0, keepdims=True)
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
            self.weights_0_1[review.astype(int)] -= self.learning_rate * layer1_delta[0]

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
        wordIdxes = [self.word2index[w] for w in review.split(' ') if w in self.word2index]
        review = np.unique(wordIdxes)

        # 注意,这种写法兼容空的 review
        self.layer_1 = self.weights_0_1[review.astype(int)].sum(axis=0, keepdims=True)
        self.layer_2 = self.sigmoid(np.dot(self.layer_1, self.weights_1_2))

        return 'POSITIVE' if self.layer_2 >= 0.5 else 'NEGATIVE'


if __name__ == '__main__':
    g = open('reviews.txt', 'r')  # What we know!
    reviews = list(map(lambda x: x[:-1], g.readlines()))
    g.close()

    g = open('labels.txt', 'r')  # What we WANT to know!
    labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
    g.close()

    # # learning_rate 要慢慢试,刚开始用0.1/0.001 都不收敛
    # mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.001)
    # # mlp.test(reviews[-1000:], labels[-1000:])
    # mlp.train(reviews[:-1000], labels[:-1000])

    # mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.05,learning_rate=0.01)
    # mlp.train(reviews[:-1000],labels[:-1000])

    mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], min_count=20, polarity_cutoff=0.8, learning_rate=0.01)
    mlp.train(reviews[:-1000], labels[:-1000])

    mlp.test(reviews[-1000:], labels[-1000:])
