import math
from tqdm import tqdm
from time import time as t


class Evaluation:
    def __init__(self):
        pass

    def intersection(self, l1, l2):
        return list(set(l1) & set(l2))

    def recall_at_k(self, true_list, predicted_list, k=40):
        """

        Args:
            true_list: list of actual relevant documents.
            predicted_list: sorted list of documents classified as relevant.
            k: int

        Returns:
            float, the recall at K
        """
        pred = predicted_list[:k]
        inter = self.intersection(pred, true_list)
        return round(len(inter) / len(true_list), 3)

    def precision_at_k(self, true_list, predicted_list, k=40):
        """

        Args:
            true_list: list of actual relevant documents.
            predicted_list: sorted list of documents classified as relevant.
            k: int

        Returns:
            float, the precision at K
        """

        pred = predicted_list[:k]
        inter = self.intersection(pred, true_list)
        return round(len(inter) / k, 3)

    def r_precision(self, true_list, predicted_list):
        pred = predicted_list[:len(true_list)]
        inter = self.intersection(pred, true_list)
        return round(len(inter) / len(true_list), 3)


    def f_score(self, true_list, predicted_list):
        """

        Args:
            true_list: list of actual relevant documents.
            predicted_list: sorted list of documents classified as relevant.
        Returns:
            float, the f score
        """

        if self.recall_at_k(true_list, predicted_list, k=100) + self.precision_at_k(true_list, predicted_list, k=100) == 0:
            return 0

        recall = self.recall_at_k(true_list, predicted_list, k=100)
        precision = self.precision_at_k(true_list, predicted_list, k =100)
        numerator = 2 * recall * precision
        denominator = recall + precision
        return numerator / denominator

    def average_precision(self, true_list, predicted_list, k=40):
        """
        This function calculate the average_precision@k metric.(i.e., precision in every recall point).
        Args:
            true_list: list of actual relevant documents.
            predicted_list: sorted list of documents classified as relevant.
            k: int

        Returns:
            float, average precision@k with 3 digits after the decimal point.
        """
        sum = 0.0
        div = 0
        for x in range(1, k + 1):
            if len(predicted_list) >= x and predicted_list[x - 1] in true_list:
                precision_k = self.precision_at_k(true_list=true_list, predicted_list=predicted_list)
                sum += precision_k
                div += 1
        if div == 0:
            return 0
        return round(sum / div, 3)


    def evaluate(self, ground_trues, predictions, k, print_scores=True):

        recall_lst = []
        precision_lst = []
        f_score_lst = []
        r_precision_lst = []
        avg_precision_lst = []
        metrices = {'recall@k': recall_lst,
                    'precision@k': precision_lst,
                    'f_score@k': f_score_lst,
                    'r-precision': r_precision_lst,
                    'MAP@k': avg_precision_lst,

                    }

        for i in range(len(ground_trues)):
            ground_true = ground_trues[i]
            predicted = predictions[i]
            recall_lst.append(self.recall_at_k(ground_true, predicted, k=k))
            precision_lst.append(self.precision_at_k(ground_true, predicted, k=k))
            f_score_lst.append(self.f_score(ground_true, predicted, k=k))
            r_precision_lst.append(self.r_precision(ground_true, predicted))
            avg_precision_lst.append(self.average_precision(ground_true, predicted, k=k))

        if print_scores:
            for name, values in metrices.items():
                print(name, sum(values) / len(values))

        return metrices