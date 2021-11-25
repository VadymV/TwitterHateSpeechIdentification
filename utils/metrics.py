import pandas as pd
import torch


class PredictionsAccumulator:
    def __init__(self):
        self.tweet_id = []
        self.preds = []
        self.rounded_preds = []
        self.actuals = []
        self.correct_preds = []

    def update(self, tweet_ids, preds, actuals):
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct_preds = (rounded_preds == actuals).float()

        self.tweet_id.extend(list(tweet_ids.cpu().numpy()))
        self.preds.extend(list(preds.cpu().numpy()))
        self.rounded_preds.extend(list(rounded_preds.cpu().numpy()))
        self.actuals.extend(list(actuals.cpu().numpy()))
        self.correct_preds.extend(list(correct_preds.cpu().numpy()))

    def create_data_array(self):
        dict = {'tweet_id': self.tweet_id,
                'preds': self.preds,
                'rounded_preds': self.rounded_preds,
                'actuals': self.actuals,
                'correct_preds': self.correct_preds
                }
        df = pd.DataFrame(dict)
        return df


class TorchMetricsAccumulator:
    def __init__(self):
        self.epoch = []
        self.loss = []
        self.set_id = []
        self.seed = []
        self.accuracy = []
        self.macro_f1 = []
        self.macro_f1_beta_2 = []
        self.precision = []
        self.recall = []

    def update(self, train_metrics, epoch, set_id, seed, loss):
        self.epoch.append(epoch + 1)
        self.loss.append(loss)
        self.set_id.append(set_id)
        self.seed.append(seed)
        self.accuracy.append(train_metrics['acc'].item())
        self.macro_f1.append(train_metrics['macro-f1'].item())
        self.macro_f1_beta_2.append(train_metrics['macro-f1_beta_2'].item())
        self.precision.append(train_metrics['precision'].item())
        self.recall.append(train_metrics['recall'].item())

    def write_metrics(self, file_name):
        dict = {'epoch': self.epoch,
                'loss': self.loss,
                'set_id': self.set_id,
                'seed': self.seed,
                'accuracy': self.accuracy,
                'macro_f1': self.macro_f1,
                'macro_f1_beta_2': self.macro_f1_beta_2,
                'precision': self.precision,
                'recall': self.recall,
                }

        df = pd.DataFrame(dict)
        df.to_csv("./output/{}".format(file_name), index=False)


class OwnMetrics:
    def __init__(self):
        self.acc = 0
        self.f1 = 0
        self.precision = 0
        self.recall = 0

        self.target_true = 0
        self.predicted_true = 0
        self.correct_true = 0

    def update_metrics(self, preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))

        # Accuracy:
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        self.acc += acc

        # Precision, Recall and F1 score
        predicted_classes = rounded_preds == 1
        self.target_true += torch.sum(y == 1).float()
        self.predicted_true += torch.sum(predicted_classes).float()
        self.correct_true += torch.sum(
            (predicted_classes == y) * (predicted_classes == 1)).float()

    def calculate_metrics(self, k):
        self.acc /= k
        self.precision = self.correct_true / self.predicted_true
        self.recall = self.correct_true / self.target_true
        self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall)

    def print_metrics(self, stage):
        print(f'\t{stage}')
        print(f'\t\tAccuracy: {self.acc * 100:.2f}%')
        print(f'\t\tF1: {self.f1}')
        print(f'\t\tPrecision: {self.precision}')
        print(f'\t\tRecall: {self.recall}')
