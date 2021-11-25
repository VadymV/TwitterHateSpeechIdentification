import time
import scipy.stats

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1, FBeta, ConfusionMatrix

from utils.metrics import PredictionsAccumulator
from visual.plot import plot_cm


def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_min_s = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_min_s * 60))
    return elapsed_min_s, elapsed_secs


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    metrics_torch = MetricCollection({'acc': Accuracy(compute_on_step=False, num_classes=1),
                                      'precision': Precision(compute_on_step=False, average='macro', num_classes=1),
                                      'recall': Recall(compute_on_step=False, average='macro', num_classes=1),
                                      'macro-f1': F1(compute_on_step=False, average='macro', num_classes=1),
                                      'macro-f1_beta_2': FBeta(compute_on_step=False, average='macro', num_classes=1,
                                                               beta=2),
                                      'cm': ConfusionMatrix(num_classes=2)})
    metrics_torch.to(device)

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.t).squeeze(1)

        loss = criterion(predictions, batch.l)

        metrics_torch(predictions, batch.l.int())

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), metrics_torch


def evaluate(model, iterator, criterion, device, plot=False, plot_file_name="cm", plot_title=None):
    epoch_loss = 0
    metrics_torch = MetricCollection({'acc': Accuracy(compute_on_step=False, num_classes=1),
                                      'precision': Precision(compute_on_step=False, average='macro', num_classes=1),
                                      'recall': Recall(compute_on_step=False, average='macro', num_classes=1),
                                      'macro-f1': F1(compute_on_step=False, average='macro', num_classes=1),
                                      'macro-f1_beta_2': FBeta(compute_on_step=False, average='macro', num_classes=1,
                                                               beta=2),
                                      'cm': ConfusionMatrix(num_classes=2)})
    metrics_torch.to(device)
    predictions_accumulator = PredictionsAccumulator()

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.t).squeeze(1)

            loss = criterion(predictions, batch.l)

            metrics_torch(predictions, batch.l.int())
            predictions_accumulator.update(batch.id, predictions, batch.l)

            epoch_loss += loss.item()

    if plot:
        plot_cm(metrics_torch.cm.confmat.to("cpu").int().numpy(), plot_file_name=plot_file_name, title=plot_title)

    return epoch_loss / len(iterator), metrics_torch, predictions_accumulator.create_data_array()


def train_model(num_epochs, model_initializer, device, checkpoint_name, seed, metrics_accumulator):
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):

        start_time = time.time()

        train_loss, train_metrics = train(model_initializer.model, model_initializer.dataset.train_iterator,
                                          model_initializer.optimizer, model_initializer.criterion, device)
        valid_loss, val_metrics, _ = evaluate(model_initializer.model, model_initializer.dataset.valid_iterator,
                                              model_initializer.criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_initializer.model.state_dict(),
                       './checkpoints/{}_seed-{}.pt'.format(checkpoint_name, seed))

        metrics_accumulator.update(train_metrics.compute(), epoch, "train", seed, train_loss)
        metrics_accumulator.update(val_metrics.compute(), epoch, "validation", seed, valid_loss)

        print("#" * 50)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print("Train metrics: {}".format(train_metrics.compute()))
        print(f'Train Loss: {train_loss:.3f}')
        print("Val metrics: {}".format(val_metrics.compute()))
        print(f'Val. Loss: {valid_loss:.3f}')


def split_data():
    orig_train_set = pd.read_csv("./raw_data/train_E6oV3lV.csv", delimiter=",")
    train_data, test_data = train_test_split(orig_train_set[["id", "label", "tweet"]],
                                             stratify=orig_train_set[["label"]],
                                             test_size=0.3,
                                             random_state=42)
    train_data.to_csv("./raw_data/train_data.csv", index=False)
    test_data.to_csv("./raw_data/test_data.csv", index=False)


def mean_confidence_interval(data, confidence=0.95):
    # taken from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, se
