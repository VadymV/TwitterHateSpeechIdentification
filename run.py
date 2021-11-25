import random

import numpy as np
import pandas as pd
import torch

import utils.misc
from models.roberta_initializer import RoBERTaInitializer
from models.cnn_initializer import CNNInitializer
from utils import misc
from utils.mcnemars_test import McNemarsTest
from utils.metrics import TorchMetricsAccumulator
from visual.plot import create_train_val_plot, plot_pdf_x, plot_pdf_y

misc.split_data()  # Creates train and test sets

EPOCHS = 10
BATCH_SIZE = 32
SEEDS = [10, 11, 12]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Train models ----
print("Training stage")
cnn_metrics_accumulator = TorchMetricsAccumulator()
roberta_metrics_accumulator = TorchMetricsAccumulator()
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    cnn_model_initializer = CNNInitializer(BATCH_SIZE, device)
    misc.train_model(EPOCHS, cnn_model_initializer, device, "cnnModel", seed, cnn_metrics_accumulator)
    cnn_metrics_accumulator.write_metrics("cnn_model_metrics.csv")

    roberta_model_initializer = RoBERTaInitializer(BATCH_SIZE, device)
    misc.train_model(EPOCHS, roberta_model_initializer, device, "robertaModel", seed, roberta_metrics_accumulator)
    roberta_metrics_accumulator.write_metrics("roberta_model_metrics.csv")

# ---- Evaluate models ----
trials = [1, 2, 3]
for i in range(0, len(trials)):
    print("Evaluation stage")

    random.seed(trials[0])
    np.random.seed(trials[0])
    torch.manual_seed(trials[0])
    torch.backends.cudnn.deterministic = True

    # Evaluate the CNN model
    print("#" * 50)
    print("CNN Evaluation")
    cnn_init = CNNInitializer(BATCH_SIZE, device, print_info=False)
    cnn_init.model.load_state_dict(torch.load('./checkpoints/cnnModel_seed-{}.pt'.format(SEEDS[i])))
    cnn_valid_loss, cnn_valid_metrics, cnn_corr_preds = utils.misc.evaluate(cnn_init.model,
                                                                            cnn_init.dataset.test_iterator,
                                                                            cnn_init.criterion,
                                                                            device,
                                                                            plot=True,
                                                                            plot_file_name="cm_cnn_trial_{}".format(trials[0]),
                                                                            plot_title="CNN-based Model")
    print(f'Eval Loss: {cnn_valid_loss:.3f}')
    print("Eval metrics: {}".format(cnn_valid_metrics.compute()))

    # Evaluate the RoBERTa model
    print("#" * 50)
    print("RoBERTa Evaluation")
    roberta_init = RoBERTaInitializer(BATCH_SIZE, device, print_info=False)
    roberta_init.model.load_state_dict(torch.load('./checkpoints/robertaModel_seed-{}.pt'.format(SEEDS[i])))
    roberta_valid_loss, roberta_valid_metrics, roberta_corr_preds = utils.misc.evaluate(roberta_init.model,
                                                                                        roberta_init.dataset.test_iterator,
                                                                                        roberta_init.criterion,
                                                                                        device,
                                                                                        plot=True,
                                                                                        plot_file_name="cm_roberta_trial_{}".format(trials[0]),
                                                                                        plot_title="RoBERTa-based Model")
    print(f'Eval Loss: {roberta_valid_loss:.3f}')
    print("Eval metrics: {}".format(roberta_valid_metrics.compute()))


# ---- Compare models (models of the last trial are used) ----
# McNemar's Test
assert cnn_corr_preds.shape == roberta_corr_preds.shape
cnn_corr_preds = cnn_corr_preds.rename(columns={"correct_preds": "correct_preds_cls1"})
roberta_corr_preds = roberta_corr_preds.rename(columns={"correct_preds": "correct_preds_cls2"})

data = pd.merge(cnn_corr_preds, roberta_corr_preds, on='tweet_id', how='outer')
data.to_csv("./output/predictions.csv", index=False)

mcnemars_test = McNemarsTest(data[["tweet_id", "correct_preds_cls1", "correct_preds_cls2"]])
mcnemars_test.calculate_value()

# ---- Train-Val plot ----
create_train_val_plot(pd.read_csv("./output/cnn_model_metrics.csv"), "train_val_cnn", "CNN model")
create_train_val_plot(pd.read_csv("./output/roberta_model_metrics.csv"), "train_val_roberta", "RoBERTa-based model")

# ---- PDF plot ----
plot_pdf_x(data, "pdf_cnn")
plot_pdf_y(data, "pdf_roberta")


# Generate values for Table 1:
# TODO: make an automatic process, because currently it is done manually :) :
#  values were copied from the output and provided to the function mean_confidence_interval in misc.
