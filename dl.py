# Standard libraries
import atexit
import argparse
import os
import numpy as np
from sklearn.metrics import auc

# Custom Libraries
from anser.dl import SIDataset, train, CrossValidation, findmodel, findoptimizer, model_eval
from anser.transforms import transform_map
from anser.filemanage import si_split, Logfiles, Model
from anser.helper import Paths
from anser.plotting import plot, plot_roc

# Pytorch libraries
import torch
from torchvision import models
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import Compose


def exit_handler(x):
    print("PROGRAM TERMINATED: Writing data to disk !")
    del x


def argparse_setup():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning-rate", "-lr",
        help="Specify learning rate of model",
        action="store",
        type=float,
        default=1e-6
    )
    parser.add_argument(
        "--batch-size", "-bs",
        help="Specify batch size for training",
        action="store",
        type=int,
        default=16
    )
    parser.add_argument(
        "--epochs", "-e",
        help="Number of epochs to run for",
        action="store",
        type=int,
        default=500
    )
    parser.add_argument(
        "--model", "-m",
        help="Choose model",
        action="store",
        default="VGG16"
    )
    parser.add_argument(
        "--batch-norm", "-bn",
        help="Use batch normalization or not",
        action="store_true"
    )
    parser.add_argument(
        "--optimizer", "-opt",
        help="Choose optimizer",
        action="store",
        default="adam"
    )
    parser.add_argument(
        "--transform", "-t",
        help="Set a transform",
        action="append",
        default=[]
    )
    parser.add_argument(
        "--tag",
        help="Create a custom tag for identifying this particular training instance",
        action="store",
        default=""
    )
    parser.add_argument(
        "--save-plots",
        help="Plot the collated data and save it as a PNG file in the corresponding logging directory",
        action="store_false",
    )
    parser.add_argument(
        "--kfolds",
        help="Perform a 10 fold cross-validation",
        action="store_true",
    )
    parser.add_argument(
        "--save-model",
        help="Save the model",
        action="store_true",
    )
    parser.add_argument(
        "--equalize",
        help="Save the model",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="Debug mode",
        action="store_true",
    )
    parser.add_argument(
        "--eval",
        help="Evaluate model with roc curve",
        action="store_true",
    )
    return parser


def run(learning_rate=1e-6, batch_size=16, epochs=500, model="VGG16", optimizer="ADAM", batch_norm=False, transform=None,
        tag="", save_plots=True, kfolds=False, save_model=False, equalize=False, debug=False, eval=False):
    if transform is None:
        transform = []
    
    model_str = model
    optimizer_str = optimizer
    eval_data = []
    # Figure out what hardware is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {str(device).upper()} device")

    # Instantiate model and add fully-connected layer to filter from 1000 classes to 2
    model = findmodel(model, batch_norm=batch_norm, device=device)

    # Instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()

    optimizer = findoptimizer(optimizer, model, learning_rate)

    # Instantiate file handling class
    files = Logfiles(model, optimizer, loss_fn, batch_size, tag=tag, kfolds=kfolds)

    with files as files:
        # Create dataset iterable
        path = Paths().csv_tfd_path  # Path to data

        if len(transform) > 0:
            transforms = Compose([transform_map[x.upper()] for x in transform])
            data = si_split(path, num_folds=10, files=files, transform=transforms, kfolds=kfolds, equalize=equalize)
        else:
            data = si_split(path, num_folds=10, files=files, transform=None, kfolds=kfolds, equalize=equalize)

        if kfolds:
            itr = CrossValidation(data)
        else:
            training, testing = data
            itr = [(SIDataset(training), SIDataset(testing))]

        plot_paths = []
        for training, testing in itr:

            # Create dataloaders
            training_dl = DataLoader(training, batch_size=batch_size, shuffle=True)
            testing_dl = DataLoader(testing, batch_size=batch_size, shuffle=False)  # Don't shuffle test data

            print("{:<30}{:>20}\n".format("TAG: ", tag))
            print("{:<30}{:>20}\n".format("MODEL: ", files.model_name))
            print("{:<30}{:>20}\n".format("LOSS FUNCTION: ", files.loss_fn_name))
            print("{:<30}{:>20}\n".format("OPTIMIZER: ", files.optimizer_name))
            print("{:<30}{:>20}\n".format("BATCH SIZE: ", files.batch_size_str))
            print("{:<30}{:>20}\n".format("BATCH NORMALIZATION: ", files.batchnorm))
            print("{:<30}{:>20}\n".format("LR: ", learning_rate))
            print("{:<30}{:>20}\n".format("EPOCHS: ", epochs))
            if len(transform) > 0:
                print("{:<30}{:>20}\n".format("TRANSFORM: ", ", ".join(transform).upper()))

            if files:
                files.logfile.write("{:<30}{:>20}\n".format("TAG: ", tag))
                files.logfile.write("{:<30}{:>20}\n".format("TRANSFORM: ", ", ".join(transform).upper()))
                files.logfile.write("{:<30}{:>20}\n".format("LR: ", learning_rate))
                files.logfile.write("{:<30}{:>20}\n".format("EPOCHS: ", epochs))

            train(epochs, device, training_dl, testing_dl, model, loss_fn, optimizer, files, debug=debug)

            if eval:
                eval_data.append(model_eval(device, model, testing_dl))

            if save_model:
                torch.save(model.state_dict(), os.path.join(files.logpath, "model"))
            plot_paths.append(files.logpath)

            if kfolds:
                # Reset for next fold
                model = findmodel(model_str, batch_norm, device=device)
                optimizer = findoptimizer(optimizer_str, model, learning_rate)

                files.refresh()

        if eval:
            div = len(eval_data)
            fpr = torch.zeros(len(eval_data[0][0]))
            tpr = torch.zeros(len(eval_data[0][0]))
            threshold = torch.zeros(len(eval_data[0][0]))
            aucscores = []
            for f, t, thresh in eval_data:
                aucscores.append(auc(f, t))
                fpr += f
                tpr += t
                threshold += thresh
            tpr = tpr / div
            fpr = fpr / div
            threshold = threshold / div
            files.roc.writerow([fpr.tolist(), tpr.tolist(), threshold.tolist()])
            plot_roc(fpr, tpr, torch.tensor(aucscores), eval_data, os.path.join("logs", files.dirname), show=False)

    for path in plot_paths:
        plot(path, save=save_plots)


if __name__ == "__main__":
    parser = argparse_setup()
    args = vars(parser.parse_args())
    run(**args)
