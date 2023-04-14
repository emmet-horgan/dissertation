# Standard library
import re
import os
import csv
import shutil
import datetime as dt
from random import shuffle
from sklearn.metrics import roc_curve
import numpy as np

# Custom library
from .helper import Paths
from .filemanage import read_grades
from .transforms import softmax

# Pytorch libraries
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import models
from torch import nn
from torchmetrics.classification import BinaryAUROC, BinaryROC


def findmodel(model, batch_norm=False, device=None):
    if model.upper() == "VGG16":
        if batch_norm:
            model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        else:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        features = list(model.classifier.children())
        features.extend([nn.Linear(1000, 128), nn.Linear(128, 2)])
        model.classifier = nn.Sequential(*features)

    elif model.upper() == "RESNET50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(model.fc, nn.Linear(1000, 128), nn.Linear(128, 2))

    elif model.upper() == "RESNET18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(model.fc, nn.Linear(1000, 128), nn.Linear(128, 2))

    elif model.upper() == "RESNET34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Sequential(model.fc, nn.Linear(1000, 128), nn.Linear(128, 2))

    else:
        raise ValueError

    if device is not None:
        # Move model to device (hopefully GPU)
        model = model.to(device)

    return model


def findoptimizer(optimizer, model, lr):
    if optimizer.upper() == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    elif optimizer.upper() == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    else:
        raise ValueError

    return optimizer


def kfolds(path: str, num_folds=10):
    diskloc = Paths().csv_tfds_path
    diskloc = os.path.join(diskloc, str(num_folds) + "FoldCV")
    if os.path.isdir(diskloc):
        shutil.rmtree(diskloc, ignore_errors=True)
    os.mkdir(diskloc)

    grades = read_grades(Paths().annotations)

    data = os.listdir(path)
    data_len = len(data)

    shuffle(data)
    len_per_fold = data_len // num_folds

    remainder_len = data_len - (num_folds * len_per_fold)
    remainder = data[-remainder_len:]

    remainder_len = [(x, len(os.listdir(os.path.join(path, x)))) for x in remainder]
    remainder_len.sort(key=lambda item: item[1], reverse=True)

    folds = []
    for x in range(num_folds):
        start = x * len_per_fold
        end = (x + 1) * len_per_fold

        folds.append(data[start: end])

    fold_len = []
    for i, fold in enumerate(folds):
        img_len = 0
        for x in fold:
            img_len += len(os.listdir(os.path.join(path, x)))

        fold_len.append((i, img_len))
    fold_len.sort(key=lambda item: item[1])

    for i, (subject, img_len) in enumerate(remainder_len):
        folds[fold_len[i][0]].append(subject)

    for i, fold in enumerate(folds):
        fold_path = os.path.join(diskloc, "fold" + str(i))
        os.mkdir(fold_path)

        for j, subject in enumerate(fold):
            subject_path = os.path.join(path, subject)

            with os.scandir(subject_path) as itr:
                for img in itr:
                    grade = "1" if grades[subject] == 1 else "0"
                    file_name = subject + "-grade-" + grade + "-" + img.name
                    dst = os.path.join(fold_path, file_name)
                    shutil.copyfile(img.path, dst)


class SIDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None, cv=False):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CrossValidation:

    def __init__(self, folds):
        self.folds = [SIDataset(fold) for fold in folds]

    def __len__(self):
        return len(self.folds)

    def __getitem__(self, idx):
        folds = self.folds.copy()
        testing = folds.pop(idx)
        training = ConcatDataset(folds)

        return training, testing


class TFDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.imgs = [x.path for x in os.scandir(img_dir) if x.is_file()]
        self.transform = transform
        self.target_transform = target_transform
        self.label_sm = re.compile(r"grade-(?P<grade>\d)-segment-\d+\.csv")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        label_str = self.label_sm.search(self.imgs[idx]).group("grade")
        label = int(label_str) if label_str == "1" else 0
        # [treat, notreat]: One hot encoding
        if label:
            label = torch.tensor([1.0, 0.0])  # Treat
        else:
            label = torch.tensor([0.0, 1.0])  # No treat

        with open(self.imgs[idx]) as f:
            reader = csv.reader(f, delimiter=",")
            image = [list(map(float, row)) for row in reader]
            image = [image, image, image]  # Bluff grayscale to RGB
        image = torch.tensor(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train_loop(device, dataloader, model, loss_fn, optimizer, epoch=None, files=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct, train_loss = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        # Backpropagation
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        train_loss += loss

        if files is not None:
            files.logfile.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")

        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if epoch is not None and files is not None:
            # Total number of batches
            files.sumwriter.add_scalar('Training Loss/batch', loss, epoch * len(dataloader) + batch)

            files.training_loss.writerow([epoch * len(dataloader) + batch, loss])

    train_loss /= num_batches
    correct /= size

    if epoch is not None and files is not None:
        files.sumwriter.add_scalar("Train Accuracy", correct, epoch)
        files.sumwriter.add_scalar('Train Loss/epoch', train_loss, epoch)
        files.training_acc.writerow([epoch, correct])


def test_loop(device, dataloader, model, loss_fn, epoch=None, files=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    AUC = BinaryAUROC(thresholds=None)
    preds = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            tmp = softmax(pred)
            labels = torch.cat((labels, y[:, 0]))
            preds = torch.cat((preds, tmp[:, 0]))
            #auc = AUC(tmp[:, 0], y[:, 0]).item()  # Calculating every loop iteration, maybe save data and move outside
            del tmp
            if epoch is not None and files is not None:
                # Total number of batches
                #files.test_auc.writerow([epoch * len(dataloader) + batch, auc])
                #files.sumwriter.add_scalar('Test AUC/batch', auc, epoch * len(dataloader) + batch)
                files.sumwriter.add_scalar('Test Loss/batch', loss, epoch * len(dataloader) + batch)
        auc = AUC(preds, labels).item()
        files.test_auc.writerow([epoch, auc])
        files.sumwriter.add_scalar('Test AUC/batch', auc, epoch)
        test_loss /= num_batches
        correct /= size

        if epoch is not None and files is not None:
            files.sumwriter.add_scalar("Test Accuracy", correct, epoch)
            files.sumwriter.add_scalar('Test Loss/epoch', test_loss, epoch)
            files.test_loss.writerow([epoch, test_loss])
            files.test_acc.writerow([epoch, correct])

    if files is not None:
        files.logfile.write(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(epochs, device, training_dl, validation_dl, model, loss_fn, optimizer, files=None, debug=False):
    start = dt.datetime.now()
    start_str = start.strftime("%H:%M:%S")

    if files is not None:
        files.logfile.write("{:<30}{:>20}\n".format("MODEL: ", files.model_name))
        files.logfile.write("{:<30}{:>20}\n".format("LOSS FUNCTION: ", files.loss_fn_name))
        files.logfile.write("{:<30}{:>20}\n".format("OPTIMIZER: ", files.optimizer_name))
        files.logfile.write("{:<30}{:>20}\n".format("BATCH SIZE: ", files.batch_size_str))
        files.logfile.write("{:<30}{:>20}\n".format("BATCH NORMALIZATION: ", files.batchnorm))

        files.logfile.write("{:<30}{:>20}\n".format("START: ", start_str))

    test_loop(device, validation_dl, model, loss_fn, epoch=-1, files=files)
    for ep in range(epochs):
        print(f"Epoch {ep + 1}\n-------------------------------")

        if files is not None:
            files.logfile.write(f"Epoch {ep + 1}\n-------------------------------\n")
        if debug:
            from torch.profiler import profile, record_function, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    train_loop(device, training_dl, model, loss_fn, optimizer, ep, files)
                    test_loop(device, validation_dl, model, loss_fn, ep, files)
            print(prof.key_averages().table())
        else:
            train_loop(device, training_dl, model, loss_fn, optimizer, ep, files)
            test_loop(device, validation_dl, model, loss_fn, ep, files)

    if files is not None:
        end = dt.datetime.now()
        elapsed = end - start
        end_str = end.strftime("%H:%M:%S")
        files.logfile.write("{:<30}{:>20}\n".format("END: ", end_str))
        files.logfile.write("{:<30}{!s:>20}\n".format("Total elapsed time: ", elapsed))

    print("Finished !")


def model_eval(device, model, validation_dl):
    y_true = []
    y_pred = []
    roc = BinaryROC(thresholds=2000)
    with torch.no_grad():
        for batch, (X, y) in enumerate(validation_dl):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            pred = softmax(pred)

            y_true.extend(y[:, 0].tolist())
            y_pred.extend(pred[:, 0].tolist())

    return roc(torch.tensor(y_pred), torch.tensor(y_true, dtype=torch.int32))
    #return roc_curve(np.array(y_true), np.array(y_pred))

