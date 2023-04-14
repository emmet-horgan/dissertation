from .parse import Parse
from . import helper
import csv
import os
import random
from random import shuffle
import numpy as np
import torch
import shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def write_grades(name: str, binary=False, treat=1, notreat=-1):
    parser = Parse()
    paths = helper.Paths()
    subject = "subject"

    annotations = []
    for i, grade in enumerate(parser.grades):
        if binary:
            annotations.append([subject + str(i), notreat if grade < 2 else treat])
        else:
            annotations.append([subject + str(i), grade])

    with open(os.path.join(paths.csv_data_path, name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(annotations)


def __read_segments(path: str):
    return [np.genfromtxt(os.path.join(path, x), delimiter=",") for x in os.listdir(path)]


def read_grades(path: str):
    """
        -1: Negative for HIE (Do not treat)
        +1: Positive for HIE (Do Treat)
    """
    data = {}
    with open(path) as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            data[row[0]] = int(row[1]) if int(row[1]) == 1 else 0

    return data


def unpack(path: str, num=30, split=0.8, name="mini-dataset"):
    paths = helper.Paths()
    grades = read_grades(paths.annotations)
    data = {
        1: [],
        -1: []
    }

    subjects = random.sample(range(0, 120), num)
    for subject in subjects:
        subject = "subject" + str(subject)
        if not os.path.isdir(os.path.join(path, subject)):
            continue
        segs = __read_segments(os.path.join(path, subject))
        data[grades[subject]].extend(segs)

    num_treat = len(data[1])
    num_notreat = len(data[-1])
    total = num_treat + num_notreat
    distrib = num_treat / total

    validation_len = round((1 - split) * total)
    num_valid_treat = round(distrib * validation_len)
    print("{:^70}\n".format("=" * 5 + " Dataset Summary " + "=" * 5)
          + "{0:>35} = {1}".format("len(total)", int(total)) + "\n"
          + "{0:>35} = {1}".format("len(validation)", int(validation_len)) + "\n"
          + "{0:>35} = {1:.2f}".format("distribution(train, treat)", distrib * 100) + "\n"
          + "{0:>35} = {1:.2f}".format("distribution(validation, treat)", (num_valid_treat / validation_len) * 100))

    validation_treat = random.sample(range(0, num_treat), num_valid_treat)
    validation_notreat = random.sample(range(0, num_notreat), validation_len - num_valid_treat)

    os.chdir(paths.csv_tfds_path)
    if os.path.isdir(name):
        shutil.rmtree(name, ignore_errors=True)
    os.mkdir(name)
    os.chdir(name)
    os.mkdir("validation")

    for grade, data in data.items():
        if grade == 1:
            for i, x in enumerate(data):
                if i in validation_treat:
                    np.savetxt("validation/grade-1-segment-" + str(i) + ".csv", x, delimiter=",")
                else:
                    np.savetxt("grade-1-segment-" + str(i) + ".csv", x, delimiter=",")
        else:
            for i, x in enumerate(data):
                if i in validation_notreat:
                    np.savetxt("validation/grade-0-segment-" + str(i) + ".csv", x, delimiter=",")
                else:
                    np.savetxt("grade-0-segment-" + str(i) + ".csv", x, delimiter=",")


def create_folds(path: str, num_folds=10, equalize=False):
    grades = read_grades(helper.Paths().annotations)
    data = os.listdir(path)
    data_len = len(data)
    if equalize:
        ones = [subject for subject in data if grades[subject] == 1]
        zeros = [subject for subject in data if grades[subject] == 0]
        shuffle(ones)
        shuffle(zeros)

        len_per_fold = (len(ones) + len(zeros)) // num_folds
        remainder_len = len(ones) + len(zeros) - (num_folds * len_per_fold)

        folds = [[] for _ in range(num_folds)]
        for i, one in enumerate(ones):
            i %= num_folds
            folds[i].append(one)
        for i, zero in enumerate(zeros):
            i %= num_folds
            folds[i].append(zero)
        for fold in folds:
            shuffle(fold)

    else:
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

    return folds


def unpack_folds(path, folds, rgb=True, print_=None, transform=None):
    grades = read_grades(helper.Paths().annotations)
    img_folds = []
    proportions = []
    for i, fold in enumerate(folds):
        images = []
        balance = 0
        for j, subject in enumerate(fold):
            grade = grades[subject]
            balance += grade
            # Create one hot-encoded tensor labels
            if grade == 1:
                grade = torch.tensor([1.0, 0.0])
            else:
                grade = torch.tensor([0.0, 1.0])
            with os.scandir(os.path.join(path, subject)) as imgs:
                for img in imgs:
                    with open(img) as f:
                        reader = csv.reader(f, delimiter=",")
                        image = [list(map(float, row)) for row in reader]
                        if rgb:
                            image = [image.copy(), image.copy(), image.copy()]  # Fudge RGB image
                        image = torch.tensor(image)
                        if transform is not None:
                            image = transform(image)
                        image = (image, grade)
                        images.append(image)
        img_folds.append(images)
        proportions.append(balance/len(fold))
        if print_:
            print_.logfile.write(f"Proportion of 1's (Fold {i:2}): {(balance / len(fold)) * 100: 4.2f}\n")
            print(f"Proportion of 1's (Fold {i:2}): {(balance / len(fold)) * 100: 4.2f}")
    return img_folds, proportions


def defold(folds, proportions=None, print_=None):
    index = 0
    if proportions is not None:
        for i, balance in enumerate(proportions):
            if 5/12 <= balance <= 7/12:
                index = i
                break
        else:
            raise SystemError
    if print_ is not None:
        print(f"Fold {index} chosen\n")
        print_.logfile.write(f"Fold {index} chosen\n")
    test = folds.pop(index)
    train = [datapoint for fold in folds for datapoint in fold]
    return train, test


def si_split(path, num_folds, files, transform=None, kfolds=False, equalize=False):
    folds = create_folds(path, num_folds, equalize=equalize)
    folds, proportions = unpack_folds(path, folds, print_=files, transform=transform)
    if not kfolds:
        return defold(folds, proportions, print_=files)
    else:
        return folds


class Model:

    def __init__(self, model, optimizer, loss_fn, batch_size, sep="_", tag=""):
        self.__model = model
        self.__optimizer = optimizer
        self.__loss_fn = loss_fn
        self.__batch_size = batch_size
        self.__sep = sep
        self.__tag = tag
        self.__time = str(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))

    @property
    def tag(self):
        return self.__tag

    @property
    def model_name(self):
        return type(self.__model).__name__

    @property
    def optimizer_name(self):
        return type(self.__optimizer).__name__

    @property
    def loss_fn_name(self):
        return type(self.__loss_fn).__name__

    @property
    def batch_size_str(self):
        return str(self.__batch_size)

    @property
    def batchnorm(self):
        for m in self.__model.children():
            if "batchnorm" in str(m).lower():
                return True
        return False

    @property
    def filename(self):
        return self.__sep.join([self.tag,
                                self.model_name,
                                self.optimizer_name,
                                self.loss_fn_name,
                                "B" + self.batch_size_str]) \
            + "@" + self.__time + ".pt"

    @property
    def dirname(self):
        return self.__sep.join([self.tag,
                                self.model_name,
                                self.optimizer_name,
                                self.loss_fn_name,
                                "B" + self.batch_size_str]) \
            + "@" + self.__time

    @property
    def logfilename(self):
        return self.__sep.join([self.tag,
                                self.model_name,
                                self.optimizer_name,
                                self.loss_fn_name,
                                "B" + self.batch_size_str]) \
            + "@" + self.__time + ".log"

    @property
    def filepath(self):
        return os.path.join("models", self.filename)

    @property
    def time(self):
        return self.__time


class Logfiles(Model):

    def __init__(self, model, optimizer, loss_fn, batch_size, tag=None, kfolds=False):
        super().__init__(model, optimizer, loss_fn, batch_size, sep="_", tag=tag)

        self.__kfolds = kfolds
        if self.__kfolds:
            self.fold = 0
            self.__logpath = os.path.join("logs", self.dirname, "fold" + str(self.fold))
        else:
            self.__logpath = os.path.join("logs", self.dirname)

        self.__datapath = os.path.join(self.__logpath, "data")

        os.makedirs(self.__logpath, exist_ok=True)
        os.makedirs(self.__datapath, exist_ok=True)

        self.__logfile = None

        self.__training_loss = None
        self.__training_acc = None

        self.__test_loss = None
        self.__test_acc = None
        self.__test_auc = None

        self.__sumwriter = None

        self.__training_loss_csv = None
        self.__training_acc_csv = None

        self.__test_loss_csv = None
        self.__test_acc_csv = None
        self.__test_auc_csv = None

        self.__roc = None
        self.__roc_csv = None

    def open(self):
        if self.__kfolds:
            self.__sumwriter = SummaryWriter(os.path.join("runs", self.dirname, "fold" + str(self.fold)))
        else:
            self.__sumwriter = SummaryWriter(os.path.join("runs", self.dirname))

        self.__logfile = open(os.path.join(self.__logpath, self.logfilename), "w")

        self.__training_loss = open(os.path.join(self.__datapath, "train_loss_batch.csv"), "w")
        self.__training_acc = open(os.path.join(self.__datapath, "train_accuracy_epoch.csv"), "w")

        self.__test_loss = open(os.path.join(self.__datapath, "test_loss_epoch.csv"), "w")
        self.__test_acc = open(os.path.join(self.__datapath, "test_accuracy_epoch.csv"), "w")
        self.__test_auc = open(os.path.join(self.__datapath, "test_auc_batch.csv"), "w")
        if self.__kfolds:
            self.__roc = open(os.path.join(self.__datapath, "..", "..", "roc.csv"), "w")
        else:
            self.__roc = open(os.path.join(self.__datapath, "..", "roc.csv"), "w")

        self.__training_loss_csv = csv.writer(self.__training_loss, delimiter=",", lineterminator="\n")
        self.__training_acc_csv = csv.writer(self.__training_acc, delimiter=",", lineterminator="\n")
        self.__roc_csv = csv.writer(self.__roc, delimiter=",", lineterminator="\n")

        self.__test_loss_csv = csv.writer(self.__test_loss, delimiter=",", lineterminator="\n")
        self.__test_acc_csv = csv.writer(self.__test_acc, delimiter=",", lineterminator="\n")
        self.__test_auc_csv = csv.writer(self.__test_auc, delimiter=",", lineterminator="\n")

    def refresh(self):
        self.fold += 1
        if self.__kfolds:
            self.__logpath = os.path.join("logs", self.dirname, "fold" + str(self.fold))
        else:
            self.__logpath = os.path.join("logs", self.dirname)

        self.__datapath = os.path.join(self.__logpath, "data")

        os.makedirs(self.__logpath, exist_ok=True)
        os.makedirs(self.__datapath, exist_ok=True)

        self.close()
        self.open()

    def close(self):
        self.__logfile.close()

        self.__training_loss.close()
        self.__training_acc.close()

        self.__test_loss.close()
        self.__test_acc.close()
        self.__test_auc.close()
        self.__roc.close()

        del self.__sumwriter

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def training_loss(self):
        return self.__training_loss_csv

    @property
    def training_acc(self):
        return self.__training_acc_csv

    @property
    def test_loss(self):
        return self.__test_loss_csv

    @property
    def test_auc(self):
        return self.__test_auc_csv

    @property
    def roc(self):
        return self.__roc_csv

    @property
    def test_acc(self):
        return self.__test_acc_csv

    @property
    def sumwriter(self):
        return self.__sumwriter

    @property
    def logfile(self):
        return self.__logfile

    @property
    def logpath(self):
        return self.__logpath

    @property
    def datapath(self):
        return self.__datapath

