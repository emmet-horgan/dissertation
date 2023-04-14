from torchvision import transforms
import torch


def norm(tensor: torch.tensor):
    mean, std = tensor.mean([1, 2]), tensor.std([1, 2])
    return transforms.Normalize(mean, std)(tensor)


def abs_log(tensor: torch.tensor, k=10):
    return k * torch.log(torch.abs(tensor))


def square_log(tensor: torch.tensor, k=10):
    return k * torch.log(torch.square(tensor))


def rangemap(tensor: torch.tensor, numrange=[0, 255]):
    maxval = torch.max(tensor)
    minval = torch.min(tensor)
    tensorspan = maxval - minval
    idealspan = numrange[-1] - numrange[0]
    scaled = (tensor - minval) / tensorspan
    return numrange[0] + (scaled * idealspan)


def zeromean(tensor: torch.tensor):
    return tensor - 0.0124


def rgbrange(tensor: torch.tensor):
    rgb255 = 2.4287
    rgb0 = -0.1902
    scaled = (tensor - rgb0) / (rgb255 - rgb0)
    return scaled * 255


def intcast(tensor: torch.tensor):
    return torch.round(tensor).type(torch.int)


def logzeromean(tensor: torch.tensor):
    return tensor - (-110.9612)


def logrgbrange(tensor: torch.tensor):
    rgb255 = 8.8736
    rgb0 = -305.1051
    scaled = (tensor - rgb0) / (rgb255 - rgb0)
    return scaled * 255


softmax = torch.nn.Softmax(dim=1)


class DatasetAnalysis:
    mean = 0
    max = 0
    min = 0
    length = 0

    def __new__(cls, image):
        cls.length += 1
        cls.mean += image.mean([1, 2])
        maxval = torch.max(image)
        if maxval > cls.max:
            cls.max = maxval

        minval = torch.min(image)
        if minval < cls.min:
            cls.min = minval
        return image


transform_map = {
    "NORM": norm,
    "ABSLOG": abs_log,
    "SQLOG": square_log,
    "RANGEMAP": rangemap,
    "DATASETANALYSIS": DatasetAnalysis,
    "ZEROMEAN": zeromean,
    "RGBRANGE": rgbrange,
    "INTCAST": intcast,
    "LOGZEROMEAN": logzeromean,
    "LOGRGBRANGE": logrgbrange,
}
