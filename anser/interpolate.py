from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from . import parse
from .helper import paths as paths
from .helper import create_epochs
import math
import os


class Resample(parse.Parse):
    __epoch_win_sec = 5 * 60  # 5 minutes in seconds

    def __init__(self, path=None, kind="cubic", fs=5, threshold_nan=3, tolerance=1.0, k=3, n=3):
        self.k = k
        self.n = n
        self.paths = paths.Paths(k=k, n=n)
        if path is None:
            self.path = self.paths.csv_hampel_path
        else:
            self.path = path
        self.tolerance = tolerance
        self.kind = kind
        self.fs = fs

        self.dropped = list()
        self.data_dropped = 0
        self.total_data = 0
        self.percentage_dropped = 0

        self.seg_data = {}

        self.__epoch_win_samples = self.__epoch_win_sec * self.fs  # Number of samples in a 5-minute window
        super().__init__(path=self.path, threshold_nan=threshold_nan)

    @staticmethod
    def clinspace(start, end, f):
        T = 1 / f
        container = list()
        tmp = start
        while tmp < end:
            container.append(tmp)
            tmp += T
        return container

    def interpolate(self):
        """
        Resample the segmented data cusing cubic spline interpolation by default. If length of time over which the
        segment operates is too small, by default this is less that 5 minutes, then add that subject and section to a
        list of dropped segments and do not resample the segment. This data will not be written to disk in the csv files.
        Returns: None

        """
        if self.path is None:
            self.path = self.paths.csv_hampel_path

        for key in self.csv_data:
            for itr in range(len(self.csv_data[key]["rr_interval"])):
                length = self.csv_data[key]["rr_interval"][itr].data.shape[0] if len(self.csv_data[key]["rr_interval"][itr].data.shape) == 1 else 1
                self.total_data += length

                x = self.csv_data[key]["rr_peaks"][itr].data
                y = self.csv_data[key]["rr_interval"][itr].data

                if (x[-1] - x[0] if len(x.shape) == 1 else 1) < self.tolerance * (5 * 60):
                    self.dropped.append([key, self.csv_data[key]["rr_interval"][itr].section])
                    self.data_dropped += length
                    continue

                spline = interp1d(x, y, fill_value="extrapolate", kind=self.kind)
                # self.samples = np.linspace(x[0], x[-1], math.ceil((x[-1] - x[0]) * self.fs), endpoint=False)
                samples = self.clinspace(x[0], x[-1], self.fs)  # Use custom linspace command to generate samples
                self.csv_data[key]["rr_interval"][itr].data = spline(samples)   # Sample using cubic spline
                self.csv_data[key]["rr_peaks"][itr].data = samples

    def data_loss(self):
        if len(self.dropped) == 0:
            raise FileNotFoundError
        self.percentage_dropped = self.data_dropped / self.total_data * 100
        print("Percentage Dropped = ", self.percentage_dropped)

    def time2sample(self, time):
        return math.ceil(time * self.fs) if math.ceil(time * self.fs) < 3600 * self.fs else math.floor(time * self.fs)

    def segment(self, overlap=0.5, f=5, length=300):
        for key in self.csv_data:
            self.seg_data[key] = []
            for data in self.csv_data[key]["rr_interval"]:
                if [key, data.section] in self.dropped:
                    continue
                epochs = create_epochs(data.data, overlap=overlap, length=length, f=f)
                for epoch in epochs:
                    self.seg_data[key].append(epoch)

        self.seg_data = {k: v for k, v in self.seg_data.items() if v}  # Check for empty keys

    def write_segment(self):
        if not os.path.isdir(self.paths.csv_segmented_dir_path):
            os.mkdir(self.paths.csv_segmented_dir_path)
        os.chdir(self.paths.csv_segmented_dir_path)
        direc = "csv-segmented-K-" + str(self.k) + "-N-" + str(self.n)
        if os.path.isdir(direc):
            raise IsADirectoryError
        else:
            os.mkdir(direc)
        os.chdir(direc)
        for key in self.seg_data:
            os.mkdir(key)
            os.chdir(key)
            i = 0
            for epoch in self.seg_data[key]:
                np.savetxt("segment" + str(i) + ".csv", epoch, delimiter=",")
                i += 1
            os.chdir("..")

def main():
    obj = Resample()
    # plt.plot(obj.csv_data["subject0"]["rr_peaks"][0].data, obj.csv_data["subject0"]["rr_interval"][0].data, '.', label="non-interpolated")
    obj.interpolate()
    obj.segment()
    obj.write_segment()

    # plt.plot(obj.csv_data["subject0"]["rr_peaks"][0].data, obj.csv_data["subject0"]["rr_interval"][0].data, label="interpolated")
    #plt.grid(True)
    #plt.xlabel("Time (sec)")
    #plt.ylabel("Interval (sec)")
    #plt.title("Original HRV Signal & Resampled HRV Signal")
    #plt.tight_layout()
    #plt.legend()
    #obj.data_loss()
    #plt.show()
    #obj.create_csv("csv-resample/csv-resampled-K3N3", dropped=obj.dropped)


def inter_pre_hampel():
    titlespec = {
        "fontname": "DejaVu Sans",
        "fontsize": 14,
    }
    axesspec = {
        "fontname": "DejaVu Sans",
        "fontstyle": "oblique",
        "fontsize": 12,
    }
    path = paths.Paths()
    obj = Resample(path=path.csv_split_path)
    plt.plot(obj.csv_data["subject1"]["rr_peaks"][1].data, obj.csv_data["subject1"]["rr_interval"][1].data, '.',
             label="non-interpolated")
    obj.interpolate()
    plt.plot(obj.csv_data["subject1"]["rr_peaks"][1].data, obj.csv_data["subject1"]["rr_interval"][1].data,
             label="interpolated")
    plt.grid(True)
    plt.xlabel("Peak Times [s]", **axesspec)
    plt.ylabel("Delta [s]", **axesspec)
    plt.title("HRV Plot: Subject 2", **titlespec)
    plt.tight_layout()
    plt.legend()
    plt.show()

    #obj.create_csv("csv-resample/csv-resampled-pre-hampel")


if __name__ == "__main__":
    main()
    # inter_pre_hampel()
