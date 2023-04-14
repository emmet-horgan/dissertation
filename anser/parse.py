import scipy.io as scio
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from .helper import paths as paths
from scipy.interpolate import interp1d
import re


class Data:

    def __init__(self, data=None, subject=None, grade=None, section=None):
        self.__data = data
        self.__subject = subject            # Defaults to None
        self.__grade = grade                # Default to None
        self.__section = section            # Defaults to None

    @property
    def data(self):
        if self.__data is None:
            raise ValueError
        else:
            return self.__data

    @property
    def section(self):
        if self.__section is None:
            raise ValueError
        else:
            return self.__section

    @property
    def subject(self):
        if self.__subject is None:
            raise ValueError
        else:
            return self.__subject

    @data.setter
    def data(self, data):
        self.__data = data

    @section.setter
    def section(self, section):
        self.__section = section

    @subject.setter
    def subject(self, subject):
        self.__subject = subject


class Parse:
    """
    This class is used to parse and process the heart rate variability data. Ideally it takes in a '.mat' or '.csv' file
    containing time series data. If there is a NaN at some point throughout the data, depending on the number of NaNs in
    a row, it will either linear interpolate to replace the NaNs or split the data into separate sections. When a split
    occurs, the data will be written into a directory, which can be specified but will be defaulted for my use during this
    project.
    This class should simply read data and perform some pre-processing operations and no more so that its complexity does
    not grow exponentially and becomes hard to maintain.
    """

    def __init__(self, path=None, threshold_nan=3):
        self.paths = paths.Paths()
        if path is None:
            self.path = self.paths.rr_path
        else:
            self.path = path
        self.__threshold_nan = threshold_nan
        self._ext = os.path.split(self.path)[-1].split('.')[-1]  # Get the file extension
        if self._ext == "mat":
            self.rr_peaks, self.rr_interval, self.grades = self.parse_mat()
            self.csv_data = self.nan_process()
        elif os.path.isdir(path):
            self.csv_data = self.parse_csv(path=self.path)
        else:
            raise FileNotFoundError

    def parse_mat(self):

        data = scio.loadmat(self.path)["rr_peaks_st"]
        rr_interval = data["rr_interval"]  # variability
        rr_peaks = data["rr_peaks"]  # time

        for x in range(len(rr_interval[0])):
            tmp = []  # Initialise an empty list
            for y in range(len(rr_interval[0][x])):
                tmp.append(rr_interval[0, x][y, 0])  # Append the values in the array to list
            rr_interval[0, x] = np.array(tmp)  # Convert list to np.array and set equal to the total array, removing the
            # pointless dimension

        for x in range(len(rr_peaks[0])):  # remove first index of timing arr so that arrays match
            rr_peaks[0, x] = np.delete(rr_peaks[0, x], 0)

        grading = data["EEG_grade"]
        for x in range(len(grading[0])):
            grading[0][x] = grading[0][x][0][0]  # Fix numpy indexing which requires a lot of indices because,
            # everything is a 2-d array

        return rr_peaks[0], rr_interval[0], grading[0]  # Return the first index to remove unnecessary dimension

    def parse_csv(self, subject=None, path=None):
        if path is None:
            path = self.paths.csv_split_path
        # paths = Paths()
        data = dict()
        os.chdir(path)
        subjects = os.listdir()
        for itr in subjects:
            data[itr] = dict()
            if not os.path.isdir(itr):
                raise NotADirectoryError
            os.chdir(itr)
            for jtr in os.listdir():
                data[itr][jtr] = list()
                if not os.path.isdir(jtr):
                    raise NotADirectoryError
                os.chdir(jtr)
                for ktr in os.listdir():
                    if not os.path.isfile(ktr) or ktr.split(".")[-1] != "csv":
                        raise FileNotFoundError
                    section = re.search(r"section-.*-", ktr)    # Identify 'section-<number>-' substring
                    if not section:     # There are two types, must be one, check for the other type
                        section = re.search(r"section-.*\.", ktr).group(0)
                        section = section.split("-")[1].split(".")[0]
                        section = int(section)
                    else:
                        section = section.group(0).split("-")[1]
                        section = int(section)

                    data[itr][jtr].append(Data(data=np.genfromtxt(ktr, delimiter=","),
                                               section=section,
                                               subject=itr))
                os.chdir("..")
            os.chdir("..")
        os.chdir(self.paths.lib_path)
        return data

    def plot(self, index, grade=False):
        if type(self.rr_interval[index]).__module__ != "numpy":
            raise TypeError
        if len(self.rr_peaks[index]) != len(self.rr_interval[index]):
            # Lengths don't match !
            raise IndexError
        fig = plt.figure(figsize=(10, 6))

        if grade is not False:
            plt.plot(self.rr_peaks[index], self.rr_interval[index], label="Grade " + str(self.grades[index]))
            plt.legend()
        else:
            plt.plot(self.rr_peaks[index], self.rr_interval[index])
        plt.show()

    @staticmethod
    def remove_nans(x, y):
        index = (False == np.isnan(y))
        tmp_y = y[index]
        tmp_x = x[index]
        if len(tmp_y) != len(tmp_x):
            raise IndexError
        # if len(nan_indices(tmp_x) or len(nan_indices(tmp_y))):
        #    raise TypeError
        return tmp_x, tmp_y

    def nan_indices(self, index):
        return list(np.argwhere(np.isnan(self.rr_interval[index])))

    def nan_process(self, index=None):
        """
        Identifies all of the NaNs in the rr_interval data and if appropriate, linear interpolates the data. Otherwise
        it will split the data up into several sections of contigious non-nan corrupted data segments. The resultant
        data is stored as object attribute called "csv_data", it is a dictionary and contatins segments of data at its
        core.
        """
        csv_data = {}

        for x in range(len(self.rr_interval)):
            nans = self.nan_indices(x)
            lst = []
            # Try to wrap this algorithm in a function/method at some point
            # It returns a list of lists, each list contains the elements of the input list which were incrementing in
            # numeric order
            y = 0
            while y < len(nans) - 1:  # for y in range(len(nans))
                sub = [nans[y]]
                for z in range(y + 1, len(nans)):
                    if nans[z] - nans[y] == z - y:
                        sub.append(nans[z])
                        if z == len(nans) - 1:  # No more values in the list to check
                            y = z - 1
                            break
                    else:
                        y = z - 1  # Subtraction will cancel when we add 1 below
                        break
                lst.append(sub)
                y += 1

            file_data_interval = []
            file_data_peaks = []
            for y in lst:
                if len(y) <= self.__threshold_nan and not y[0] == 0 and not y[-1] == len(self.rr_interval[x]):
                    model = interp1d(np.array([self.rr_peaks[x][y[0] - 1], self.rr_peaks[x][y[-1] + 1]]).squeeze(),  # time axis data
                                     np.array([self.rr_interval[x][y[0] - 1], self.rr_interval[x][y[-1] + 1]]).squeeze(),  # interval axis data
                                     kind="linear")
                    for z in y:
                        self.rr_interval[x][z] = model(self.rr_peaks[x][z])  # replace NaNs with linear interpolation
                    lst.remove(y)
                else:
                    pass  # Create new CSV files
            indices = []
            for itr in lst:
                if itr[0][0] != 0:
                    indices.append(itr[0][0])
                if itr[-1][0] != len(self.rr_interval[x]) - 1:
                    indices.append(itr[-1][0] + 1)

            split_interval = np.split(self.rr_interval[x], indices)
            split_peaks = np.split(self.rr_peaks[x], indices)

            for itr in range(len(split_interval)):
                if not np.any(np.isnan(split_interval[itr])):
                    file_data_interval.append(split_interval[itr])
                    file_data_peaks.append(split_peaks[itr])

            csv_data["subject" + str(x)] = dict()

            csv_data["subject" + str(x)]["rr_interval"] = list()
            csv_data["subject" + str(x)]["rr_peaks"] = list()

            csv_data["subject" + str(x)]["rr_interval"] = file_data_interval
            csv_data["subject" + str(x)]["rr_peaks"] = file_data_peaks

        return csv_data

    def create_csv(self, name=None, dropped=False):
        """
        Create csv files from the csv_data attribute
        Args:
            name: Name/Path of root folder
            dropped: List containing information which identifies which data sections have been dropped and is used such
                     that dropped data is not written to the csv files. This means that resampled data must be written
                     to disk as csv files and then re-read using the parse_csv function. This is not ideal should be
                     improved in future software iterations. The form of the dropped data should be:
                     list(list(str, int)) where the str is the subject identifier and the int is the section number.

        Returns: None

        """
        if name is None:
            name = self.paths.csv_split_dir

        os.chdir(self.paths.data_path)
        if os.path.isdir(self.paths.csv_data_dir):
            os.chdir(self.paths.csv_data_dir)
        else:
            os.mkdir(self.paths.csv_data_dir)
            os.chdir(self.paths.csv_data_dir)

        if os.path.isdir(name):
            raise FileExistsError
        elif len(os.path.split(name)) > 1:
            os.makedirs(name)
        else:
            os.mkdir(name)

        os.chdir(name)
        for key in self.csv_data:
            os.mkdir(key)
            os.chdir(key)
            for itr in self.csv_data[key]:
                count = 0
                os.mkdir(itr)
                os.chdir(itr)
                for jtr in self.csv_data[key][itr]:
                    ctd = False
                    if dropped:
                        for lst in self.dropped:
                            if lst == [key, jtr.section]:
                                ctd = True
                    if ctd:
                        continue
                    if type(jtr) is Data:
                        np.savetxt(key + "-" + itr + "-" + "section" + "-" + str(count) + ".csv", jtr.data, delimiter=",")
                    else:
                        np.savetxt(key + "-" + itr + "-" + "section" + "-" + str(count) + ".csv", jtr, delimiter=",")
                    count += 1
                os.chdir("..")

            os.chdir("..")
        os.chdir(self.paths.lib_path)


def main():
    obj = Parse()
    obj.create_csv()


if __name__ == '__main__':
    main()

