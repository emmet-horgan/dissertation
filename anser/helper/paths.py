import os

##
# Idea:
# Allow this class to take in a name/path and then automatically generate the paths based on directories and sub-
# directories. Create the attributes and private attributes dynamically in the __init__ method.


class Paths:
    __fyp_dir = "fyp"
    __fyp_data_dir = "fyp-data"
    __fyp_lib_dir = "fyp-lib"
    __csv_data_dir = "csv-data"
    __csv_split_dir = "csv-split"
    __csv_hampel_dir = "csv-hampel"
    __csv_resample_dir = "csv-resample"
    __csv_tfds_dir = "csv-tfds"
    __csv_segmented = "csv-segmented"
    __user_path = os.environ["USERPROFILE"]
    __rr_data_dir = "rr_peaks_HIE_grades_anon_v2"
    __rr_file = "rr_peaks_HIE_grades_anon_v2.mat"

    def __init__(self, k=3, n=3):
        self.__k = k
        self.__n = n
        self.lib_path = os.path.join(self.__user_path, self.__fyp_dir, self.__fyp_lib_dir)
        self.data_path = os.path.join(self.__user_path, self.__fyp_dir, self.__fyp_data_dir)
        self.csv_data_path = os.path.join(self.data_path, self.__csv_data_dir)
        self.csv_split_path = os.path.join(self.csv_data_path, self.__csv_split_dir)
        self.csv_hampel_dir = self.__csv_hampel_dir + "-K-" + str(self.__k) + "-N-" + str(self.__n)  # k & n dependent
        self.csv_hampel_path = os.path.join(self.csv_data_path, self.csv_hampel_dir)  # k & n dependent
        self.csv_segmented_dir_path = os.path.join(self.csv_data_path, self.__csv_segmented)
        self.csv_segmented_path = os.path.join(self.csv_segmented_dir_path, self.__csv_segmented + "-K-" +
                                               str(self.__k) + "-N-" + str(self.__n))
        self.rr_path = os.path.join(self.data_path, self.__rr_data_dir, self.__rr_file)

        self.csv_tfds_path = os.path.join(self.csv_data_path, self.__csv_tfds_dir)
        self.csv_tfd_dir = "csv-tfd-K-" + str(self.__k) + "-N-" + str(self.__n)
        self.csv_tfd_path = os.path.join(self.csv_tfds_path, self.csv_tfd_dir)

        self.annotations = os.path.join(self.csv_data_path, "binary_annotations.csv")

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, new_k):
        self.__init__(new_k, self.__n)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, new_n):
        self.__init__(self.__k, new_n)

    @property
    def csv_data_dir(self):
        return self.__csv_data_dir

    @property
    def csv_split_dir(self):
        return self.__csv_split_dir
