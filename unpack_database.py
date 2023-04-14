from anser import filemanage as fm
from anser import helper

if __name__ == "__main__":
    # write_grades("binary_annotations.csv", binary=True)
    tfds = helper.Paths().csv_tfd_path
    fm.unpack(tfds, name="unpacked", split=0.9, num=120)
