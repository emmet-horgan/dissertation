from anser.parse import Parse
from anser import helper
import csv
import os


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


if __name__ == "__main__":
    write_grades("annotations_bin.csv", binary=True)
