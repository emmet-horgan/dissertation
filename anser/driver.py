from . import interpolate
from . import mfuncs
from . import helper
from . import parse


def pipeline(k=3, n=3, dopp_win=100, lag_win=100, Ntime=256, Nfreq=256,
             init=False,
             start="hampel"):
    process = ("hampel", "resample", "segment", "tfd")
    if init:
        parser = parse.Parse()
        parser.create_csv()

    paths = helper.Paths(k=k, n=n)
    if start is process[0]:
        mfuncs.hampel(k, n, paths.csv_split_path)

    if start in process[0:2]:
        resample = interpolate.Resample(path=paths.csv_hampel_path, k=k, n=n)
        resample.interpolate()
        resample.create_csv(name="csv-resample/csv-resampled-K-" + str(k) + "-N-" + str(n), dropped=True)

        resample.segment()
        resample.write_segment()

    elif start in process[0:3]:
        segment = interpolate.Resample(path=paths.csv_hampel_path, k=k, n=n)
        segment.segment()
        segment.write_segment()
    
    mfuncs.tfd(dopp_win, lag_win, Ntime, Nfreq, paths.csv_segmented_path, k, n)


# pipeline(3, 3, Ntime=224, Nfreq=224)

