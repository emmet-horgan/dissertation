import matlab.engine


def hampel(k, n, src_path):
    eng = matlab.engine.start_matlab()
    eng.hampel_filter_func(k, n, src_path, nargout=0)


def tfd(dopp_win, lag_win, Ntime, Nfreq, path, k, n):
    eng = matlab.engine.start_matlab()
    eng.tfd_func(dopp_win, lag_win, Ntime, Nfreq, path, k, n, nargout=0)

