import math


def create_epochs(data, length: float, overlap: float, seconds=True, f=5):
    """
    Create epochs from the decimated data by default, or some input data. The length of the data must be specified
    and the level of oversampling. If the oversampling fraction/percentage is not possible, based on the window
    length, the next possible overlap which will be lower will be used
    :param length: Window length, defaults to units of seconds, but sample length can also be specified
    :param overlap: Overlap percentage, given as a float in the range of 0.0 -> 1.0
    :param seconds: Boolean to specify the units of the input, defaults to true, meaning the length is assumed to
                    be in seconds
    :param data: Input data to create epochs from, defaults to None as the default data is the decimate instance
                 attribute
    :return: List of epochs stored in lists
    """
    epochs = list()
    data_len = len(data)

    if seconds:
        length = length * f  # Calculate the number of samples from the decimated output frequency

    sampleoverlap = math.floor(overlap * length)  # Number of samples which will be overlapped
    actualoverlap = sampleoverlap / length  # Actual percentage overlap used after flooring operation

    samplenew = math.ceil((1 - overlap) * length)  # Number of new samples per epoch
    actualnew = samplenew / length  # Actual new percentage, after ceiling operation

    if samplenew + sampleoverlap != length:  # Sanity check
        raise ValueError

    total_epochs = math.floor(1 + (data_len - length) / samplenew)  # Total number of epochs

    startslice = 0  # Initialise slice indices
    endslice = length

    for epoch in range(total_epochs):
        epochs.append(data[startslice:endslice])

        startslice += length - sampleoverlap  # Increment slice indices
        endslice += length - sampleoverlap

    return epochs
