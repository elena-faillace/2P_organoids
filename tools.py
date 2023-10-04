import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from oasis.functions import deconvolve
from scipy.ndimage import gaussian_filter1d

# TODO: change this path to the path where you have the data
path_to_origin = '/Volumes/T7/organoids/for_paper/data/'


def moving_average(traces, window):
    """Returns the moving average of the traces, a np.array (time_points, n_cells)"""
    time_points, n_cells = traces.shape[0], traces.shape[1]
    moving_average = np.zeros((time_points, n_cells))
    for cell in range(n_cells):
        moving_average[:, cell] = np.convolve(traces[:, cell], np.ones(window), 'same') / window
    return moving_average


def bin_traces(traces, bin_size):
    """Bin the traces (time_points x cells) by averaging each bin."""
    new_length, n_cells = int(traces.shape[0]/bin_size), traces.shape[1]
    new_signal = np.zeros((new_length, n_cells))
    for i in range(new_length):
        new_signal[i, :] = np.mean(traces[i*bin_size:(i+1)*bin_size, :], axis=0)
    return new_signal


def get_dff(traces):
    """Calculate the df/f of the traces, a np.array (time_points, n_cells)"""
    time_points, n_cells = traces.shape[0], traces.shape[1]
    dff = np.zeros((time_points, n_cells))
    for cell in range(n_cells):
        d0 = np.mean(traces[:, cell])
        dff[:, cell] = (traces[:, cell] - d0) / d0
    return dff


def get_spike_trains(traces):
    """Use OASIS to deconvolute spike trains"""
    events_train = np.zeros(traces.shape)
    for ev in range(traces.shape[1]):
        c, s, b, g, lam = deconvolve(traces[:,ev], g=(None,None), penalty=1)
        events_train[:,ev] = s
    return events_train


def load_dff(age, recording):
    """Load the dff of the recording, a np.array (time_points, n_cells)"""
    dff = np.load(path_to_origin + age + '/' + recording + '/dff.npy')
    return dff


def load_raw_traces(age, recording):
    """Load the traces as outputs from segmentation"""
    traces = np.load(path_to_origin + age + '/' + recording + '/raw_traces.npy')
    return traces


# TODO: might need to delete this
def load_events_train(age, recording):
    """Load the events train found with OASIS"""
    events_train = np.load(path_to_origin + age + '/' + recording + '/spike_trains.npy')
    return events_train


def load_video(age, recording):
    """Load the .tif video of the recording"""
    input_file = path_to_origin + age + '/' + recording + '/video.tif'
    video = io.imread(input_file)
    return video


def load_projection(age, recording):
    """Load the projection of the video"""
    input_file = path_to_origin + age + '/' + recording + '/projection.npy'
    z_proj = np.load(input_file)
    return z_proj


def get_colors(n, colormap='rainbow'):
    colors = []
    cmap = plt.cm.get_cmap(colormap)
    for colorVal in np.linspace(0, 1, n+1):
        colors.append(cmap(colorVal))
    return colors[:-1]


def get_spectrum(x, f):
    ''' Extract the spectrum with Fourier transform and the associated frequencies.
    INPUT:
    x = data to compute the fourier tranform on, in the form NxT (N=number of neurons, T=time samples)
    f = sampling frequency

    OUTPUTS:
    Sxx = spectrum of the signal, it is mirrored along the center
    faxis = frequency axis, based on sampling frequqncy, it will end with the Nyquist frequency
    '''
    xf = fft((x.T - x.mean(axis=1)).T)
    Sxx = np.real(np.abs(xf)) # Compute spectrum
    faxis = fftfreq(Sxx.shape[-1], d=1/f)

    return Sxx, faxis


def get_syncronicity_coef(age, recording):
    """Syncronicity found as the correlation coefficient of the spike trains"""

    # Load spike trains
    events = load_dff(age, recording)

    # Get correlation coefficients of each neuron with each other
    corr = np.corrcoef(events.T)
    corr = np.tril(corr, k=-1)
    corr = corr[corr>0]

    return list(corr)


