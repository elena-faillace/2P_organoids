import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import chain

from tools import load_dff, get_colors, load_video, get_spectrum, load_events_train, get_syncronicity_coef



def plot_zproj_rois(age, recording, main_fig, subplot_grid): 
    """Load the video and get it's projection"""

    # Load data
    video = load_video(age, recording)
    
    # Get the projection
    z_proj = np.mean(video, axis=0)

    # Plot the projection
    ax = main_fig.add_subplot(subplot_grid)
    ax.imshow(z_proj, cmap='gray')
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    if age=='1.young': ax.set_title('67 days')
    elif age=='2.medium': ax.set_title('154 days')
    elif age=='3.old': ax.set_title('240 days')


def plot_dff(age, recording, main_fig, subplot_grid, max_cells=10):
    """Plot the dff of all the traces of a recording"""

    # Load data
    dff = load_dff(age, recording)
    time_points, n_cells = dff.shape
    if n_cells > max_cells:
        n_cells = max_cells
        dff = dff[:, :max_cells]
    time_stamps = np.linspace(0, time_points/(30/5), time_points)

    colors = get_colors(n_cells, colormap='cool')
    # Creat a subgrid to the main grid object
    sub_grid = gridspec.GridSpecFromSubplotSpec(nrows=n_cells, ncols=1, subplot_spec=subplot_grid)

    # Plot traces
    for i in range(n_cells):
        # add a subplot to the main figure
        ax = main_fig.add_subplot(sub_grid[i,0])
        ax.plot(time_stamps, dff[:,i], c=colors[i], linewidth=0.5, clip_on=False)
        if i == int(n_cells/2):
            ax.set_ylabel('dF/F')
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        else:
            ax.yaxis.set_visible(False)
        ax.spines[['left', 'top']].set_visible(False)
        ax.spines[['right', 'bottom']].set_visible(False)
        ax.tick_params(bottom = False)
        ax.tick_params(labelbottom=False)
        ax.set_ylim(-0.25, 0.01)
    ax.tick_params(bottom = True)
    ax.tick_params(labelbottom=True)
    ax.set_xlabel('time (s)')


def plot_power_spectrum(recordings, main_fig, subplot_grid):
    """Plot the power spectrum of all the data"""

    # Add axes to the main figure
    ax = main_fig.add_subplot(subplot_grid)

    colors = ['#fb8b24', '#d90368', '#820263']
    ages = ['2 months', '5 months', '8 months']

    for a, age in enumerate(list(recordings.keys())):

        age_pw_s = []
        for recording in recordings[age]:

            # Load the data 
            dff = load_dff(age, recording)

            # Get the power spectrum
            pw_s, freqs = get_spectrum(dff.T, 30/5)

            # Use half of it for plotting
            mean_pw_s = np.mean(pw_s, axis=0)
            std_pw_s = np.std(pw_s, axis=0)

            half_freq = int(len(freqs)/2)
            half_freqs = freqs[:half_freq]
            half_mean_pw_s = mean_pw_s[:half_freq]
            half_std_pw_s = std_pw_s[:half_freq]

            # Save the mean power spectrum of this recording
            age_pw_s.append(half_mean_pw_s)

        # Find the average power spectrum of all the recordings of this age
        age_pw_s = np.array(age_pw_s)
        mean_age_pw_s = np.mean(age_pw_s, axis=0)
        std_age_pw_s = np.std(age_pw_s, axis=0)
        # Plot the power spectrum
        ax.plot(half_freqs, mean_age_pw_s, c=colors[a], label=ages[a], linewidth=1)
        ax.fill_between(half_freqs, mean_age_pw_s - std_age_pw_s, mean_age_pw_s + std_age_pw_s, color=colors[a], alpha=0.3)
            
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectrum')
    ax.set_xscale('log')
    legend_without_duplicate_labels(ax)


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def plot_ISI(recordings, main_fig, subplot_grid, n_bins=80):
    """Plot the ISI of all the data. Recordings is a dictionary where every key is an age, each element is a list of recordings' names. """

    # Add axes to the main figure
    ax = main_fig.add_subplot(subplot_grid)

    colors = ['#fb8b24', '#d90368', '#820263']
    ages = ['2 months', '5 months', '8 months']
    freq = 30/5
    bin_edges = np.linspace(0, 250, n_bins)

    for a, age in enumerate(list(recordings.keys())):
        
        # Save all the ISIs of an age as a long list with all the ISIs 
        ISI_age = []
        for recording in recordings[age]:
            # Load the data
            events = load_events_train(age, recording)

            ISI_recording = []
            for c in range(events.shape[1]):
                # find spikes times
                spikes_time = np.where(events[:,0]>0)[0]
                spikes_time = spikes_time/freq
                # find ISI
                ISI_recording.append(np.diff(spikes_time))
            ISI_age.append(list(chain.from_iterable(ISI_recording)))
            
        ISI_age = list(chain.from_iterable(ISI_age))
        # Plot the ISI of this age
        ax.hist(ISI_age, color=colors[a], label=ages[a], alpha=0.3, bins=bin_edges, ec=colors[a])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('ISI (s)')
    ax.set_ylabel('Count')
    ax.legend()


def plot_syncronicity(recordings, main_fig, subplot_grid):


    # Creat a subgrid to the main grid object
    sub_grid = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=subplot_grid, hspace=0)

    colors = ['#fb8b24', '#d90368', '#820263']
    ages = ['2 months', '5 months', '8 months']
    bin_edges = np.linspace(0, 1, 10)

    for a, age in enumerate(list(recordings.keys())):
        sync_age = []
        for recording in recordings[age]:
            sync_coeffs = get_syncronicity_coef(age, recording)
            sync_age.append(sync_coeffs)
        sync_age = list(chain.from_iterable(sync_age))

        ax = main_fig.add_subplot(sub_grid[a,0])
        ax.hist(sync_age, color=colors[a], label=ages[a], alpha=0.3, ec=colors[a], bins=bin_edges, density=True)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.set_yticks([])
        if a==1: ax.set_ylabel('Density')
    ax.set_xlabel('Synchronicity coefficient')
    



