import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import chain
from skimage import exposure
from scipy.stats import gaussian_kde

from tools import load_dff, get_colors, load_projection, get_spectrum, load_events_train, get_syncronicity_coef

# Parameters used for all plots
font_size = 20


def scale_matrix(matrix, min_out, max_out):
    min_in = np.min(matrix)
    max_in = np.max(matrix)
    return (matrix - min_in) * ((max_out - min_out) / (max_in - min_in)) + min_out


def plot_zproj_rois(age, recording, main_fig, subplot_grid): 
    """Load the video and get it's projection"""

    try:
        # Load projection
        z_proj = load_projection(age, recording)
        # Histogram equalization
        z_proj = exposure.equalize_hist(z_proj)

        # Plot the projection
        ax = main_fig.add_subplot(subplot_grid)
        ax.imshow(z_proj, cmap='gray')
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        if age=='1.young': ax.set_title('67 days', fontsize=font_size)
        elif age=='2.medium': ax.set_title('154 days', fontsize=font_size)
        elif age=='3.old': ax.set_title('240 days', fontsize=font_size)
    except: print('Could not plot zproj for ' + age + ' ' + recording)


def plot_dff(age, recording, main_fig, subplot_grid, max_cells=10):
    """Plot the dff of all the traces of a recording"""

    try:
        # Load data
        dff = load_dff(age, recording)
        time_points, n_cells = dff.shape
        if n_cells > max_cells:
            n_cells = max_cells
            dff = dff[:, :max_cells]
        time_stamps = np.linspace(0, time_points/(30/5), time_points)

        #colors = get_colors(n_cells, colormap='cool') TODO: delete this line
        # Creat a subgrid to the main grid object
        sub_grid = gridspec.GridSpecFromSubplotSpec(nrows=n_cells, ncols=1, subplot_spec=subplot_grid, hspace=1)

        # Plot traces
        for i in range(n_cells):
            # add a subplot to the main figure
            ax = main_fig.add_subplot(sub_grid[i,0])
            ax.plot(time_stamps, dff[:,i], c='black', linewidth=0.5, clip_on=False)
            if i == int(n_cells/2):
                ax.axvline(x=time_stamps[-1]+10, ymin=0, ymax=1, color='black', linestyle='-')
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                #ax.text(time_stamps[-1]+12, 0.5, '$100\%$', rotation=270) TODO: need to decide if to put a number or not
                if age=='1.young':
                    ax.set_ylabel('$\Delta f/f$', fontsize=font_size)
            else:
                ax.yaxis.set_visible(False)
            ax.spines[['left', 'top']].set_visible(False)
            ax.spines[['right', 'bottom']].set_visible(False)
            ax.tick_params(bottom = False)
            ax.tick_params(labelbottom=False)
            ax.set_ylim(-0.25, 0.35) # -0.25 is to not cut the bottom of the traces
        ax.tick_params(bottom = True)
        ax.tick_params(labelbottom=True)
        if age=='2.medium': ax.set_xlabel('time (s)', fontsize=font_size)
    except: print('Could not plot dff for ' + age + ' ' + recording)


def plot_power_spectrum(recordings, main_fig, subplot_grid):
    """Plot the power spectrum of all the data"""

    try: 
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
        ax.set_xlabel('Frequency (Hz)', fontsize=font_size)
        ax.set_ylabel('Power Spectrum', fontsize=font_size)
        ax.set_xscale('log')
        legend_without_duplicate_labels(ax)
    except: print('Could not plot power spectrum')


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def plot_ISI(recordings, main_fig, subplot_grid, n_bins=80):
    """Plot the ISI of all the data. Recordings is a dictionary where every key is an age, each element is a list of recordings' names. """

    try: 
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
        ax.set_xlabel('ISI (s)', fontsize=font_size)
        ax.set_ylabel('Count', fontsize=font_size)
        ax.legend()
    except: print('Could not plot ISI')


def plot_syncronicity(recordings, main_fig, subplot_grid):

    #try:
    # Creat a subgrid to the main grid object
    sub_grid = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=subplot_grid, hspace=0)

    colors = ['#fb8b24', '#d90368', '#820263']
    ages = ['2 months', '5 months', '8 months']
    bin_edges = np.linspace(0, 1, 10)

    for a, age in enumerate(list(recordings.keys())):
        sync_age = []
        for recording in recordings[age]:
            # Get the density coefficients
            sync_coeffs = get_syncronicity_coef(age, recording)
            sync_age.append(sync_coeffs)

        sync_age = list(chain.from_iterable(sync_age))

        ax = main_fig.add_subplot(sub_grid[a,0])
        ax.hist(sync_age, color=colors[a], label=ages[a], alpha=0.3, ec=colors[a], bins=bin_edges, density=True)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.set_yticks([])
        if a==1: ax.set_ylabel('Density', fontsize=font_size)
    ax.set_xlabel('Synchronicity coefficient', fontsize=font_size)
    #except: print('Could not plot synchronicity')


def plot_syncronicity_estimation(recordings, main_fig, subplot_grid):

    #try:
    colors = ['#fb8b24', '#d90368', '#820263']
    ages = ['2 months', '5 months', '8 months']
    bin_edges = np.linspace(0, 1, 10)

    ax = main_fig.add_subplot(subplot_grid)
    for a, age in enumerate(list(recordings.keys())):
        sync_age = []
        for recording in recordings[age]:
            # Get the density coefficients
            sync_coeffs = get_syncronicity_coef(age, recording)
            sync_age.append(sync_coeffs)

        sync_age = list(chain.from_iterable(sync_age))

        # Get the density estimation
        # Perform KDE
        density = gaussian_kde(sync_age)
        xs = np.linspace(min(sync_age), max(sync_age), 1000)
        density.covariance_factor = lambda: .5
        density._compute_covariance()
        curve = density(xs)

        ax.plot(curve, color=colors[a], label=ages[a], linewidth=1)
    
    ax.set_ylabel('Density', fontsize=font_size)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([])
    ax.set_xlabel('Synchronicity coefficient', fontsize=font_size)
    ax.spines[['right', 'top', 'left']].set_visible(False)
    #except: print('Could not plot synchronicity')


