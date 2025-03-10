# -*- coding: utf-8 -*-
"""
Module containing the SubplotCycler Class, a utility to plot datasets
containing multiple curves in a single window, cycle through them
using buttons, and save the figures.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

class SubplotCycler:
    """
    Create an object that permits to cycle between graphs on the same window.
    Supports plots with multiple rows as well.
    """
    def __init__(self,
                 figure: plt.Figure,
                 axes: np.ndarray,
                 simultaneous_plots: int = 1):
        """
        Initialises the cycler object by passing it the figure and the
        axes on which it has to act, and the number of plots that the
        user wishes to show simultaneously

        Parameters
        ----------
        figure : plt.Figure
            Figure object on which the cycler will act.
        axes : np.ndarray
            Array containing.
        simultaneous_plots : int, optional
            Number of subplots columns that the user wishes to show at once.
            The default is 1.

            The maximum is 4 in case of multi-row plots (i.e. ``axes`` is
            a 2D array), for compatibility with the 4 buttons.

            In case of a single-row plot (i.e. ``axes`` is a 1D array)
            and ``simultaneous_plots > 4``, the subplots will be rearranged
            on multiple rows to make them fit in the same window.

        Raises
        ------
        RuntimeError
            An error is raised if you have N subplots and want to
            show M at a time, but M is not an even divisor of N, or
            if you have a multi-row plot (i.e. ``axes`` is a 2D array)
            but you set ``simultaneous_plots > 4``.

        Returns
        -------
        None.

        """
        self.fig = figure
        # Number of columns to be shown at once
        if simultaneous_plots <= 4:
            self.n_cols = simultaneous_plots
        else:
            self.n_cols  = 4

        if axes.size % simultaneous_plots != 0:
            raise RuntimeError('simultaneous_plots is not an even divisor '
                               'of the total number of plots.')

        if (axes.ndim > 1) and (axes.shape[0] > 1):
            add_ratios = [0.15]
            self.axes = axes
            self.n_subplots = len(axes[0])  # Number of columns in axes
            self.n_rows = len(axes)         # Number of rows in axes
            self.n_pages = self.n_subplots // simultaneous_plots
            if simultaneous_plots > 4:
                raise RuntimeError('If input axes are 2D, the maximum number '
                                   'of simultaneous_plots should be 4')
            if self.n_subplots % simultaneous_plots != 0:
                raise RuntimeError('simultaneous_plots is not an even divisor '
                                   'of the total number of plots.')
        elif (axes.ndim > 1) and (axes.shape[0] == 1):
            axes = axes.reshape(-1)

        if axes.ndim == 1:
            if simultaneous_plots <= 4:
                add_ratios = [0.15]
                self.axes = np.reshape(np.append(axes, np.zeros(len(axes))),
                                       (2, len(axes)))
                self.n_rows = 1         # Number of rows in axes
                self.n_subplots = len(axes)  # Number of columns in axes
                self.n_pages = len(axes) // simultaneous_plots
            else:
                add_ratios = [1 for _ in range(simultaneous_plots//4)] + [0.15]
                self.n_rows = (simultaneous_plots // 4) + 1
                self.n_subplots = len(axes)
                self.n_pages = len(axes) // simultaneous_plots
                axes = axes.reshape((self.n_pages, -1))
                self.axes = np.empty((self.n_rows, (4 * self.n_pages)),
                                     dtype = plt.Axes)
                for i, row in enumerate(axes):
                    for j, ax in enumerate(row):
                        self.axes[j//4, 4*i + j%4] = ax

        self.current_subplot = 0
        self.width = 4 // self.n_cols

        # Get height ratios to be preserved
        self.height_ratios = (
            self.axes[0,0].get_subplotspec().get_gridspec().get_height_ratios()
            + add_ratios)

        # Reshape grid to accomodate the buttons
        self.gs = GridSpec(self.n_rows+1,
                           4,
                           height_ratios = self.height_ratios,
                           figure = self.fig)

        # Reposition all subplots in the new grid and hide them
        for i in range(self.n_rows):
            for j in range(self.n_pages * self.n_cols):
                if self.axes[i,j] is None:
                    self.axes[i,j] = plt.subplot()
                    self.axes[i,j].axis('off')
                self.axes[i,j].set_subplotspec(self.gs[i, j % 4])
                self.axes[i,j].set_visible(False)

        # Create buttons for navigating through subplots within the same figure
        self.axsave = self.fig.add_subplot(self.gs[self.n_rows, 0])
        self.axsaveall = self.fig.add_subplot(self.gs[self.n_rows, 1])
        self.axprev = self.fig.add_subplot(self.gs[self.n_rows, 2])
        self.axnext = self.fig.add_subplot(self.gs[self.n_rows, 3])
        self.bnext = Button(self.axnext, 'Next')
        self.bprev = Button(self.axprev, 'Previous')
        self.bnext.on_clicked(self.next_subplot)
        self.bprev.on_clicked(self.prev_subplot)

        self.bsave = Button(self.axsave, 'Save Figure')
        self.bsaveall = Button(self.axsaveall, 'Save All Figures')
        self.bsave.on_clicked(self.save)
        self.bsaveall.on_clicked(self.save_all)

        # Draw the initial subplot
        self.show_subplot()
        #self.fig.set_size_inches(16, 9)

    def show_subplot(self):
        """
        Show the current subplots and adjust their positions dynamically
        to fill the figure
        """
        # Hide all subplots
        for i in range(self.n_rows):
            for j in range(self.n_pages * self.n_cols):
                self.axes[i,j].set_visible(False)

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                axis = self.axes[i, self.current_subplot + j]
                axis.set_subplotspec(
                    self.gs[i,(j * self.width):((j+1) * self.width)])
                axis.set_visible(True)

        self.fig.canvas.draw()

    def next_subplot(self, event):
        """
        Switches to the next group of subplots when the buton is clicked
        """
        if self.current_subplot < (self.n_pages - 1) * self.n_cols:
            self.current_subplot += self.n_cols
            self.show_subplot()

    def prev_subplot(self, event):
        """
        Switches to the previous group of subplots when the buton is clicked
        """
        if self.current_subplot > 0:
            self.current_subplot -= self.n_cols
            self.show_subplot()

    def save(self, event):
        """
        Saves current figure in png format when the buton is clicked
        """
        if os.getcwd().split('\\')[-1] != 'figures':
            if not os.path.exists('./figures/'):
                os.mkdir('./figures/')
            os.chdir('./figures')
        if self.axes[0,self.current_subplot].get_title() != '':
            title = self.axes[0,self.current_subplot].get_title()
            title = title.replace('.', '_')
            title = re.sub(r'\W+', '', title)
            fname = title + f'_{self.current_subplot:>03}.png'
        else:
            title = self.axes[0,self.current_subplot].get_ylabel()
            title = title.replace('.', '_')
            title = re.sub(r'\W+', '', title)
            fname = title + f'{self.current_subplot:>03}.png'
        fname = fname.replace(' ', '_')
        print('Saving Figure: ' + fname)
        self.fig.savefig(fname, dpi = 300)

    def save_all(self, event):
        """
        Saves all figures in png format when the buton is clicked
        """
        self.current_subplot = 0
        self.show_subplot()
        for _ in range(self.n_subplots // self.n_cols):
            self.save(event)
            self.next_subplot(event)

if __name__ == "__main__":

    # Example usage
    plt.close('all')

    plt.rc('axes', titlesize=14)    # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)   # fontsize of the tick labels
    plt.rc('legend', fontsize=10)   # legend fontsize
    plt.rc('figure', titlesize=16)  # fontsize of the figure title

    N_STEPS = 28
    fig, ax = plt.subplots(1,
                           N_STEPS,
                           figsize=(15, 8),
                           #height_ratios = [1,1],
                           layout = "tight",
                           squeeze = False)

    # Example plotting code
    for k in range(N_STEPS):
        q = np.linspace(0.01, 1, 100)
        I = np.exp(-q * (k + 1))
        frame_ids = np.arange(100)
        isum_i = np.sin(frame_ids * 0.1) + k
        good_frames = frame_ids[frame_ids % 10 == 0]

        ax[0,k].plot(q, I, color='tab:red')
        ax[0,k].set_xlabel(r'$q\ (\AA^{-1})$')
        ax[0,k].set_ylabel('Scattering Intensity (A.U.)')
        ax[0,k].set_title('I(q)')

        if ax.shape[0] > 1:
            ax[1,k].plot(frame_ids - k * N_STEPS, isum_i)
            ax[1,k].plot(good_frames - k * N_STEPS,
                          isum_i[good_frames],
                          'o',
                          color=f'C{k}')
            ax[1,k].set_xlabel('Frame Number')
            ax[1,k].set_ylabel('Integrated Scattering Intensity (A.U.)')

    # Create the SubplotCycler instance with the figure and axes
    cycler = SubplotCycler(fig, ax, 7)
    plt.show()
