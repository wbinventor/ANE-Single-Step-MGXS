#!/usr/bin/env python

"""Plot Batchwise results, including reaction rate errors, by batch 
and clustering method."""

from collections import OrderedDict

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_particles = 1E5 # particles / batch

benchmark = str(input('Benchmark: '))

# Open the file of Batchwise results
f = h5py.File('{}/batchwise.h5'.format(benchmark), 'r')
batchwise = f['70-groups']

# Specify iterables of parameters over which to plot
clusterizers = ['openmc', 'null', 'degenerate', 'lns']
rxn_types = ['capture']
metrics = ['max', 'mean']
nuclides = ['U-238']

# Select colors from seaborn
sns.palplot(sns.color_palette("hls", len(clusterizers)))

# Add ticks to the normal Seaborn 'darkgrid' style
sns.set_style('darkgrid',
    {'xtick.major.size': 5.0,
    'xtick.minor.size': 2.0,
    'ytick.major.size': 5.0,
    'ytick.minor.size': 2.0})

# Sort keys in "batchwise.h5" file such that legends are consistently ordered
batchwise_keys = \
    [key for key in batchwise if key not in ['null', 'degenerate', 'lns']]
batchwise_keys.insert(0, 'lns')
batchwise_keys.insert(0, 'degenerate')
batchwise_keys.insert(0, 'null')


################################################################################
#  Eigenvalue Bias by Batch
################################################################################

fig = plt.figure()
legend = []

# Plot keff bias separately for pinch, combined and local clustering
for c in clusterizers:
    if c == 'openmc':
        continue

    # Add each clustering
    plt.semilogx(batchwise[c]['batches'][...] * num_particles,
                 batchwise[c]['keff']['bias'], '-o')

    if c == 'null':
        legend.append('Null')
    elif c == 'degenerate':
        legend.append('Degenerate')    
    elif c == 'lns':
        legend.append('LNS')    

# Annotate plot
plt.xlabel('# Histories', fontsize=16)
plt.ylabel(r'Bias $\Delta\rho$ [pcm]', fontsize=16)
plt.legend(legend, loc='best')
plt.grid(True)
filename = 'keff-bias-evo.png'
plt.savefig(filename, bbox_inches='tight')
plt.close(fig)


################################################################################
#  Reaction Rate Errors by Batch
################################################################################

def rel_err_stats(batchwise, rxn_type='fission',
                  metric='max', exclude=None):
    """Compute the relative error of various OpenMOC reaction rates
    with respect to fully converged OpenMC results.

    Parameters
    ----------
    batchwise : h5py.Group
        An HDF5 group of Batchwise results with reaction rates
    rxn_type : {'fission', 'capture'}
        Reaction rate of interest
    metric : {'max', 'mean'}
        The metric to compute for each batch
    exclude : re.SRE_Pattern or None, optional
        Exclude clustering algorithms which match this regular expression in
        their key in the HDF5 file

    Returns
    -------
    rel_err : dict
        A dictionary of the batchwise reaction rate relative errors
        indexed by clustering algorithm.

    """

    # Initialize a container for the relative errors
    rel_err = OrderedDict()

    # Store the percent relative error for each clustering algorithm
    # which does not meet the exclude criteria
    for c in batchwise_keys:
        if exclude is None or exclude.match(c) is None:
            rel_err[c] = \
                np.fabs(batchwise[c][rxn_type]['openmoc rel. err.'][...])
            rel_err['openmc'] = \
                np.fabs(batchwise[c][rxn_type]['openmc rel. err.'][...])

            if metric == 'max':
                rel_err[c] = np.nanmax(rel_err[c], axis=(1,2))
                rel_err['openmc'] = np.nanmax(rel_err['openmc'], axis=(1,2))
            elif metric == 'mean':
                rel_err[c] = np.nanmean(rel_err[c], axis=(1,2))
                rel_err['openmc'] = np.nanmean(rel_err['openmc'], axis=(1,2))

    return rel_err


for metric in metrics:
    rel_err = rel_err_stats(batchwise, 'capture', metric)
    fig = plt.figure()
    legend = []

    for c in clusterizers:

        if c == 'openmc':
            start = 1
        else:
            start = 0
        
        if c == 'openmc':
            plt.loglog(
                batchwise['null']['batches'][...][start:] * num_particles, rel_err[c][start:], 'k--')
        else:
            plt.loglog(
                batchwise['null']['batches'][...][start:] * num_particles, rel_err[c][start:], '-o')

        if c == 'openmc':
            legend.append('OpenMC')
        elif c == 'null':
            legend.append('Null')
        elif c == 'degenerate':
            legend.append('Degenerate')    
        elif c == 'lns':
            legend.append('LNS')    

    # Annotate plot
    plt.xlabel('# Histories', fontsize=16)
    plt.ylabel('Relative Error [%]', fontsize=16)
    plt.legend(legend, loc='best')
    plt.grid(True)

    if benchmark == 'assembly':
        plt.xlim((1e5, 1e9))
    else:
        plt.xlim((1e6, 1e9))
        
    # Save the plot
    filename = 'capt-{}-evo.png'.format(metric)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
