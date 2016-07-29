"""
Simulation of ADF z-test critical values.  Closely follows MacKinnon (2010).
Running this files requires an IPython cluster, which is assumed to be
on the local machine.  This can be started using a command similar to

    ipcluster start -n 4

Remote clusters can be used by modifying the Client initiation.

This version has been optimized for execution on a large cluster and should
scale well with 128 or more engines.
"""
from __future__ import division, print_function

import datetime
import time

from IPython.parallel import Client
from numpy import array, savez, percentile, nan

from arch.compat.python import range, lmap

# Time in seconds to sleep before checking if ready
SLEEP = 10
# Number of repetitions
EX_NUM = 500
# Number of simulations per exercise
EX_SIZE = 200000
# Approximately controls memory use, in MiB
MAX_MEMORY_SIZE = 100

rc = Client()
dview = rc.direct_view()
with dview.sync_imports():
    from numpy import ones, vstack, arange, cumsum, sum, dot, zeros
    from numpy.random import RandomState, seed, random_integers
    from numpy.linalg import pinv


def clear_cache(client, view):
    """Cache-clearing function from mailing list"""
    assert not rc.outstanding, "don't clear history when tasks are outstanding"
    client.purge_results('all')  # clears controller
    client.results.clear()
    client.metadata.clear()
    view.results.clear()
    client.history = []
    view.history = []
    client.session.digest_history.clear()


def wrapper(n, trend, b, rng_seed=0):
    """
    Wraps and blocks the main simulation so that the maximum amount of memory
    can be controlled on multi processor systems when executing in parallel
    """
    rng = RandomState()
    rng.seed(rng_seed)
    remaining = b
    res = zeros(b)
    finished = 0
    block_size = int(2 ** 20.0 * MAX_MEMORY_SIZE / (8.0 * n))
    for j in range(0, b, block_size):
        if block_size < remaining:
            count = block_size
        else:
            count = remaining
        st = finished
        en = finished + count
        res[st:en] = adf_simulation(n, trend, count, rng)
        finished += count
        remaining -= count

    return res


def adf_simulation(n, trend, b, rng=None):
    """
    Simulates the empirical distribution of the ADF z-test statistic
    """
    if rng is None:
        seed(0)
        from numpy.random import standard_normal
    else:
        standard_normal = rng.standard_normal

    nobs = n - 1
    z = None
    if trend == 'c':
        z = ones((nobs, 1))
    elif trend == 'ct':
        z = vstack((ones(nobs), arange(1, nobs + 1))).T
    elif trend == 'ctt':
        tau = arange(1, nobs + 1)
        z = vstack((ones(nobs), tau, tau ** 2.0)).T

    y = standard_normal((n + 50, b))
    y = cumsum(y, axis=0)
    y = y[50:, :]
    lhs = y[1:, :]
    rhs = y[:-1, :]
    if z is not None:
        z_inv = pinv(z)
        beta = dot(z_inv, lhs)
        lhs = lhs - dot(z, beta)
        beta = dot(z_inv, rhs)
        rhs = rhs - dot(z, beta)

    xpy = sum(rhs * lhs, 0)
    xpx = sum(rhs ** 2.0, 0)
    gamma = xpy / xpx
    nobs = lhs.shape[0]
    stat = nobs * (gamma - 1.0)
    return stat


# Push variables and functions to all engines
dview.execute('import numpy as np')
dview['MAX_MEMORY_SIZE'] = MAX_MEMORY_SIZE
dview['wrapper'] = wrapper
dview['adf_simulation'] = adf_simulation
lview = rc.load_balanced_view()

trends = ('nc', 'c', 'ct', 'ctt')
T = array(
    (20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160,
     180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900,
     1000, 1200, 1400, 2000))
T = T[::-1]
m = T.shape[0]
percentiles = list(arange(0.5, 100.0, 0.5))
seed(0)
seeds = list(random_integers(0, 2 ** 31 - 2, size=EX_NUM))

for tr in trends:
    results = zeros((len(percentiles), m, EX_NUM)) * nan
    filename = 'adf_z_' + tr + '.npz'

    for i, t in enumerate(T):
        print("Time series length {0} for Trend {1}".format(t, tr))
        now = datetime.datetime.now()
        # Serial version
        # out = lmap(wrapper, [t] * EX_NUM, [tr] * EX_NUM,
        #            [EX_SIZE] * EX_NUM, seeds)

        # Parallel version
        res = lview.map_async(wrapper, [t] * EX_NUM, [tr] * EX_NUM,
                              [EX_SIZE] * EX_NUM, seeds)
        sleep_count = 0
        while not res.ready():
            sleep_count += 1
            elapsed = datetime.datetime.now() - now
            if sleep_count % 10:
                print('Elapsed time {0}, waiting for results'.format(elapsed,
                                                                     SLEEP))
            time.sleep(SLEEP)

        out = res.get()
        # Prevent unnecessary results from accumulating
        clear_cache(rc, lview)

        elapsed = datetime.datetime.now() - now
        print('Total time {0} for T={1}'.format(elapsed, t))
        quantiles = lmap(lambda x: percentile(x, percentiles), out)
        results[:, i, :] = array(quantiles).T

        savez(filename, trend=tr, results=results,
              percentiles=percentiles, T=T)
