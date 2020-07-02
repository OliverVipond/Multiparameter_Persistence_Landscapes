import subprocess
import tempfile
import os

import logging
import numpy as np
import socket
# note we use a constant instead of inf because of a bug in bottleneck_dist.
import time


def bottleneck_distance(left,
                        right,
                        inf=1e10,
                        cap=10,
                        # Needed to keep hera from crashing, which it does on
                        # some inputs with
                        relative_error=1e-10
                        # default relative_error. This default value is high
                        # enough to prevent it.
                        ):
    # Hera crashes when one or both barcodes are empty
    if len(left.bars) == 0 or len(right.bars) == 0:
        if len(left.bars) == 0 and len(right.bars) == 0:
            return 0
        return cap
    else:
        with tempfile.TemporaryDirectory() as temp:
            t1_name = os.path.join(temp, 'self.txt')
            t2_name = os.path.join(temp, 'other.txt')
            with open(t1_name, 'wt') as t1:
                for bar in left.bars:
                    t1.writelines(["%s %s\n" %
                                   (bar.start, min(inf, bar.end))
                                   for _ in range(bar.multiplicity)])
            with open(t2_name, 'wt') as t2:
                for bar in right.bars:
                    t2.writelines(["%s %s\n" %
                                   (bar.start, min(inf, bar.end))
                                   for _ in range(bar.multiplicity)])
            if relative_error is None:
                dist = subprocess.check_output(
                    ["bottleneck_dist", t1_name, t2_name])
            else:
                dist = subprocess.check_output(
                    ["bottleneck_dist", t1_name, t2_name, str(relative_error)])
            return min(cap, float(dist))

# note we use a constant instead of inf because of a bug in bottleneck_dist.


def multi_bottleneck_distance(lefts,
                              rights,
                              inf=1e10,
                              cap=10,
                              # Needed to keep hera from crashing, which it
                              # does on some inputs with
                              relative_error=1e-10
                              # default relative_error. This default value is
                              # high enough to prevent it.
                              ):
    if not len(lefts) == len(rights):
        raise ValueError("Lengths of `lefts` and `rights` must match")
    with tempfile.TemporaryDirectory() as temp:
        t1_name = os.path.join(temp, 'self.txt')
        t2_name = os.path.join(temp, 'other.txt')
        with open(t1_name, 'wt') as t1:
            for bars in lefts:
                for bar in bars.bars:
                    t1.writelines(["%s %s\n" %
                                   (bar.start, min(inf, bar.end))
                                   for _ in range(bar.multiplicity)])
                t1.write("--\n")
        with open(t2_name, 'wt') as t2:
            for bars in rights:
                for bar in bars.bars:
                    t2.writelines(["%s %s\n" %
                                   (bar.start, min(inf, bar.end))
                                   for _ in range(bar.multiplicity)])
                t2.write("--\n")
        try:
            if relative_error is None:
                dists = subprocess.check_output(["bottleneck_dist", t1_name, t2_name])
            else:
                dists = subprocess.check_output(
                    ["bottleneck_dist", t1_name, t2_name, str(relative_error)])
        except Exception as e:
            error_dir = "error-hera-%s-%d-%s" % (socket.gethostname(), os.getpid(), time.time())
            os.mkdir(error_dir)
            with open(os.path.join(error_dir, 'self.txt'), 'wt') as f:
                f.write(open(t1_name, 'rt').read())
            with open(os.path.join(error_dir, 'other.txt'), 'wt') as f:
                f.write(open(t2_name, 'rt').read())
            logging.error("Failure in invocation of Hera, input files copied to %s for reference",
                          error_dir, exc_info=e)
            raise
        return [min(cap, float(d)) for d in dists.splitlines()]

# note we use a constant instead of inf because of a bug in bottleneck_dist.


def array_bottleneck_distance(lefts,
                              rights,
                              inf=1e10,
                              cap=10,
                              # Needed to keep hera from crashing, which it
                              # does on some inputs with
                              relative_error=1e-10
                              # default relative_error. This default value is
                              # high enough to prevent it.
                              ):
    if len(lefts.shape) != 3:
        raise ValueError(
            "`lefts` must have shape (# of barcodes, # of bars in each code, 3)")
    if len(rights.shape) != 3:
        raise ValueError(
            "`rights` must have shape (# of barcodes, # of bars in each code, 3)")
    if lefts.shape[0] != rights.shape[0]:
        raise ValueError(
            "First dimension of both arrays must have the same length")
    with tempfile.TemporaryDirectory() as temp:
        t1_name = os.path.join(temp, 'self.txt')
        t2_name = os.path.join(temp, 'other.txt')
        with open(t1_name, 'wt') as t1:
            for code in range(lefts.shape[0]):
                for bar in range(lefts.shape[1]):
                    one_bar = lefts[code, bar, :]
                    if np.isnan(one_bar[2]) or one_bar[0] == one_bar[1]:
                        continue
                    t1.writelines(["%s %s\n" %
                                   (one_bar[0], min(inf, one_bar[1]))] *
                                  int(one_bar[2]))
                t1.write("--\n")
        with open(t2_name, 'wt') as t2:
            for code in range(rights.shape[0]):
                for bar in range(rights.shape[1]):
                    one_bar = rights[code, bar, :]
                    if np.isnan(one_bar[2]) or one_bar[0] == one_bar[1]:
                        continue
                    t2.writelines(["%s %s\n" %
                                   (one_bar[0], min(inf, one_bar[1]))] *
                                  int(one_bar[2]))
                t2.write("--\n")
        if relative_error is None:
            dists = subprocess.check_output(["bottleneck_dist", t1_name, t2_name])
        else:
            dists = subprocess.check_output(
                ["bottleneck_dist", t1_name, t2_name, str(relative_error)])
        return np.array([min(cap, float(d)) for d in dists.splitlines()])


def wasserstein_distance(left,
                         right,
                         degree,
                         inf=1e10,
                         cap=10,
                         # Needed to keep hera from crashing, which it does on
                         # some inputs with
                         relative_error=1e-10
                         # default relative_error. This default value is high
                         # enough to prevent it.
                         ):
    # Hera crashes when one or both barcodes are empty
    if len(left.bars) == 0 or len(right.bars) == 0:
        if len(left.bars) == 0 and len(right.bars) == 0:
            return 0
        return cap
    else:
        with tempfile.TemporaryDirectory() as temp:
            t1_name = os.path.join(temp, 'self.txt')
            t2_name = os.path.join(temp, 'other.txt')
            with open(t1_name, 'wt') as t1:
                for bar in left.bars:
                    t1.writelines(["%s %s\n" %
                                   (bar.start, min(inf, bar.end))
                                   for _ in range(bar.multiplicity)])
            with open(t2_name, 'wt') as t2:
                for bar in right.bars:
                    t2.writelines(["%s %s\n" %
                                   (bar.start, min(inf, bar.end))
                                   for _ in range(bar.multiplicity)])
            if relative_error is None:
                dist = subprocess.check_output(
                    ["wasserstein_dist", t1_name, t2_name, str(degree)])
            else:
                dist = subprocess.check_output(
                    ["wasserstein_dist", t1_name, t2_name, str(degree), str(relative_error)])
            return min(cap, float(dist))

# note we use a constant instead of inf because of a bug in wasserstein_dist.


def array_wasserstein_distance(lefts,
                               rights,
                               degree,
                               inf=1e10,
                               cap=10,
                               # Needed to keep hera from crashing, which it
                               # does on some inputs with
                               relative_error=1e-10
                               # default relative_error. This default value is
                               # high enough to prevent it.
                               ):
    if len(lefts.shape) != 3:
        raise ValueError(
            "`lefts` must have shape (# of barcodes, # of bars in each code, 3)")
    if len(rights.shape) != 3:
        raise ValueError(
            "`rights` must have shape (# of barcodes, # of bars in each code, 3)")
    if lefts.shape[0] != rights.shape[0]:
        raise ValueError(
            "First dimension of both arrays must have the same length")
    with tempfile.TemporaryDirectory() as temp:
        t1_name = os.path.join(temp, 'self.txt')
        t2_name = os.path.join(temp, 'other.txt')
        with open(t1_name, 'wt') as t1:
            for code in range(lefts.shape[0]):
                for bar in range(lefts.shape[1]):
                    one_bar = lefts[code, bar, :]
                    if np.isnan(one_bar[2]) or one_bar[0] == one_bar[1]:
                        continue
                    t1.writelines(["%s %s\n" %
                                   (one_bar[0], min(inf, one_bar[1]))] *
                                  int(one_bar[2]))
                t1.write("--\n")
        with open(t2_name, 'wt') as t2:
            for code in range(rights.shape[0]):
                for bar in range(rights.shape[1]):
                    one_bar = rights[code, bar, :]
                    if np.isnan(one_bar[2]) or one_bar[0] == one_bar[1]:
                        continue
                    t2.writelines(["%s %s\n" %
                                   (one_bar[0], min(inf, one_bar[1]))] *
                                  int(one_bar[2]))
                t2.write("--\n")
        if relative_error is None:
            dists = subprocess.check_output(["wasserstein_dist", t1_name, t2_name, str(degree)])
        else:
            dists = subprocess.check_output(
                ["wasserstein_dist", t1_name, t2_name, str(degree), str(relative_error)])
        return np.array([min(cap, float(d)) for d in dists.splitlines()])
