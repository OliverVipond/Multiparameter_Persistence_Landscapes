import time
import barcode
import subprocess
import shlex
import fractions
import tempfile
import os
import shutil
import numpy as np
from typing import List, Tuple

"""An interface for rivet_console, using the command line
and subprocesses."""

rivet_executable = '<rivet_console location>'


class PointCloud:
    """
    Input format for RIVET point cloud data
    """
    def __init__(self, points, appearance=None, second_param_name=None,
                 comments=None, max_dist=None):
        """
        :param points: list of tuples, or 2D numpy array of float or int
        :param appearance: list or 1D array of float or int
        :param second_param_name: str
        :param comments: str
            the comments to appear in the generated file
        :param max_dist:
            a cutoff distance beyond which no calculations will be done. If not
            provided, a maximal distance will be calculated
        """
        if second_param_name:
            self.second_param_name = second_param_name
        else:
            self.second_param_name = None
        self.points = np.array(points)
        self._appearance_has_len = hasattr(appearance, '__len__')
        if self._appearance_has_len:
            self.appearance = np.array(appearance)
            if len(appearance) != len(points):
                raise ValueError('appearance must either be None, a scalar, '
                                 'or a sequence of the same length as the points')
        else:
            self.appearance = appearance
        self.comments = comments
        self.dimension = self.points.shape[1]
        self.max_dist = max_dist or self._calc_max_dist()

    def _calc_max_dist(self):
        # Simplest possible max distance measure
        lo, hi = 0, 0
        for p in self.points:
            for coord in p.coordinates:
                if coord < lo:
                    lo = coord
                if coord > hi:
                    hi = coord
        return abs(hi - lo)

    def save(self, out):
        """
        Writes the data set to a file in RIVET point-cloud format
        :param out: a file-like object with a `write` method
        """
        if self.comments:
            out.writelines(["# " + line + "\n"
                            for line in str(self.comments).split("\n")])
        out.write("points\n")
        out.write(str(self.dimension) + "\n")
        out.write('{:f}'.format(self.max_dist) + "\n")
        if self.second_param_name is not None:
            out.write(self.second_param_name + "\n")
        else:
            out.write("no function\n")
        for i, p in enumerate(self.points):
            for c in p:
                out.write('{:f}'.format(c))
                out.write(" ")
            if self.second_param_name is not None:
                if self._appearance_has_len:
                    out.write('{:f} '.format(self.appearance[i]))
                else:
                    out.write('{:f} '.format(self.appearance or 0))
            out.write("\n")
        out.write("\n")


class Bifiltration:
    def __init__(self, x_label, y_label, simplices, appearances):
        self.x_label = x_label
        self.y_label = y_label
        self.simplices = simplices
        self.appearances = appearances

        if len(simplices) != len(appearances):
            raise ValueError("Appearances and simplices must be the same length")

    def save(self, out):
        out.write('bifiltration\n')
        out.write(self.x_label + '\n')
        out.write(self.y_label + '\n')
        for i, (simplex, appears) in enumerate(zip(self.simplices, self.appearances)):
            for v in simplex:
                out.write('{:d} '.format(v))
                out.write(" ")
            out.write('; ')
            for a in appears:
                for g in a:
                    out.write('{:d} '.format(g))
                    out.write(" ")
            out.write("\n")
        out.write("\n")


# To make multi_critical (with no appearance_values), initialize with appearance_values=None
# TODO: set appearance_values to None by default
class MetricSpace:
    def __init__(self, appearance_label, distance_label, appearance_values, distance_matrix, comment=None):
        """distance_matrix must be upper triangular"""
        self.comment = comment
        self.appearance_label = appearance_label
        self.distance_label = distance_label
        self.appearance_values = appearance_values
        self.distance_matrix = distance_matrix

    def save(self, out):
        out.seek(0)
        out.truncate()
        out.write('metric\n')
        if self.comment:
            out.write('#')
            out.write(self.comment.replace('\n', '\n#'))
            out.write('\n')
        if self.appearance_values is not None:
            out.write(self.appearance_label + '\n')
            out.write(" ".join(['{:f} '.format(s) for s in self.appearance_values]) + "\n")
        else:
            out.write("no function\n")
            out.write(str(len(self.distance_matrix)) + "\n")
        out.write(self.distance_label + '\n')
        dim = len(self.distance_matrix)
        max_dist = max(*[self.distance_matrix[i][j] for i in range(dim) for j in range(dim)])
        out.write('{:f}'.format(max_dist) + '\n')
        for row in range(dim):
            for col in range(row + 1, dim):
                # This line determines the precise representation of the output format.
                out.write('{:f} '.format(self.distance_matrix[row][col]))
            out.write('\n')


def compute_point_cloud(cloud, homology=0, x=0, y=0, verify=False):
    """
    Precomputes point cloud data using RIVET. Many functions of RIVET require
    precomputed data as input.

    :param cloud: PointCloud
        the point cloud
    :param homology: int
        the homology dimension to compute
    :param x: int
        the number of bins in the x parameter. If 0, no binning is used.
    :param y: int
        the number of bins in the y parameter. If 0, no binning is used.
    :param verify: bool
        if true, check that a valid file was generated that RIVET can read
    :return: bytes
        RIVET computes some internal data structures and these are returned as
        a byte array that can be passed to other functions in this package
    """
    return _compute_bytes(cloud, homology, x, y, verify)


def compute_bifiltration(bifiltration, homology=0, verify=False):
    return _compute_bytes(bifiltration, homology, 0, 0, verify)


def compute_metric_space(metric_space, homology=0, x=0, y=0, verify=False):
    return _compute_bytes(metric_space, homology, x, y, verify)


def _compute_bytes(saveable, homology, x, y, verify):
    with TempDir() as dir:
        saveable_name = os.path.join(dir, 'rivet_input_data.txt')
        with open(saveable_name, 'w+t') as saveable_file:
            saveable.save(saveable_file)
        output_name = compute_file(saveable_name,
                                   homology=homology,
                                   x=x,
                                   y=y)
        with open(output_name, 'rb') as output_file:
            output = output_file.read()
        if verify:
            assert bounds(output)
        return output


def barcodes(bytes, slices):
    """Returns a Barcode for each (angle, offset) tuple in `slices`.

    :param bytes: byte array
        RIVET data from one of the compute_* functions in this module
    :param slices: list of (angle in degrees, offset) tuples
        These are the angles and offsets of the lines to draw through the
        persistence module to obtain fibered barcodes.

    :return:
        A list of ((angle, offset), `Barcodes`) pairs, associating a Barcodes
        instance to each (angle, offset) in the input
    """

    with TempDir() as dir:
        with open(os.path.join(dir, 'precomputed.rivet'), 'wb') as precomp:
            precomp.write(bytes)
        with open(os.path.join(dir, 'slices.txt'), 'wt') as slice_temp:
            for angle, offset in slices:
                slice_temp.write("%s %s\n" % (angle, offset))
        return barcodes_file(precomp.name, slice_temp.name)


def _rivet_name(base, homology, x, y):
    output_name = base + (".H%d_x%d_y%d.rivet" % (homology, x, y))
    return output_name


def compute_file(input_name, output_name=None, homology=0, x=0, y=0):
    if not output_name:
        output_name = _rivet_name(input_name, homology, x, y)
    cmd = "%s %s %s -H %d -x %d -y %d -f msgpack" % \
          (rivet_executable, input_name, output_name, homology, x, y)
    subprocess.check_output(shlex.split(cmd))
    return output_name


def barcodes_file(input_name, slice_name):
    cmd = "%s %s --barcodes %s" % (rivet_executable, input_name, slice_name)
    return _parse_slices(
        subprocess.check_output(
            shlex.split(cmd)).split(b'\n'))


def betti(saveable, homology=0, x=0, y=0):
    # print("betti")
    with TempDir() as dir:
        name = os.path.join(dir, 'rivet-input.txt')
        with open(name, 'wt') as betti_temp:
            saveable.save(betti_temp)
        return betti_file(name, homology=homology, x=x, y=y)


def betti_file(name, homology=0, x=0, y=0):
    cmd = "%s %s --betti -H %d -x %d -y %d" % (rivet_executable, name, homology, x, y)
    return _parse_betti(subprocess.check_output(shlex.split(cmd)).split(b'\n'))


def bounds_file(name):
    cmd = "%s %s --bounds" % (rivet_executable, name)
    return parse_bounds(subprocess.check_output(shlex.split(cmd)).split(b'\n'))


class TempDir(os.PathLike):
    def __enter__(self):
        self.dirname = os.path.join(tempfile.gettempdir(),
                                    'rivet-' + str(os.getpid()) + '-' + str(time.time()))
        os.mkdir(self.dirname)

        return self

    def __exit__(self, etype, eval, etb):
        if etype is None:
            shutil.rmtree(self.dirname, ignore_errors=True)
        else:
            print("Error occurred, leaving RIVET working directory intact: " + self.dirname)

    def __str__(self):
        return self.dirname

    def __fspath__(self):
        return self.dirname


def bounds(bytes):
    # print("bounds", len(bytes), "bytes")
    assert len(bytes) > 0
    with TempDir() as dir:
        precomp_name = os.path.join(dir, 'precomp.rivet')
        with open(precomp_name, 'wb') as precomp:
            precomp.write(bytes)
        return bounds_file(precomp_name)


class Bounds:
    """The lower left and upper right corners of a rectangle, used to capture the parameter range for a RIVET
    2-parameter persistence module"""

    def __init__(self, lower_left, upper_right):
        self.lower_left = lower_left
        self.upper_right = upper_right

    def __repr__(self):
        return "Bounds(lower_left=%s, upper_right=%s)" % (self.lower_left, self.upper_right)

    def common_bounds(self, other: 'Bounds'):
        """Returns a minimal Bounds that encloses both self and other"""

        # TODO: rename to 'union'?
        # the lower left bound taken to be the min for the two modules,
        # and the upper right taken to be the max for the two modules.
        lower_left = [min(self.lower_left[0], other.lower_left[0]),
                      min(self.lower_left[1], other.lower_left[1])]
        upper_right = [max(self.upper_right[0], other.upper_right[0]),
                       max(self.upper_right[1], other.upper_right[1])]
        return Bounds(lower_left, upper_right)


def parse_bounds(lines):
    low = (0, 0)
    high = (0, 0)
    for line in lines:
        line = str(line, 'utf-8')
        line = line.strip()
        if line.startswith('low:'):
            parts = line[5:].split(",")
            low = tuple(map(float, parts))
        if line.startswith('high:'):
            parts = line[6:].split(",")
            high = tuple(map(float, parts))
    return Bounds(low, high)


class Dimensions:
    def __init__(self, x_grades, y_grades, ):
        self.x_grades = x_grades
        self.y_grades = y_grades

    def __repr__(self):
        return "Dimensions(x_grades=%s, y_grades=%s)" % (self.x_grades, self.y_grades)

    def __eq__(self, other):
        return isinstance(other, Dimensions) \
               and self.x_grades == other.x_grades \
               and self.y_grades == other.y_grades

    def bounds(self):
        """Calculates a minimal Bounds for these Dimensions"""
        return Bounds(
            (min(self.x_grades), min(self.y_grades)),
            (max(self.x_grades), max(self.y_grades))
        )


class MultiBetti:
    """Multi-graded Betti numbers for a 2-parameter persistence module"""
    def __init__(self,
                 dimensions: Dimensions,
                 graded_rank: np.ndarray,
                 xi_0: List[Tuple[int, int, int]],
                 xi_1: List[Tuple[int, int, int]],
                 xi_2: List[Tuple[int, int, int]]):
        self.dimensions = dimensions
        self.graded_rank = graded_rank
        self.xi_0 = xi_0
        self.xi_1 = xi_1
        self.xi_2 = xi_2

    def __repr__(self):
        return "MultiBetti(dimensions=%s, graded_rank=%s, xi_0=%s, xi_1=%s, xi_2=%s)" % \
               (self.dimensions, self.graded_rank, self.xi_0, self.xi_1, self.xi_2)


def _parse_betti(text):
    x_grades = []
    y_grades = []
    current_grades = None
    xi = [[], [], []]

    current_xi = None
    ranks = {}
    in_ranks = False

    for line in text:
        line = line.strip()
        if len(line) == 0:
            line = None
        else:
            line = str(line, 'utf-8')
        if line == 'x-grades':
            current_grades = x_grades
        elif line == 'y-grades':
            current_grades = y_grades
        elif line == 'Dimensions > 0:':
            in_ranks = True
        elif line == 'Betti numbers:':
            in_ranks = False
        elif line is None:
            current_grades = None
            current_xi = None
        elif current_grades is not None:
            current_grades.append(fractions.Fraction(line))
        elif line.startswith('xi_'):
            current_xi = xi[int(line[3])]
        elif in_ranks:
            x, y, rank = tuple(map(int, line[1:-1].split(',')))
            ranks[(x, y)] = rank
        elif current_xi is not None:
            current_xi.append(tuple(map(int, line[1:-1].split(','))))

    max_x = max_y = 0

    for x, y in ranks.keys():
        max_x = max(x, max_x)
        max_y = max(y, max_y)

    shape = (max_y + 1, max_x + 1)
    rank_mat = np.zeros(shape)
    for (x, y), rank in ranks.items():
        rank_mat[y, x] = rank

    return MultiBetti(Dimensions(x_grades, y_grades), rank_mat, *xi)


def _parse_slices(text):
    slices = []
    for line in text:
        line = line.strip()
        if not line:
            continue
        header, body = line.split(b':')
        angle, offset = header.split(b' ')
        bars = []
        for part in body.split(b','):
            part = part.strip()
            if not part:
                continue
            birth, death, mult = part.split(b' ')
            bars.append(barcode.Bar(float(birth), float(death), int(mult[1:])))

        code = barcode.Barcode(bars)
        slices.append(((float(angle), float(offset)), code))
    return slices
