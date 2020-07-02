import numpy as np
import matplotlib.pyplot as plt
import copy
import numbers


class Bar(object):
    """A single bar, which should be contained in a Barcode"""

    def __init__(self, start, end, multiplicity):
        """Constructor. Takes start/birth, end/death, and multiplicity."""
        self.start = start
        self.end = end
        self.multiplicity = int(round(multiplicity))

    def __repr__(self):
        return "Bar(%s, %s, %d)" % (self.start, self.end, self.multiplicity)

    def expand(self):
        """Returns self.multiplicity copies of this bar,
        all with multiplicity 1"""
        return [Bar(self.start, self.end, 1)] * self.multiplicity

    def to_array(self):
        return np.array([self.start, self.end, self.multiplicity])


class Barcode(object):
    """A collection of bars"""

    def __init__(self, bars=None):
        if bars is None:
            bars = []
        self.bars = bars

    def __repr__(self):
        return "Barcode(%s)" % self.bars

    def expand(self):
        return Barcode([be for b in self.bars for be in b.expand()])

    def to_array(self):
        """Returns a numpy array [[start1, end1, multiplicity1], [start2, end2, multiplicity2]...]."""
        return np.array([(b.start, b.end, b.multiplicity) for b in self.bars])


"""
Landscape classes
"""


class Landscape(object):
    """ A single landscape for a chosen k """

    def __init__(self, index, critical_points):
        self.index = index
        self.critical_points = critical_points  # an nx2 array

    def __repr__(self):
        return "Landscape(%d,%s)" % (self.index, self.critical_points)

    def plot_landscapes(self):
        """ Plots a single landscape"""
        n = np.shape(self.critical_points)[0]
        x = self.critical_points[1:n, 0]
        y = self.critical_points[1:n, 1]
        plt.plot(x, y)
        plt.show()

    def evaluate(self, xvalue):
        """ Returns the landscape value at a queried x value """
        return np.interp(xvalue, self.critical_points[1:, 0], self.critical_points[1:, 1], left=0, right=0)


# approximate data structure too
# introduce plot function for individual landscape

class Landscapes(object):
    """ Collection of non zero landscapes """

    def __init__(self, landscapes=None):
        if landscapes is None:
            landscapes = []
        self.landscapes = landscapes

    def __repr__(self):
        return "Landscapes(%s)" % self.landscapes

    def plot_landscapes(self):
        """ Plots the landscapes in the collection to a single axes"""
        for k in range(len(self.landscapes)):
            n = np.shape(self.landscapes[k].critical_points)[0]
            x = self.landscapes[k].critical_points[1:n, 0]
            y = self.landscapes[k].critical_points[1:n, 1]
            plt.plot(x, y)
        plt.show()


# Add colormap options to plots

# barcode is a barcode in the class form as in the implementation of pyrivet
# the maxind is the maximal index landscape to compute - default is max non-zero landscape
# introduce plot function for all landscapes at once


MAXIMUM_NUMBER_OF_MEGABYTES_FOR_LANDSCAPES = 50.0


class landscape(object):
    """ Collection of non zero landscapes """

    def __init__(self, barcode, x_values_range=None, x_value_step_size=None, y_value_max=None, maxind=None):
        """ Computes the collection of persistence landscapes associated to a barcode up to index maxind
        using the algorithm set out in Bubenik + Dlotko.
        :param barcode: A barcode object
        :param maxind: The maximum index landscape to calculate
        """

        barcode = barcode.expand()
        barcode = barcode.to_array()
        if maxind is None:
            maxind = len(barcode)

        self.maximum_landscape_depth = maxind

        # Apply y-value threshold
        if y_value_max is not None:
            def max_y_value(persistence_pair):
                birth, death, _ = persistence_pair
                return (death - birth) / 2.0

            barcode = filter(max_y_value, barcode)

        # Determine the minimum and maximum x-values for the
        # landscape if none are specified
        # Using map semantics here in case we want to exchange it for something
        # parallelized later
        if x_values_range is None:
            def max_x_value_of_persistence_point(persistence_pair):
                _, death, _ = persistence_pair
                return death

            def min_x_value_of_persistence_point(persistence_pair):
                birth, _, _ = persistence_pair
                return birth

            death_vector = np.array(list(map(max_x_value_of_persistence_point, barcode)))
            birth_vector = np.array(list(map(min_x_value_of_persistence_point, barcode)))
            self.x_values_range = [np.amin(birth_vector), np.amax(death_vector)]
        else:
            self.x_values_range = x_values_range

        # This parameter value is recommended; if it's not provided,
        # this calculation tries to keep the total memory for the landscape under
        # the threshold number of MiB
        if x_value_step_size is None:
            self.x_value_step_size = maxind * (self.x_values_range[1] - self.x_values_range[0]) * 64.0 / (
                    MAXIMUM_NUMBER_OF_MEGABYTES_FOR_LANDSCAPES * pow(2, 23))
        else:
            self.x_value_step_size = x_value_step_size

        def tent_function_for_pair(persistence_pair):
            birth, death, _ = persistence_pair

            def evaluate_tent_function(x):
                if x <= (birth + death) / 2.0:
                    return max(0, x - birth)
                else:
                    return max(0, death - x)

            return evaluate_tent_function

        x_values_start, x_value_stop = self.x_values_range
        width_of_x_values = x_value_stop - x_values_start
        number_of_steps = int(round(width_of_x_values / self.x_value_step_size))
        #         print('nbr of step='+str(number_of_steps))
        x_values = np.array(range(number_of_steps))
        x_values = x_values * self.x_value_step_size + x_values_start
        self.grid_values = x_values

        def x_value_to_slice(x_value):
            unsorted_slice_values = np.array(list(map(lambda pair: tent_function_for_pair(pair)(x_value), barcode)))
            return unsorted_slice_values

        landscape_slices = np.array(list(map(x_value_to_slice, x_values)))

        if maxind > landscape_slices.shape[1]:
            padding = np.zeros((number_of_steps, maxind - landscape_slices.shape[1]))
            landscape_slices = np.hstack((landscape_slices, padding))

            self.landscape_matrix = np.empty([maxind, number_of_steps])
            for i in range(number_of_steps):
                self.landscape_matrix[:, i] = landscape_slices[i, :]

        if maxind <= landscape_slices.shape[1]:
            self.landscape_matrix = np.empty([landscape_slices.shape[1], number_of_steps])
            for i in range(number_of_steps):
                self.landscape_matrix[:, i] = landscape_slices[i, :]

        # sorts all the columns using numpy's sort
        self.landscape_matrix = -np.sort(-self.landscape_matrix, axis=0)
        self.landscape_matrix = self.landscape_matrix[:maxind, :]

    def __repr__(self):
        return "Landscapes(%s)" % self.landscape_matrix

    def plot_landscapes(self, landscapes_to_plot=None):
        """ Plots the landscapes in the collection to a single axes"""
        if landscapes_to_plot is None:
            landscapes_to_plot = range(self.maximum_landscape_depth)
        elif type(landscapes_to_plot) is int:
            landscapes_to_plot = range(landscapes_to_plot)
        for k in landscapes_to_plot:
            x = self.grid_values
            y = self.landscape_matrix[k, :]
            plt.plot(x, y)
        plt.show()

    def __add__(self, other_landscape):
        if np.shape(self.landscape_matrix) != np.shape(other_landscape.landscape_matrix):
            raise TypeError("Attempted to add two landscapes with different shapes.")
        if self.x_values_range != other_landscape.x_values_range:
            raise TypeError("Attempted to add two landscapes with different ranges of x-values.")
        added_landscapes = copy.deepcopy(self)
        added_landscapes.landscape_matrix = self.landscape_matrix + other_landscape.landscape_matrix
        return added_landscapes

    def __mul__(self, multiple):
        # Scalar multiplication
        if isinstance(multiple, numbers.Number):
            multiplied_landscape = copy.deepcopy(self)
            multiplied_landscape.landscape_matrix *= multiple
            return multiplied_landscape
        # Inner product, * is element-wise multiplication
        else:
            return np.sum(self.landscape_matrix * multiple.landscape_matrix)

    def __sub__(self, other):
        return self + (-1.0) * other
