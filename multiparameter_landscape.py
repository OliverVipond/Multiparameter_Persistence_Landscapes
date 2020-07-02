from numpy.core.multiarray import ndarray
from rivet import Bounds
import rivet
import numpy as np
import rank
from matching_distance import find_offsets
from one_parameter_classes import landscape


class multiparameter_landscape(object):
    """ Collection of non zero multiparameter landscapes """
    MAXIMUM_NUMBER_OF_MEGABYTES_FOR_LANDSCAPES = 50

    def __init__(self, computed_data, maxind=10, bounds=None, grid_step_size=None, weight=None):
        """ Returns a multiparameter landscape object, the weighted multiparameter landscape calculated in parameter
            range specified by 'bounds'

        :param computed_data: byte array
            RIVET data from a compute_* function in the rivet module
        :param bounds: [[xmin,ymin],[xmax,ymax]]
            The parameter range over which to calculate the landscape
        :resolution: int
            The number of subdivisions of the parameter space for which the multiparameter landscape is to be computed
        :index: int
            The index of the landscape wanting to be calculated
        :weight: [w_1,w_2]
            A rescaling of the parameter space corresponding to alternative interleaving distances (integer weights)
        """

        self.computed_data = computed_data

        if weight is None:
            weight = [1, 1]
        if type(weight[0]) is int and type(weight[1]) is int:
            weight = np.array(weight)
            self.weight = weight
        else:
            raise TypeError('Weighting must provide integer values')

        if bounds is None:
            self.bounds = rivet.bounds(computed_data)
        else:
            self.bounds = Bounds(lower_left=bounds[0], upper_right=bounds[1])

        self.maximum_landscape_depth = maxind

        if grid_step_size is None:
            self.grid_step_size = maxind * (self.bounds.upper_right[0] - self.bounds.lower_left[0]) * (
                    self.bounds.upper_right[1] - self.bounds.lower_left[1]) * 64.0 / (
                                          multiparameter_landscape.MAXIMUM_NUMBER_OF_MEGABYTES_FOR_LANDSCAPES
                                          * pow(2, 23))
        else:
            self.grid_step_size = grid_step_size

        self.landscape_matrix = self.compute_multiparameter_landscape

    def number_of_ysteps(self):
        number_of_ysteps = int(round((self.bounds.upper_right[1] - self.bounds.lower_left[1])
                                     / self.grid_step_size * self.weight[0] + 1))
        return number_of_ysteps

    def number_of_xsteps(self):
        number_of_xsteps = int(round((self.bounds.upper_right[0] - self.bounds.lower_left[0])
                                     / self.grid_step_size * self.weight[1] + 1))
        return number_of_xsteps

    def grid_xvalues(self):
        grid_xvalues = np.linspace(self.bounds.lower_left[0], self.bounds.upper_right[0], self.number_of_xsteps())
        return grid_xvalues

    def grid_yvalues(self):
        grid_yvalues = np.linspace(self.bounds.lower_left[1], self.bounds.upper_right[1], self.number_of_ysteps())
        return grid_yvalues

    def get_barcodes_slopes(self):
        slopes = np.degrees(np.arctan(self.number_of_xsteps() / self.number_of_ysteps())) * \
                 np.ones(self.number_of_xsteps() + self.number_of_ysteps() - 1)
        return slopes

    def get_lower_boundary_points(self):
        lower_points = np.zeros((self.number_of_xsteps() + self.number_of_ysteps() - 1, 2))
        lower_points[:self.number_of_ysteps(), 0] = self.bounds.lower_left[0]
        lower_points[self.number_of_ysteps() - 1:, 0] = self.grid_xvalues()
        lower_points[self.number_of_ysteps():, 1] = self.bounds.lower_left[1]
        lower_points[:self.number_of_ysteps(), 1] = np.flip(self.grid_yvalues(), 0)
        return lower_points

    def get_offsets(self):
        offsets = find_offsets(self.get_barcodes_slopes(), self.get_lower_boundary_points())
        return offsets

    def find_slices(self):
        slices = np.stack((self.get_barcodes_slopes(), self.get_offsets()), axis=-1)
        return slices

    def compute_landscape_barcodes(self):
        barcodes = rivet.barcodes(self.computed_data, self.find_slices())
        return barcodes

    def get_parameter_step_size(self):
        parameter_step_size = np.sqrt((self.grid_xvalues()[1] - self.grid_xvalues()[0]) ** 2
                                      + (self.grid_yvalues()[1] - self.grid_yvalues()[0]) ** 2)
        return parameter_step_size

    def get_1D_xvalues_range(self, antidiagonal_index):
        k = antidiagonal_index
        if k < min(self.number_of_ysteps(), self.number_of_xsteps()):
            x_values_range = [rank.find_parameter_of_point_on_line(self.get_barcodes_slopes()[k],
                                                                   self.get_offsets()[k],
                                                                   self.get_lower_boundary_points()[k, :]),
                              rank.find_parameter_of_point_on_line(self.get_barcodes_slopes()[k],
                                                                   self.get_offsets()[k],
                                                                   self.get_lower_boundary_points()[k, :]) +
                              self.number_of_xsteps() * self.get_parameter_step_size()]
            return x_values_range

        elif (k >= self.number_of_xsteps()) and (k < self.number_of_ysteps()):
            x_values_range = [rank.find_parameter_of_point_on_line(self.get_barcodes_slopes()[k],
                                                                   self.get_offsets()[k],
                                                                   self.get_lower_boundary_points()[k, :]),
                              rank.find_parameter_of_point_on_line(self.get_barcodes_slopes()[k],
                                                                   self.get_offsets()[k],
                                                                   self.get_lower_boundary_points()[k, :]) + (
                                  self.number_of_xsteps()) *
                              self.get_parameter_step_size()]
            return x_values_range

        elif (k >= self.number_of_ysteps()) and (k < self.number_of_xsteps()):
            x_values_range = [rank.find_parameter_of_point_on_line(self.get_barcodes_slopes()[k],
                                                                   self.get_offsets()[k],
                                                                   self.get_lower_boundary_points()[k, :]) - (
                                      k + 1 - self.number_of_ysteps()) * self.get_parameter_step_size(),
                              rank.find_parameter_of_point_on_line(self.get_barcodes_slopes()[k],
                                                                   self.get_offsets()[k],
                                                                   self.get_lower_boundary_points()[k, :]) + (
                                      self.number_of_xsteps() +
                                      self.number_of_ysteps() - k - 1) *
                              self.get_parameter_step_size()]
            return x_values_range

        elif k >= max(self.number_of_xsteps(), self.number_of_ysteps()):
            x_values_range = [rank.find_parameter_of_point_on_line(self.get_barcodes_slopes()[k],
                                                                   self.get_offsets()[k],
                                                                   self.get_lower_boundary_points()[k, :]) - (
                                      k + 1 - self.number_of_ysteps()) *
                              self.get_parameter_step_size(),
                              rank.find_parameter_of_point_on_line(self.get_barcodes_slopes()[k],
                                                                   self.get_offsets()[k],
                                                                   self.get_lower_boundary_points()[k, :]) + (
                                      self.number_of_xsteps() - k - 1 +
                                      self.number_of_ysteps()) *
                              self.get_parameter_step_size()]
            return x_values_range

    def compute_multiparameter_landscape(self):
        weighted_landscape_matrix = np.empty((self.maximum_landscape_depth,
                                              self.number_of_ysteps(),
                                              self.number_of_xsteps()))
        antidiagonal_landscape_slices = list(map(
            lambda antidiagonal_ind: landscape(self.compute_landscape_barcodes()[antidiagonal_ind][1],
                                               self.get_1D_xvalues_range(antidiagonal_ind),
                                               self.get_parameter_step_size(), maxind=self.maximum_landscape_depth),
            range(self.number_of_xsteps() + self.number_of_ysteps() - 1)))

        holder_matrix = np.zeros((self.maximum_landscape_depth,
                                  self.number_of_xsteps() + self.number_of_ysteps() - 1,
                                  self.number_of_xsteps()))

        for antidiagonal_index in range(self.number_of_xsteps() + self.number_of_ysteps() - 1):
            holder_matrix[:, antidiagonal_index, :] = antidiagonal_landscape_slices[antidiagonal_index].landscape_matrix

        for antidiagonal_index in range(self.number_of_ysteps()):
            weighted_landscape_matrix[:, antidiagonal_index, :] = holder_matrix.diagonal(antidiagonal_index, 2, 1)

        # TODO check rescaling is appropriate
        landscape_matrix: ndarray = weighted_landscape_matrix[:, ::self.weight[0], ::self.weight[1]] / np.sqrt(
            self.weight[0] ** 2 + self.weight[1] ** 2)

        return landscape_matrix

    def __repr__(self):
        return "Multiparameter Landscapes(%s)" % self.landscape_matrix
