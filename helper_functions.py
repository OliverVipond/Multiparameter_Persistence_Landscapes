import numpy as np
from one_parameter_classes import Bar, Barcode, Landscape, Landscapes
import rivet


def sample_circle_disc(alpha, n):
    a = int(np.floor(alpha * n))
    b = n - a
    theta = np.random.uniform(0, 2 * np.pi, a)
    circle_points = np.vstack((np.sin(theta), np.cos(theta)))
    r = np.array(np.random.uniform(0, 1, b))
    t = np.random.uniform(0, 2 * np.pi, b)
    disc_points = np.multiply(np.vstack((np.sin(t), np.cos(t))), np.sqrt(r))
    noisy_points = np.hstack((circle_points, disc_points)).T
    return noisy_points


# Return ptcloud (samples,dims) with zero mean and unit variance (expected l2 norm is 1)

def normalise_pointcloud(pts):
    pts = pts - np.mean(pts, axis=0)
    variance = np.multiply(pts, pts).sum() / pts.shape[0]
    pts = pts / np.sqrt(variance)
    return pts


# Normalise a filter by making 100-alpha% of the points lie in the range 0 to 1
def normalise_filter(filter_values, alpha):
    filter_values = filter_values - np.percentile(filter_values, alpha / 200)
    filter_values = filter_values / np.percentile(filter_values, 100 - alpha / 200)
    return filter_values


def subsample(points, n):
    nbr_of_points = points.shape[0]
    inds = np.zeros(nbr_of_points, dtype=int)
    inds[:np.min((n, nbr_of_points))] = 1
    np.random.shuffle(inds)
    inds = inds.astype(bool)
    return points[inds, :]


def ripser_to_rivet_bcode(ripserBars):
    bars = [Bar(ripserBars[n][0], ripserBars[n][1], 1) for n in range(len(ripserBars))]
    return Barcode(bars)


def compute_landscapes(barcode, maxind=None):
    """ Computes the collection of persistence landscapes associated to a barcode up to index maxind
        using the algorithm set out in Bubenik + Dlotko.
    :param barcode: A barcode object
    :param maxind: The maximum index landscape to calculate
    """
    L = []
    barcode = barcode.expand()
    barcode = barcode.to_array()
    # sort by increasing birth and decreasing death
    sortedbarcode = barcode[np.lexsort((-barcode[:, 1], barcode[:, 0]))]
    k = 1  # initialise index for landscape
    if maxind is None:
        while np.sum(sortedbarcode[:, 2]) > 0:
            p = 0  # pointer to position in barcode
            [b, d, _], sortedbarcode = pop(sortedbarcode, p)
            critical_points = np.array([[float("-inf"), 0], [b, 0], [(b + d) / 2, (d - b) / 2]])
            while critical_points[-1, 0] != float("inf"):  # check last row is not trivial
                if np.shape(sortedbarcode)[0] == 0:
                    critical_points = np.vstack([critical_points, [[d, 0], [float("inf"), 0]]])
                    L.append(Landscape(k, critical_points))
                elif d >= np.max(sortedbarcode[:, 1]):
                    critical_points = np.vstack([critical_points, [[d, 0], [float("inf"), 0]]])
                    L.append(Landscape(k, critical_points))
                else:
                    # find (b',d') the first bar with d'>d
                    p = np.min(np.nonzero(sortedbarcode[:, 1] > d))  # returns min arg of row with death larger than d
                    [bnew, dnew, _], sortedbarcode = pop(sortedbarcode, p)
                    if bnew > d:
                        critical_points = np.vstack([critical_points, [d, 0]])
                    if bnew >= d:
                        critical_points = np.vstack([critical_points, [bnew, 0]])
                    else:
                        critical_points = np.vstack([critical_points, [(bnew + d) / 2, (d - bnew) / 2]])
                        sortedbarcode = np.vstack([sortedbarcode, [bnew, d, 1]])
                        sortedbarcode = sortedbarcode[np.lexsort((-sortedbarcode[:, 1], sortedbarcode[:, 0]))]
                        p += 1
                    critical_points = np.vstack([critical_points, [(bnew + dnew) / 2, (dnew - bnew) / 2]])
                    b, d = [bnew, dnew]
            k += 1

    # add ability to truncate to calculate first K lscapes

    return Landscapes(L)


def pop(array, row):
    poppedrow = array[row, :]
    array = np.delete(array, row, 0)
    return poppedrow, array


def Compute_Rivet(filtered_points, dim=1, resolution=20, RipsMax=1, description='default_description'):
    filename = write_sample(filtered_points, RipsMax, description)
    computed_file_path = rivet.compute_file(filename,
                                            homology=dim,
                                            x=resolution,
                                            y=resolution)
    with open(computed_file_path, 'rb') as f:
        computed_data = f.read()
    return computed_data


# TODO manage directory in which sample is written
def write_sample(filtered_points, RipsMax, description):
    A = filtered_points
    np.savetxt(description + 'RipsMax' + str(RipsMax) + '.txt', A, fmt='%1.4f',
               header='points \n 2 \n ' + str(RipsMax) + ' \n' + description, comments='')
    filename = description + 'RipsMax' + str(RipsMax) + '.txt'
    return filename
