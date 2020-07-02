import math
import numpy as np
import rivet
import matching_distance


def find_parameters(slopes, offsets, points):
    result = np.zeros(slopes.shape)
    result[slopes == 90] = 1
    result[slopes == 0] = 0

    not_extreme = np.logical_and(slopes != 90, slopes != 0)
    m = np.tan(np.radians(slopes))
    offset_gt_0 = np.logical_and(not_extreme, offsets > 0)

    pts_gt_0 = points[offset_gt_0]

    # y_int = pt[1] - m * pt[0]
    # dist = np.sqrt(pow(pt[1] - y_int, 2) + pow(pt[0], 2))
    # if pt[0] > 0:
    #     return dist
    # else:
    #     return -dist

    y_int = pts_gt_0[:, 1] - m[offset_gt_0] * pts_gt_0[:, 0]
    dist = np.sqrt(np.power(pts_gt_0[:, 1] - y_int, 2) + np.power(pts_gt_0[:, 0], 2))
    dist[pts_gt_0[:, 0] <= 0] *= -1
    result[offset_gt_0] = dist

    # x_int = pt[0] - pt[1] / m
    # dist = np.sqrt(pow(pt[1], 2) + pow(pt[0] - x_int, 2))
    # if pt[1] > 0:
    #     return dist
    # else:
    #     return -dist

    offset_lte_0 = np.logical_and(not_extreme, offsets <= 0)
    pts_lte_0 = points[offset_lte_0]

    x_int = pts_lte_0[:, 0] - pts_lte_0[:, 1] / m[offset_lte_0]
    dist = np.sqrt(np.power(pts_lte_0[:, 1], 2) + np.power(pts_lte_0[:, 0] - x_int, 2))
    dist[pts_lte_0[:, 1] <= 0] *= -1
    result[offset_lte_0] = dist

    return result


def find_parameter_of_point_on_line(sl, offset, pt):
    """Finds the RIVET parameter representation of point on the line
    (sl,offset).  recall that RIVET parameterizes by line length, and takes the
    point where the line intersects the positive x-axis or y-axis to be
    parameterized by 0.  If the line is itself the x-axis or y-axis, then the
    origin is parameterized by 0.  
    
    WARNING: Code assumes that the point lies on the line, and does not check
    this.  Relatedly, the function could be written using only slope or
    offset as input, not both.  """

    if sl == 90:
        return pt[1]

    if sl == 0:
        return pt[0]

    # Otherwise the line is neither horizontal or vertical.
    m = math.tan(math.radians(sl))

    # Find the point on the line parameterized by 0.

    # If offset is positive, this is a point on the y-axis, otherwise, it is
    # a point on the x-axis.

    # Actually, the above is what SHOULD be true, but in the current implementation of RIVET
    # 0 is the point on the line closest to the origin.

    if offset > 0:
        y_int = pt[1] - m * pt[0]
        dist = np.sqrt(pow(pt[1] - y_int, 2) + pow(pt[0], 2))
        if pt[0] > 0:
            return dist
        else:
            return -dist
    else:
        x_int = pt[0] - pt[1] / m
        dist = np.sqrt(pow(pt[1], 2) + pow(pt[0] - x_int, 2))
        if pt[1] > 0:
            return dist
        else:
            return -dist


def slope_offset(a, b):
    """Determine the line containing a and b, in RIVET's (slope,offset) format.
    If a==b, we will just choose the vertical line."""

    vertical = a[:, 0] == b[:, 0]
    slopes = np.zeros(len(a))
    slopes[vertical] = 90
    not_vertical = np.logical_not(vertical)
    slopes[not_vertical] = np.degrees(np.arctan(
        (b[not_vertical, 1] - a[not_vertical, 1]) / (b[not_vertical, 0] - a[not_vertical, 0])))

    # 2.Find the offset
    offsets = matching_distance.find_offsets(slopes, a)
    return slopes, offsets


def barcode_rank(barcode, birth, death):
    """Return the number of bars that are born by 
    `birth` and die after `death`."""
    arr = barcode.to_array()
    if arr.shape[0] == 0:
        return 0
    included = np.logical_and(arr[:, 0] <= birth, arr[:, 1] > death)
    return np.sum(arr[included, 2])


def rank_norm(module1, module2=None, grid_size=20, fixed_bounds=None,
              use_weights=False, normalize=False, minimum_rank=0):
    """If module2==None, approximately computes the approximate (weighted or unweighted)
    L_1-norm of the rank invariant of module1 on a rectangle.  
    
    If module2!=None, computes this for the the difference of the rank
    invariants of module1 and module2.
    
    Note that the rank function is unstable with respect to choice of a,b.
    Because of numerical error, this function can instead return the value of
    the rank functon at points a',b' very close to a and b, which can be
    different.  In a more careful implementation (done by tweaking the innards
    of RIVET) this could be avoided, but shouldn't be a serious issue in our
    intended applications.  

    Input: 
        module1,module2: RIVET "precomputed" representations of
        a persistence module, in Bryn's python bytes format

        grid_size: This is a non-negative integer which should be at least 2.
        We will compute the norm approximately using a grid_size x grid_size
        grid.

        fixed_bound: A rivet.bounds object.  Specifies the rectangle over which
        we compute. If none, the bounds are taken to be the bounds for the
        module provided by RIVET.

        use_weights: Boolean; Should we compute the norm in a weighted fashion,
        so that ranks M(a,b) with a and b lying (close to) a horizontal or
        vertical line are weighted less?  Weights used are the same ones as for
        computing the matching distance.

        normalize: Boolean.  If true, the weights and volume elements are
        chosen as if the bounding rectangle were a rescaled to be a unit
        square.
        
        minimum_rank: Treat all ranks below this value as 0.  [Motivation: For
                hypothesis testing where the hypothesis is of the form: This
                data has at least k topological features.] """

    if fixed_bounds is None:
        # determine bounds from the bounds of the given module(s)
        if module2 is None:
            bounds = rivet.bounds(module1)
        else:
            bounds = matching_distance.common_bounds(
                rivet.bounds(module1), rivet.bounds(module2))
    else:
        bounds = fixed_bounds

    LL = bounds.lower
    UR = bounds.upper

    x_increment = (UR[0] - LL[0]) / grid_size
    y_increment = (UR[1] - LL[1]) / grid_size
    if x_increment == 0 or y_increment == 0:
        raise ValueError('Rectangle is degenerate!  Behavior of the function in this case is not defined.')

    if normalize:
        volume_element = pow(1 / grid_size, 4)
    else:
        # we don't need to define delta_x and delta_y if we aren't normalizing
        volume_element = pow(x_increment * y_increment, 2)

    lows = []
    highs = []
    for x_low in range(grid_size):
        for y_low in range(grid_size):
            for x_high in range(x_low, grid_size):
                for y_high in range(y_low, grid_size):
                    a = [LL[0] + x_low * x_increment, LL[1] + y_low * y_increment]
                    b = [LL[0] + x_high * x_increment, LL[1] + y_high * y_increment]
                    lows.append(a)
                    highs.append(b)
    highs = np.array(highs)
    lows = np.array(lows)

    slopes, offsets = slope_offset(lows, highs)
    if np.any(slopes < 0) or np.any(slopes > 90):
        raise ValueError("Slope out of bounds!")

    weights = np.ones(len(lows))
    if use_weights:
        if normalize:
            delta_x = UR[0] - LL[0]
            delta_y = UR[1] - LL[1]
            weights = matching_distance.calculate_weight(slopes, True, delta_x, delta_y)
        else:
            weights = matching_distance.calculate_weight(slopes)

        # if a and b lie on the same vertical or horizontal line, weight is 0.
        weights[lows[:, 0] == highs[:, 0]] = 0
        weights[lows[:, 1] == highs[:, 1]] = 0

    births = find_parameters(slopes, offsets, np.array(lows))
    deaths = find_parameters(slopes, offsets, np.array(highs))
    birth_deaths = np.c_[births, deaths].tolist()
    slope_offsets = np.c_[slopes, offsets]

    barcodes1 = rivet.barcodes(module1, slope_offsets)

    ranks1 = np.array(
        [barcode_rank(bars, b, d)
         for (_, bars), (b, d) in zip(barcodes1, birth_deaths)])
    ranks1[ranks1 < minimum_rank] = 0

    if module2 is None:
        ranks2 = np.zeros(len(slope_offsets))
    else:
        barcodes2 = rivet.barcodes(module2, slope_offsets)
        ranks2 = np.array([barcode_rank(bars, b, d)
                           for (_, bars), (b, d) in zip(barcodes2, birth_deaths)])
    ranks2[ranks2 < minimum_rank] = 0

    norm = np.sum(np.abs(ranks1 - ranks2) * weights * volume_element)

    return norm


def array_rank_norm(lefts, rights):
    if len(lefts.shape) != 3:
        raise ValueError(
            "`lefts` must have shape (# of barcodes, # of bars in each code, 3)")
    if len(rights.shape) != 3:
        raise ValueError(
            "`rights` must have shape (# of barcodes, # of bars in each code, 3)")
    if lefts.shape[0] != rights.shape[0]:
        raise ValueError(
            "First dimension of both arrays must have the same length")
    results = []
    for i in range(lefts.shape[0]):
        left = lefts[i]
        right = rights[i]
        results.append(rank_norm(left, right))
    return np.array(results)
