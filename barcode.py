import numpy as np

"""
Standard bar code classes that can be shared among
the wrappers for multiple tools
"""


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
