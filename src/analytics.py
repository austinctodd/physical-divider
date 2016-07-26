#!/usr/bin/env python -tt

import numpy as np
from scipy.stats import norm
from math import modf, floor, ceil

def extendForConv(ys, bandwidth):
    """The convolve function of numpy can produce unwanted effects at the
    boundaries. To prevent this, one should call convolve with the option
    "valid" and extend the function to be smoothed on both sides by half of the
    bandwidth. Here this extension is done via mirroring at the boundary.
    ys is presumed to be a numpy array.
    """
    bw = int(bandwidth)
    bwH = bw/2
    if bw%2 == 0:
        # even
        leftExtended = np.append(ys[bwH-1::-1], ys)
        bothExtended = np.append(leftExtended, ys[:-bwH:-1])
    else:
        # odd
        leftExtended = np.append(ys[bwH:0:-1], ys)
        bothExtended = np.append(leftExtended, ys[-2:-bwH-2:-1])
    return bothExtended


def movingAvgWoExtDensityHist(funXandY, domain, bandwidth, binLen):
    return movingAvgWoExt(funXandY, domain, bandwidth, binLen, True)


def movingAvgWoExt(funXandY, domain, bandwidth, binLen, density = False):
    """
    Here funXandY are the (x,y) pairs of a function which is presumed to have
    as y-values just 0 or 1. The x-values with value 1 are combined into bins
    of a histogram which then is smoothed with a moving average.
    """
    bw = int(bandwidth)
    hist, binEdges = np.histogram(
            a = [xy[0] for xy in funXandY if xy[1] > 0],
            bins = (domain[1] - domain[0])/binLen,
            range = domain,
            density = density)
    mvAvg = np.array([1 for i in range(bw)])/float(bw)
    conv = np.convolve(hist, mvAvg, 'same')

    off = .5 if bw%2 == 1 else 0
    return zip(binEdges[:-1] + off, conv), zip(binEdges[:-1] + off, hist)


def movingAvg(funXandY, domain, bandwidth, binLen):
    """
    Here funXandY are the (x,y) pairs of a function which is presumed to have
    as y-values just 0 or 1. The x-values with value 1 are combined into bins
    of a histogram which then is smoothed with a moving average.
    """
    bw = int(bandwidth)
    hist, binEdges = np.histogram(
            [xy[0] for xy in funXandY if xy[1] > 0],
            (domain[1] - domain[0])/binLen,
            domain)
    mvAvg = np.array([1 for i in range(bw)])/float(bw)
    histExt = extendForConv(hist, bw)
    conv = np.convolve(histExt, mvAvg, 'valid')

    off = .5 if bw%2 == 1 else 0
    return zip(binEdges[:-1] + off, conv), zip(binEdges[:-1] + off, hist)


def getPairs(l):
    return zip(l[:-1], l[1:])


def inferFuzzyIndicators(funXandY):
    xStart = min(funXandY, key = lambda p: p[0])
    xEnd   = max(funXandY, key = lambda p: p[0])
    maxY = max(funXandY, key = lambda p: p[1])
    minY = min(funXandY, key = lambda p: p[1])

    threshold = (maxY[1] + minY[1])/2.0

    def intersects(p):
        if p[0][1] <= threshold and threshold < p[1][1]:
            return p[0], 1
        elif p[0][1] > threshold and threshold >= p[1][1]:
            return p[0], -1
        else:
            return p[0], 0

    isecs = filter(lambda p: p[1] != 0,
                           map(intersects,
                               getPairs(funXandY)))

    if len(isecs) == 0:
        return []

    optionalFirst = None
    if isecs[0][1] == -1:
        optionalFirst = (xStart[0], isecs[0][0][0])
        isecs = isecs[1:]
        if len(isecs) == 0:
            return [optionalFirst]

    optionalLast = None
    if isecs[len(isecs)-1][1] == 1:
        optionalLast = (isecs[len(isecs)-1][0][0], xEnd[0])
        isecs = isecs[:-1]
        if len(isecs) == 0:
            if optionalFirst:
                return [optionalFirst, optionalLast]
            else:
                return [optionalLast]

    inds = getPairs([p[0][0] for p in isecs])[::2]
    if optionalFirst:
        inds = [optionalFirst] + inds
    if optionalLast:
        inds = inds + [optionalLast]

    return inds


def inferFuzzyIndicatorsThresholdOneHalf(funXandY):
    xStart = min(funXandY, key = lambda p: p[0])
    xEnd   = max(funXandY, key = lambda p: p[0])
    maxY = max(funXandY, key = lambda p: p[1])

    threshold = maxY[1]/2.0

    def intersects(p):
        if p[0][1] <= threshold and threshold < p[1][1]:
            return p[0], 1
        elif p[0][1] > threshold and threshold >= p[1][1]:
            return p[0], -1
        else:
            return p[0], 0

    isecs = filter(lambda p: p[1] != 0,
                           map(intersects,
                               getPairs(funXandY)))

    if len(isecs) == 0:
        return []

    optionalFirst = None
    if isecs[0][1] == -1:
        optionalFirst = (xStart[0], isecs[0][0][0])
        isecs = isecs[1:]
        if len(isecs) == 0:
            return [optionalFirst]

    optionalLast = None
    if isecs[len(isecs)-1][1] == 1:
        optionalLast = (isecs[len(isecs)-1][0][0], xEnd[0])
        isecs = isecs[:-1]
        if len(isecs) == 0:
            if optionalFirst:
                return [optionalFirst, optionalLast]
            else:
                return [optionalLast]

    inds = getPairs([p[0][0] for p in isecs])[::2]
    if optionalFirst:
        inds = [optionalFirst] + inds
    if optionalLast:
        inds = inds + [optionalLast]

    return inds


def gaussIntervalProb(gaussian, interval):
    return gaussian.cdf(interval[1]) - gaussian.cdf(interval[0])


def indicator(listOfIvs):
    def ind(x):
        if any([p[0] < x and x <= p[1] for p in listOfIvs]):
            return 1
        else:
            return 0

    return ind


def convGaussWithPd(supportIv, sigma, totalIv, binCount):
    xs = np.linspace(totalIv[0], totalIv[1], binCount)
    ys = [gaussIntervalProb(norm(x, sigma), supportIv) for x in xs]
    return xs, ys


class InfiniteUniformIntervalsSet(object):
    """A set given by infinitly many consecutive intervals. The intervals are
    all of the same length ivLen and the distance between two back-to-back
    intervals gapLen is constant, too. The intervals are closed. One interval
    starts at offset.
    """
    def __init__(self, ivLen, gapLen, offset):
        self.ivLen = ivLen
        self.gapLen = gapLen
        self.offset = offset
        self.period = float(self.ivLen + self.gapLen)
        self.ratio = self.ivLen / self.period

    def contains(self, x):
        normalizedX = (x - self.offset) / self.period
        frac = normalizedX - floor(normalizedX)
        if frac > self.ratio:
            return False
        return True
        

class UniformIntervalsSet(object):
    """A set given by a set of count consecutive intervals. The intervals are
    all of the same length ivLen and the distance between two back-to-back
    intervals gapLen is constant, too. The intervals are closed.
    """
    def __init__(self, start, ivLen, gapLen, count):
        self.start = start
        self.ivLen = ivLen
        self.gapLen = gapLen
        self.count = count
        self.end = self.start + self.count * (self.ivLen + self.gapLen)
        self.period = float(self.ivLen + self.gapLen)
        self.ratio = self.ivLen / self.period

    def contains(self, x):
        if x < self.start or self.end < x:
            return False
        fracPart, intPart = modf((x-self.start)/self.period)
        if fracPart > self.ratio:
            return False
        return True
        
