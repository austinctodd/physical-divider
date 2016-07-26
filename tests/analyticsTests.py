#!/usr/bin/env python -tt

from unittest import TestCase
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt

# get the configuration
from sys import path as sPath
from os.path import dirname, join as pJoin
projDir = pJoin(dirname(__file__), "..")
sPath.append(projDir)
import conf

from analytics import movingAvg, inferFuzzyIndicators, getPairs,\
        gaussIntervalProb, convGaussWithPd, UniformIntervalsSet,\
        InfiniteUniformIntervalsSet, indicator, extendForConv


class AnalyticsTests(TestCase):
    def testMovingAvg(self):
        orig = [
                [ 0, 0],
                [ 1, 0],
                [ 2, 0],
                [ 3, 0],
                [ 4, 1],
                [ 5, 1],
                [ 6, 1],
                [ 7, 1],
                [ 8, 0],
                [ 9, 0],
                [10, 0],
                [11, 0],
               ]

        expected = [
                [0.5, 0],
                [1.5, 0],
                [2.5, .2],
                [3.5, .4],
                [4.5, .6],
                [5.5, .8],
                [6.5, .8],
                [7.5, .6],
                [8.5, .4],
                [9.5, .2],
                [10.5, 0],
                [11.5, 0],
                   ]

        mAv, hist = movingAvg(orig, [0, 12], 5, 1)
        mAvRounded = [[p[0], round(float(p[1]), 2)] for p in mAv]
        self.assertEqual(mAvRounded, expected)

        # the extension of the histogram on both sides before using convolution
        # with the option "valid" should reduce boundary effects while not
        # reducing the domain.
        orig = [
                [ 4, 1],
                [ 4, 1],
                [ 5, 1],
                [ 5, 1],
                [ 6, 1],
                [ 6, 1],
                [ 7, 1],
                [ 7, 1],
               ]

        expected = [
                [ 4, 2],
                [ 5, 2],
                [ 6, 2],
                [ 7, 2],
               ]

        mAv, hist = movingAvg(orig, [4, 8], 4, 1)
        mAvRounded = [[p[0], round(float(p[1]), 2)] for p in mAv]
        self.assertEqual(mAvRounded, expected)

        
    def testInferFuzzyIndicators(self):
        fun = lambda x: 1 if 3 < x and x <= 7 else 0
        xs = range(12)
        funXandY = zip(xs, [fun(x) for x in xs])

        self.assertEqual(inferFuzzyIndicators(funXandY),
                         [(3,7)])

        ########################################

        fun = lambda x: 1 if 2 < x and x <= 4 or 6 < x and x <= 8 else 0
        xs = range(12)
        funXandY = zip(xs, [fun(x) for x in xs])

        self.assertEqual(inferFuzzyIndicators(funXandY),
                         [(2, 4), (6, 8)])

        ########################################

        fun = lambda x: 1 if x <= 1 else 0
        xs = range(4)
        funXandY = zip(xs, [fun(x) for x in xs])

        self.assertEqual(inferFuzzyIndicators(funXandY),
                         [(0, 1)])

        ########################################

        fun = lambda x: 0 if x <= 1 else 1
        xs = range(4)
        funXandY = zip(xs, [fun(x) for x in xs])

        self.assertEqual(inferFuzzyIndicators(funXandY),
                         [(1, 3)])

        ########################################

        fun = lambda x: 1 if x <= 1 or 4 < x and x <= 6 else 0
        xs = range(8)
        funXandY = zip(xs, [fun(x) for x in xs])

        self.assertEqual(inferFuzzyIndicators(funXandY),
                         [(0, 1), (4, 6)])

        ########################################

        fun = lambda x: 1 if 4 < x and x <= 6 or 8 < x else 0
        xs = range(12)
        funXandY = zip(xs, [fun(x) for x in xs])

        self.assertEqual(inferFuzzyIndicators(funXandY),
                         [(4, 6), (8, 11)])

        ########################################

        fun = lambda x: 1 if x <= 1 or 4 < x and x <= 6 or 8 < x else 0
        xs = range(12)
        funXandY = zip(xs, [fun(x) for x in xs])

        self.assertEqual(inferFuzzyIndicators(funXandY),
                         [(0, 1), (4, 6), (8, 11)])

    def testGetPairs(self):
        l = range(5)
        ps = getPairs(l)
        self.assertEqual([(0, 1), (1, 2), (2, 3), (3, 4)], ps)
        self.assertEqual([], getPairs(range(1)))
        self.assertEqual([], getPairs([]))

    def testGaussIntervalProb(self):
        self.assertEqual(.68,
                         round(gaussIntervalProb(norm(0, 1), [-1, 1]), 2))

    def testUniformIntervalsSet(self):
        s = UniformIntervalsSet(10, 2, 4, 10)
        self.assertFalse(s.contains(0))
        self.assertTrue(s.contains(10.1))
        self.assertFalse(s.contains(12.1))
        self.assertFalse(s.contains(15.9))
        self.assertTrue(s.contains(16.1))
        self.assertTrue(s.contains(65.9))
        self.assertFalse(s.contains(66.1))

    def testInfinteUniformIntervalsSet(self):
        s = InfiniteUniformIntervalsSet(2, 4, 0)
        self.assertTrue(s.contains(0.1))
        self.assertTrue(s.contains(1.9))
        self.assertFalse(s.contains(2.1))
        self.assertFalse(s.contains(5.9))
        self.assertTrue(s.contains(6.1))
        self.assertTrue(s.contains(7.9))
        self.assertFalse(s.contains(8.1))
        self.assertFalse(s.contains(11.9))

        self.assertFalse(s.contains(-.1))
        self.assertFalse(s.contains(-3.9))
        self.assertTrue(s.contains(-4.1))
        self.assertTrue(s.contains(-5.9))
        self.assertFalse(s.contains(-6.1))
        self.assertFalse(s.contains(-9.9))
        self.assertTrue(s.contains(-10.1))

        s = InfiniteUniformIntervalsSet(2, 4, 1)
        self.assertTrue(s.contains(1.1))
        self.assertTrue(s.contains(2.9))
        self.assertFalse(s.contains(3.1))
        self.assertFalse(s.contains(6.9))
        self.assertTrue(s.contains(7.1))
        self.assertTrue(s.contains(8.9))
        self.assertFalse(s.contains(9.1))
        self.assertFalse(s.contains(12.9))

        self.assertFalse(s.contains(.9))
        self.assertFalse(s.contains(-2.9))
        self.assertTrue(s.contains(-3.1))
        self.assertTrue(s.contains(-4.9))
        self.assertFalse(s.contains(-5.1))
        self.assertFalse(s.contains(-8.9))
        self.assertTrue(s.contains(-9.1))

    def testIndicator(self):
        ind = indicator([
            [0, 1],
            [2, 3],
            [4, 5],
            ])

        for i in range(3):
            self.assertEqual(ind(2*i+.5), 1)
            self.assertEqual(ind(2*i+1.5), 0)

    def testExtendForConv(self):
        ys = np.array([11, 12, 13, 14])
        self.assertTrue(
                all(extendForConv(ys, 4) ==\
                    np.array([12, 11, 11, 12, 13, 14, 14])))
        self.assertTrue(
                all(extendForConv(ys, 5) ==\
                    np.array([13, 12, 11, 12, 13, 14, 13, 12])))

###############################################################################

def testConvGaussianWithPd():
    binCount = 1000
    pdIvMiddle = 5
    totalIv = [0, 10]
    xs = np.linspace(totalIv[0], totalIv[1], binCount)
    sigmas = [.1, .5, 1]

    fig = plt.figure()
    fig.suptitle(r'$\sigma_1 = {}, \sigma_2 = {}, \sigma_3 = {}$'.format(
        sigmas[0], sigmas[1], sigmas[2]))
    for sp, pdIvHalf in zip(range(1, 9), np.linspace(.5, 3, 8)):
        plt.subplot(2, 4, sp)
        pdIv = [pdIvMiddle - pdIvHalf, pdIvMiddle + pdIvHalf]
        plt.title("pDiv = {}".format(np.round(pdIv, 1)))
        truth = [indicator([pdIv])(x) for x in xs]
        xs, ys1 = convGaussWithPd(pdIv, 1, totalIv, binCount)
        xs, ysH = convGaussWithPd(pdIv, .5, totalIv, binCount)
        xs, ysT = convGaussWithPd(pdIv, .1, totalIv, binCount)

        plt.axis([totalIv[0], totalIv[1], -.5, 1.5])
        plt.grid(True)
        plt.plot(xs, ys1, 'r-', xs, ysH, 'r-', xs, ysT, 'r-', xs, truth, 'b-')

    plt.show()


###############################################################################

if __name__ == "__main__":
    testconvGaussianWithPd()

