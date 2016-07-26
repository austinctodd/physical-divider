#!/usr/bin/env python -tt

from unittest import TestCase
from math import sin, pi

from matplotlib import pyplot as plt


# get the configuration
from sys import path as sPath
from os.path import dirname, join as pJoin
projDir = pJoin(dirname(__file__), "..")
sPath.append(projDir)
import conf

from simulation import FunTools, Car, FunIterEqualIntervals,\
        FunIterBounded, FunIterInfinite, MarkovChain, OneLaneSimulation,\
        PdfGivenTrueVal, BoundedMarkovChain
from analytics import indicator

class FunToolsTests(TestCase):
    def testGetValsForInterval(self):
        self.assertEqual(
            FunTools.getValsForInterval(lambda x: 42, [0, 1], 3),
            [42, 42, 42])
        self.assertEqual(
            FunTools.getValsForInterval(lambda x: 2*x + 42, [0, 2], 3),
            [42, 44, 46])

    def testGetXsAndValsForInterval(self):
        xs, vals = FunTools.getXsAndValsForInterval(
                       lambda x: x*0 + 42, [0, 1], 3)
        self.assertTrue((xs == [0, .5, 1]).all())
        self.assertTrue((vals == [42, 42, 42]).all())


class FunItersTests(TestCase):
    def testFunIterEqualIntervals(self):
        self.assertEqual([0, 1, 0, -1, 0],
                         [round(x) for x in\
                              FunIterEqualIntervals(sin, [0, 2*pi], 5)])


    def testFunIterBounded(self):
        it = FunIterBounded(lambda x: x, 0, 2, .5)
        self.assertEqual([0, .5, 1, 1.5, 2],
                         [y for y in it])
        it2 = FunIterBounded(lambda x: x, 0, .5, 1)
        self.assertEqual([0],
                         [y for y in it2])

    def testFunIterInfinite(self):
        it = FunIterInfinite(lambda x: x, 0, 1)
        col = []
        for i in it:
            if i >= 1000:
                break
            col.append(i)
        self.assertEqual(range(1000), col)


class CarTests(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIter(self):
        c = Car(lambda x: 2, [0, 5], 0, 1, None)
        self.assertEqual(
                [p for p in c],
                [(0, 0), (1, 2), (2, 4)])


class OneLaneSimulationTests(TestCase):
    def testCarSensesPd(self):
        c = Car(lambda x: 2, [0, 5], 0, 1, PdfGivenTrueVal.binomial(.9))
        phD = indicator([[1, 2], [3,4]])
        ols = OneLaneSimulation([c], phD, 1)
        sd = ols.carSensesPd(c)
        self.assertEqual(len(sd), 3)
        self.assertTrue(all([0 <= i[1] and i[1] <= 1 for i in sd]))

    def testSimulate(self):
        # see the module physical-divider.py for how to simulate and plot
        pass

###############################################################################

def plotMarkovChains():
    mch = MarkovChain(10, 1)
    ys = [mch.next() for i in range(1000)]
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(ys, 'b-')

    mch = BoundedMarkovChain(10, 1, 5, 15)
    ys = [mch.next() for i in range(1000)]
    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.ylim([0, 20])
    plt.axhline(y = 5, color = 'red')
    plt.axhline(y = 15, color = 'red')
    plt.plot(ys, 'b-')
    plt.show()

###############################################################################

if __name__ == "__main__":
    plotMarkovChains()
