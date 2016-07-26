#!/usr/bin/env python

# this script will run all the tests

# get the configuration
from sys import path as sPath
from os.path import dirname, join as pJoin
projDir = pJoin(dirname(__file__), "..")
sPath.append(projDir)
import conf

from unittest import TestLoader, TestSuite, TextTestRunner
from simulationTests import CarTests
from simulationTests import FunToolsTests
from simulationTests import FunItersTests
from simulationTests import OneLaneSimulationTests
from analyticsTests import AnalyticsTests

CarTestSuite = TestLoader().loadTestsFromTestCase(CarTests)
FunToolsTestSuite = TestLoader().loadTestsFromTestCase(FunToolsTests)
FunItersTestSuite = TestLoader().loadTestsFromTestCase(FunItersTests)
OlsTestSuite = TestLoader().loadTestsFromTestCase(OneLaneSimulationTests)
AnalyticsTestSuite = TestLoader().loadTestsFromTestCase(AnalyticsTests)

allTests = TestSuite([
                      CarTestSuite,
                      FunToolsTestSuite,
                      FunItersTestSuite,
                      OlsTestSuite,
                      AnalyticsTestSuite,
                     ])

TextTestRunner(verbosity=1).run(allTests)
