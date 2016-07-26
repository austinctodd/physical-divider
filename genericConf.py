#!/usr/bin/env python

# Every script that uses some user adjustable configuration variables should
# import this configuration module.

from sys import path as sPath
from os.path import dirname, join as pJoin

###############################################################################
#this is site specific configuration that should be customized for each site.

###############################################################################
#this is general configuration which should be the same for all sites.

projDir = dirname(__file__)
srcDir = pJoin(projDir, "src")
testsDir = pJoin(projDir, "tests")

# here we extend the system path
sPath.append(srcDir)
sPath.append(testsDir)
