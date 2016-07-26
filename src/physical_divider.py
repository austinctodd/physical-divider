#!/usr/bin/env python -tt

from math import sin, pi
from time import time
import numpy as np
from matplotlib import pyplot as plt, patches
from scipy.fftpack import fft
from scipy.stats import beta
from scipy.stats import norm
import pandas as pd

# get the configuration
from sys import path as sPath
from os.path import dirname, join as pJoin
projDir = pJoin(dirname(__file__), "..")
sPath.append(projDir)
import conf

from analytics import movingAvg, movingAvgWoExt, inferFuzzyIndicators, \
        inferFuzzyIndicatorsThresholdOneHalf, InfiniteUniformIntervalsSet, \
        indicator, convGaussWithPd, movingAvgWoExtDensityHist
from simulation import OneLaneSimulation, Car, PdfGivenTrueVal, \
        SecondLaneSimulation, MarkovChain, BoundedMarkovChain


def visualizeInfluenceOfSensorError():
    # units: the units of length and time are arbitrary, but must of course be
    # consistent, so, once you choose e.g. meter and seconds, stick with it.

    # Define the section of road
    road = [0, 1000]

    # Define the locations of physical dividers as intervals
    pDiv=[[300, 700]]
    physDiv = indicator(pDiv)

    # Minimum GPS error
    gpsSigma = 50

    # Moving average length (i.e. the number of bins used) and the bin length
    convLen = 1000
    convBinLen = .1

    # Information for histogram plot
    histBinCount = (road[1] - road[0])/convBinLen
    normed = True
    histcolor = "#3F5D7D"

    # Start time of car trace, time steps in car trace
    timeStart = 0
    timeDelta = 1

    # Sensor accuracy (fractional percentage)
    binomialP = [.99, .8]

    # this is the extension of the road to diminish the boundary effects
    ext = max(2*4*gpsSigma, convLen*convBinLen/2.0)
    extRoad = [road[0] - ext, road[1] + ext]

    # number of cars used for simulation
    carCount = 5000

    spTitles = ["threshold max/2", "threshold max/2", "threshold (max+min)/2"]
    fig = plt.figure(figsize = (12, 5))
    fig.suptitle("bandwidth = {}, conv. bin length: {}".\
                     format(convBinLen*convLen, convBinLen),
                 fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = .85)

    for spCol in (1, 2, 3):
        #cars = map(lambda i: Car(lambda s:\
        #                             .5*sin(float(i)/(carCount-1)*2*pi)+1,
        #                         extRoad,
        #                         timeStart,
        #                         timeDelta,
        #                         PdfGivenTrueVal.binomial(binomialP[spCol-1])),
        #           range(carCount))
        if spCol != 3:
            cars = map(lambda i: Car(BoundedMarkovChain(
                                         50/3.6,.1/3.6,30/3.6,65/3.6),
                                     extRoad,
                                     timeStart,
                                     timeDelta,
                                     PdfGivenTrueVal.binomial(binomialP[spCol-1])),
                       range(carCount))
            sim = OneLaneSimulation(cars,
                                    physDiv,
                                    gpsSigma)
            sd = sim.getSimulationData()

            mAvExt, sdHistExt = movingAvgWoExt(sd, extRoad, convLen, convBinLen)
            mAv = [i for i in mAvExt if road[0] <= i[0] and i[0] <= road[1]]
            mAvx, mAvy = zip(*mAv)

        if spCol != 3:
            inds = inferFuzzyIndicatorsThresholdOneHalf(mAv)
        else:
            inds = inferFuzzyIndicators(mAv)

        # Remove the plot frame lines. They are unnecessary chartjunk.
        ax = plt.subplot(1, 3, spCol)
        plt.title(spTitles[spCol-1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
        plt.tick_params(axis="both", which="both", bottom="on", top="off",
                        labelbottom="on", left="off", right="off", labelleft="on")

        # Ensure that the axis ticks only show up on the bottom and left of the plot.
        # Ticks on the right and top of the plot are generally unnecessary chartjunk.
        ax.get_xaxis().tick_bottom()

    
        # Plot histogram of physical divider probability 
        plt.hist([p[0] for p in sd if p[1] > 0],
                 bins=np.linspace(road[0], road[1], histBinCount),
                 histtype='stepfilled',
                 normed=normed,
                 color=histcolor,
                 linewidth=0 ,
                 zorder=4)

        # Plot the moving average of the histogram
        plt.plot(mAvx, mAvy/(sum(mAvy)*convBinLen),color='k',lw=1,zorder=5)

        # Plot lines that indicate where the line crosses 1/2 the max of the moving average
        # and add text above that indicates the distance from the actual divider
        for ind in range(0,len(inds)):
            plt.plot([inds[ind][0],inds[ind][0]],
                     [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                     color = (44/ 255., 160/ 255., 44/ 255.),
                     lw=1.5,linestyle='-',zorder=5)
            plt.text(inds[ind][0],1.28/sum([i[1]-i[0] for i in pDiv]),
                     '%4.2f'%(inds[ind][0]-pDiv[ind][0]),
                      verticalalignment='bottom',
                      horizontalalignment='center',
                      color=(44/ 255., 160/ 255., 44/ 255.), 
                      fontsize=8)
            plt.plot([inds[ind][1],inds[ind][1]],
                     [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                     color = (214/ 255., 39/ 255., 40/ 255.),
                     lw=1.5,linestyle='-',zorder=5)
            plt.text(inds[ind][1],1.28/sum([i[1]-i[0] for i in pDiv]),
                     '%4.2f'%(inds[ind][1]-pDiv[ind][1]),
                      verticalalignment='bottom',
                      horizontalalignment='center',
                      color=(214/ 255., 39/ 255., 40/ 255.), 
                      fontsize=8)

        # Plot simulation specifications
        plt.text((road[1]-road[0])/100,1.45/sum([i[1]-i[0] for i in pDiv]),
                 '$\sigma$(GPS) = %4.1f m' % (gpsSigma),
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 color='k', 
                 fontsize=10)
        plt.text((road[1]-road[0]),1.45/sum([i[1]-i[0] for i in pDiv]),
                 '$\sigma$(sensor) = %4.1f%%' % \
                         ((1.0-binomialP[min(1, spCol-1)])*100),
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 color='k', 
                 fontsize=10)
        plt.text((road[1]-road[0])/100,1.625/sum([i[1]-i[0] for i in pDiv]),
                 'n = %d' % (carCount),
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 color='k', 
                 fontsize=10)

        # Add patch in background to indicate the actual physical divider
        for ind in pDiv:
            ax.add_patch(patches.Rectangle((ind[0],0),   # (x,y)
                                           (ind[1]-ind[0]),          # width
                                           1.25/sum([i[1]-i[0] for i in pDiv]), # height
                         facecolor=(0.7,0.7,0.7),lw=0,
                         zorder=3))
                          
        # Only print x label if the bottom row
        xticks=np.linspace(road[0],road[1],6,dtype=int)                        
        plt.xlabel("Along-track distance (m)", fontsize=14,fontname='Helvetica')
        plt.xticks(xticks, [str(x) for x in xticks], fontsize=10)    
   
        # Provide tick lines across the plot to help your viewers trace along    
        # the axis ticks. Make sure that the lines are light and small so they    
        # don't obscure the primary data lines.    
        yticks=np.arange((1.5/sum([i[1]-i[0] for i in pDiv]))/5.0,
                       1.5/sum([i[1]-i[0] for i in pDiv]),
                      (1.5/sum([i[1]-i[0] for i in pDiv]))/5.0)

        # Only print y label if the first column
        for y in yticks:
            plt.plot([road[0], road[1]], [y,y], ":", lw=0.75, color="black", alpha=0.6)    
        if (spCol==1): 
            plt.yticks(yticks, [str(x) for x in yticks], fontsize=12)    
            plt.ylabel("Probability", fontsize=14,fontname='Helvetica')
        else: 
            plt.yticks([])    

        # Set axis properties
        plt.axis([road[0], road[1], 0, 1.75/sum([i[1]-i[0] for i in pDiv])])

    plt.show()


def visualizeBandwidthEffect():
    # units: the units of length and time are arbitrary, but must of course be
    # consistent, so, once you choose e.g. meter and seconds, stick with it.
    
    # Define the section of road
    road = [0, 400]

    # Define the locations of physical dividers as intervals
    #pDiv = [[500,900],[1100,1500]]
    #pDiv = [[500,950],[1050,1500]]
    #pDiv = [[500,970],[1030,1500]]
    #pDiv = [[500,990],[1010,1500]]
    pDiv = [[100,190],[210,300]]
    physDiv = indicator(pDiv)

    # Minimum GPS error
    gpsSigma = 5

    # Moving average length (i.e. the number of bins used) and the bin length
    convLens = [50, 100, 200, 300, 400]
    convBinLen = .1

    # Information for histogram plot
    histBinCount = (road[1] - road[0])/convBinLen
    normed = True
    histcolor = "#3F5D7D"

    # Start time of car trace, time steps in car trace
    timeStart = 0
    timeDelta = .1

    # Sensor accuracy (fractional percentage)
    binomialP = .9

    # this is the extension of the road to diminish the boundary effects
    ext = max(2*4*gpsSigma, max(convLens)*convBinLen/2.0)
    extRoad = [road[0] - ext, road[1] + ext]

    # some subplots (sp: SubPlot)
    fig = plt.figure()
    fig.suptitle("physDivs: {}, gps err: {}, conv. bin len = {}".format(
                                        pDiv, gpsSigma, convBinLen),
                 fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = .9, left = .05, right = .95)
    spRows = (1, 2, 3)
    startTime = time()
    for spRow in spRows:
        carCount = 5 * 10**(spRow)

        #cars = map(lambda i: Car(lambda s:\
        #                             .5*sin(float(i)/(carCount-1)*2*pi)+1,
        #                         extRoad,
        #                         timeStart,
        #                         timeDelta,
        #                         PdfGivenTrueVal.binomial(binomialP)),
        #           range(carCount))
        cars = map(lambda i: Car(BoundedMarkovChain(
                                     50/3.6,.1/3.6,30/3.6,65/3.6),
                                 extRoad,
                                 timeStart,
                                 timeDelta,
                                 PdfGivenTrueVal.binomial(binomialP)),
                   range(carCount))
        sim = OneLaneSimulation(cars,
                                physDiv,
                                gpsSigma)
        sd = sim.getSimulationData()

        spCols = (1, 2, 3, 4, 5)
        for spCol in spCols:
            mAvExt, sdHistExt = movingAvgWoExt(
                    sd, extRoad, convLens[spCol-1], convBinLen)
            mAv = [i for i in mAvExt if road[0] <= i[0] and i[0] <= road[1]]

            inds = inferFuzzyIndicators(mAv)

            # Remove the plot frame lines. They are unnecessary chartjunk.
            ax = plt.subplot(max(spRows), max(spCols), max(spCols)*(spRow-1) + spCol)
            ax.spines["top"].set_visible(False)    
            ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(False)    

            # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
            plt.tick_params(axis="both", which="both", bottom="on", top="off",    
                            labelbottom="on", left="off", right="off", labelleft="on")    

            # Ensure that the axis ticks only show up on the bottom and left of the plot.    
            # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
            ax.get_xaxis().tick_bottom()    

            # Plot histogram of physical divider probability 
            sightingSites = [p[0] for p in sd if p[1] > 0]
            print "len(sd): {}".format(len(sd))
            print "spCol: {}".format(spCol)
            print "sightingSites: {}".format(sightingSites[:10])
            plt.hist(sightingSites,
                     bins=np.linspace(road[0], road[1], histBinCount),
                     histtype='stepfilled',
                     normed=normed,
                     color=histcolor,
                     linewidth=0 ,
                     zorder=4)

            # Plot the moving average of the histogram
            mAvx, mAvy = zip(*mAv)
            plt.plot(mAvx, mAvy/(sum(mAvy)*convBinLen),color='k',lw=1,zorder=5)

            # Plot lines that indicate where the line crosses 1/2 the max of the moving average
            # and add text above that indicates the distance from the actual divider
            if len(inds) != len(pDiv):
                print "WARNING: there are {} pds but {} had been detected".format(len(pDiv), len(inds))
            else:
                for ind in range(0,len(inds)):
                    plt.plot([inds[ind][0],inds[ind][0]],
                             [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                             color = (44/ 255., 160/ 255., 44/ 255.),
                             lw=1.5,linestyle='-',zorder=5)
                    plt.text(inds[ind][0],1.28/sum([i[1]-i[0] for i in pDiv]),
                             '%4.2f'%(inds[ind][0]-pDiv[ind][0]),
                              verticalalignment='bottom',
                              horizontalalignment='left',
                              color=(44/ 255., 160/ 255., 44/ 255.), 
                              fontsize=8)
                    plt.plot([inds[ind][1],inds[ind][1]],
                             [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                             color = (214/ 255., 39/ 255., 40/ 255.),
                             lw=1.5,linestyle='-',zorder=5)
                    plt.text(inds[ind][1],1.28/sum([i[1]-i[0] for i in pDiv]),
                             '%4.2f'%(inds[ind][1]-pDiv[ind][1]),
                              verticalalignment='bottom',
                              horizontalalignment='right',
                              color=(214/ 255., 39/ 255., 40/ 255.), 
                              fontsize=8)

            # Plot simulation specifications
            plt.text((road[1]-road[0])/100,1.45/sum([i[1]-i[0] for i in pDiv]),
                     'bw = %4.1f m' % (convLens[spCol-1]*convBinLen),
                     verticalalignment='bottom',
                     horizontalalignment='left',
                     color='k', 
                     fontsize=10)
            plt.text((road[1]-road[0]),1.45/sum([i[1]-i[0] for i in pDiv]),
                     '$\sigma$(sensor) = %4.1f%%' % ((1.0-binomialP)*100),
                     verticalalignment='bottom',
                     horizontalalignment='right',
                     color='k', 
                     fontsize=10)
            plt.text((road[1]-road[0])/100,1.625/sum([i[1]-i[0] for i in pDiv]),
                     'n = %d' % (carCount),
                     verticalalignment='bottom',
                     horizontalalignment='left',
                     color='k', 
                     fontsize=10)
 
            # Add patch in background to indicate the actual physical divider
            for ind in pDiv:
                ax.add_patch(patches.Rectangle((ind[0],0),   # (x,y)
                                               (ind[1]-ind[0]),          # width
                                               1.25/sum([i[1]-i[0] for i in pDiv]), # height
                             facecolor=(0.7,0.7,0.7),lw=0,
                             zorder=3))
                              
            # Only print x label if the bottom row
            xticks=np.linspace(road[0],road[1],6,dtype=int)                        
            if (spRow==max(spRows)): 
                plt.xlabel("Along-track distance (m)", fontsize=14,fontname='Helvetica')
            plt.xticks(xticks, [str(x) for x in xticks], fontsize=10)    
       
            # Provide tick lines across the plot to help your viewers trace along    
            # the axis ticks. Make sure that the lines are light and small so they    
            # don't obscure the primary data lines.    
            yticks=np.round(np.arange((1.5/sum([i[1]-i[0] for i in pDiv]))/5.0,
                                       1.5/sum([i[1]-i[0] for i in pDiv]),
                                      (1.5/sum([i[1]-i[0] for i in pDiv]))/5.0),
                            4)
 
            # Only print y label if the first column
            for y in yticks:
                plt.plot([road[0], road[1]], [y,y], ":", lw=0.75, color="black", alpha=0.6)    
            if (spCol==1): 
                plt.yticks(yticks, [str(x) for x in yticks], fontsize=12)    
                plt.ylabel("Probability", fontsize=14,fontname='Helvetica')
            else: 
                plt.yticks([])    
 
            # Set axis properties
            plt.axis([road[0], road[1], 0, 1.75/sum([i[1]-i[0] for i in pDiv])])

    endTime = time()
    print "computation time: {}".format(endTime - startTime)
    plt.show()


def visualizeSimulation():
    road = [0, 400]
    pDiv = [[50,150],[250,350]]
    binomialP = .95
    carCounts = (100, 300, 500)
    gpsSigmas = (5, 10, 15)
    convLen = 60
    convBinLen = .1
    visSim(road, pDiv, binomialP, carCounts, gpsSigmas, convBinLen, convLen)


def visSim(road, pDiv, binomialP, carCounts, gpsSigmas, convBinLen, convLen):
    """
    road: is a list of pairs that discribe the support of the physical divider
    pDiv: Define the locations of physical dividers as intervals
    binomialP: sensor accuracy
    carCounts: is a list of number of cars that are used for the simulation.
    gpsSigmas: is a list of gps errors
    convBinLen: the length of one bin used for discretizing the data to use
                discrete moving average
    convLen: this is the number of bins that defines the bandwidth of the
             moving average
    """
    # units: the units of length and time are arbitrary, but must of course be
    # consistent, so, once you choose e.g. meter and seconds, stick with it.
    
    physDiv = indicator(pDiv)

    # Moving average length (i.e. the number of bins used) and the bin length
    convLen = 50
    convBinLen = .1

    # Information for histogram plot
    histBinCount = (road[1] - road[0])/convBinLen
    normed = True
    histcolor = "#3F5D7D"

    # Start time of car trace, time steps in car trace
    timeStart = 0
    timeDelta = .1

    # this is the extension of the road to diminish the boundary effects
    ext = max(3*max(gpsSigmas), convLen*convBinLen/2.0)
    extRoad = [road[0] - ext, road[1] + ext]

    # some subplots (sp: SubPlot)
    #fig = plt.figure()
    fig = plt.figure(figsize = (20, 10))
    fig.suptitle("physDivs: {}, conv. bw. = {}, conv. bin len = {}".format(
                                        pDiv, convLen*convBinLen, convBinLen),
                 fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = .9, left = .05, right = .95)
    spRows = range(1, len(carCounts)+1)
    spCols = range(1, len(gpsSigmas)+1)
    startTime = time()
    for spRow in spRows:
        #carCount = 5 * 10**(spRow)
        carCount = carCounts[spRow-1]

        print "carCount = {}".format(carCount)
        for spCol in spCols:
            #cars = map(lambda i: Car(lambda s:\
            #                             .5*sin(float(i)/(carCount-1)*2*pi)+1,
            #                         extRoad,
            #                         timeStart,
            #                         timeDelta,
            #                         PdfGivenTrueVal.binomial(binomialP)),
            #           range(carCount))
            cars = map(lambda i: Car(BoundedMarkovChain(
                                         50/3.6,.1/3.6,30/3.6,65/3.6),
                                     extRoad,
                                     timeStart,
                                     timeDelta,
                                     PdfGivenTrueVal.binomial(binomialP)),
                       range(carCount))
            sim = OneLaneSimulation(cars,
                                    physDiv,
                                    gpsSigmas[spCol-1])
            sd = sim.getSimulationData()

            mAvExt, sdHistExt = movingAvgWoExt(sd, extRoad, convLen, convBinLen)
            mAv = [i for i in mAvExt if road[0] <= i[0] and i[0] <= road[1]]

            inds = inferFuzzyIndicators(mAv)

            # Remove the plot frame lines. They are unnecessary chartjunk.
            ax = plt.subplot(max(spRows), 4, 4*(spRow-1) + spCol)
            ax.spines["top"].set_visible(False)    
            ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(False)    

            # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
            plt.tick_params(axis="both", which="both", bottom="on", top="off",    
                            labelbottom="on", left="off", right="off", labelleft="on")    

            # Ensure that the axis ticks only show up on the bottom and left of the plot.    
            # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
            ax.get_xaxis().tick_bottom()    

            # Plot histogram of physical divider probability 
            sightingSites = [p[0] for p in sd if p[1] > 0]
            #print "len(sd): {}".format(len(sd))
            #print "spCol: {}".format(spCol)
            #print "sightingSites: {}".format(sightingSites[:10])
            plt.hist(sightingSites,
                     bins=np.linspace(road[0], road[1], histBinCount),
                     histtype='stepfilled',
                     normed=normed,
                     color=histcolor,
                     linewidth=0 ,
                     zorder=4)

            # Plot the moving average of the histogram
            mAvx, mAvy = zip(*mAv)
            plt.plot(mAvx, mAvy/(sum(mAvy)*convBinLen),color='k',lw=1,zorder=5)

            # Plot lines that indicate where the line crosses 1/2 the max of the moving average
            # and add text above that indicates the distance from the actual divider
            if len(inds) != len(pDiv):
                print "WARNING: there are {} pds but {} had been detected".format(len(pDiv), len(inds))
            else:
                for ind in range(0,len(inds)):
                    plt.plot([inds[ind][0],inds[ind][0]],
                             [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                             color = (44/ 255., 160/ 255., 44/ 255.),
                             lw=1.5,linestyle='-',zorder=5)
                    plt.text(inds[ind][0],1.28/sum([i[1]-i[0] for i in pDiv]),
                             '%4.2f'%(inds[ind][0]-pDiv[ind][0]),
                              verticalalignment='bottom',
                              horizontalalignment='left',
                              color=(44/ 255., 160/ 255., 44/ 255.), 
                              fontsize=8)
                    plt.plot([inds[ind][1],inds[ind][1]],
                             [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                             color = (214/ 255., 39/ 255., 40/ 255.),
                             lw=1.5,linestyle='-',zorder=5)
                    plt.text(inds[ind][1],1.28/sum([i[1]-i[0] for i in pDiv]),
                             '%4.2f'%(inds[ind][1]-pDiv[ind][1]),
                              verticalalignment='bottom',
                              horizontalalignment='right',
                              color=(214/ 255., 39/ 255., 40/ 255.), 
                              fontsize=8)

            # Plot simulation specifications
            plt.text((road[1]-road[0])/100,1.45/sum([i[1]-i[0] for i in pDiv]),
                     '$\sigma$(GPS) = %4.1f m' % (gpsSigmas[spCol-1]),
                     verticalalignment='bottom',
                     horizontalalignment='left',
                     color='k', 
                     fontsize=10)
            plt.text((road[1]-road[0]),1.45/sum([i[1]-i[0] for i in pDiv]),
                     '$\sigma$(sensor) = %4.1f%%' % ((1.0-binomialP)*100),
                     verticalalignment='bottom',
                     horizontalalignment='right',
                     color='k', 
                     fontsize=10)
            plt.text((road[1]-road[0])/100,1.625/sum([i[1]-i[0] for i in pDiv]),
                     'n = %d' % (carCount),
                     verticalalignment='bottom',
                     horizontalalignment='left',
                     color='k', 
                     fontsize=10)
 
            # Add patch in background to indicate the actual physical divider
            for ind in pDiv:
                ax.add_patch(patches.Rectangle((ind[0],0),   # (x,y)
                                               (ind[1]-ind[0]),          # width
                                               1.25/sum([i[1]-i[0] for i in pDiv]), # height
                             facecolor=(0.7,0.7,0.7),lw=0,
                             zorder=3))
                              
            # Only print x label if the bottom row
            xticks=np.linspace(road[0],road[1],6,dtype=int)                        
            if (spRow==max(spRows)): 
                plt.xlabel("Along-track distance (m)", fontsize=14,fontname='Helvetica')
            plt.xticks(xticks, [str(x) for x in xticks], fontsize=10)    
       
            # Provide tick lines across the plot to help your viewers trace along    
            # the axis ticks. Make sure that the lines are light and small so they    
            # don't obscure the primary data lines.    
            yticks=np.round(np.arange((1.5/sum([i[1]-i[0] for i in pDiv]))/5.0,
                                       1.5/sum([i[1]-i[0] for i in pDiv]),
                                      (1.5/sum([i[1]-i[0] for i in pDiv]))/5.0),
                            4)
 
            # Only print y label if the first column
            for y in yticks:
                plt.plot([road[0], road[1]], [y,y], ":", lw=0.75, color="black", alpha=0.6)    
            if (spCol==1): 
                plt.yticks(yticks, [str(x) for x in yticks], fontsize=12)    
                plt.ylabel("Probability", fontsize=14,fontname='Helvetica')
            else: 
                plt.yticks([])    
 
            # Set axis properties
            plt.axis([road[0], road[1], 0, 1.75/sum([i[1]-i[0] for i in pDiv])])

    endTime = time()
    print "computation time: {}".format(endTime - startTime)
    plt.show()


def visualizeSecondLaneSimulation():
    physDivIvs = [[200, 400], [600, 800]]
    roadIv = [0, 1000]
    roadIvLen = roadIv[1] - roadIv[0]
    speed = lambda carId: \
                lambda t: 10/3.6 * np.sin(2*np.pi*t/roadIvLen - carId) + 40/3.6 
    timeStart = 0
    timeDelta = 1
    sensorPdfGivenTrueVal = PdfGivenTrueVal.binomial(1)
    gpsSigma = 1
    obsCarLen = 4
    obsGapLen = 26
    obsOffset = -5
    obsCars = InfiniteUniformIntervalsSet(obsCarLen, obsGapLen, obsOffset)
    obsCarsSpeed = 80/3.6
    carCount = 1

    plt.subplot(211)
    for i in range(carCount):
        xs = np.linspace(roadIv[0], roadIv[1], 1000)
        plt.plot(xs, speed(i)(xs), "b-")
    plt.title("speeds used")
    plt.grid(True)

    cars = [Car(speed(i),
                roadIv, timeStart, timeDelta, sensorPdfGivenTrueVal)\
               for i in range(carCount)]
    physDiv = indicator(physDivIvs)

    sls = SecondLaneSimulation(cars, physDiv, gpsSigma, obsCars, obsCarsSpeed)
    sd = sls.getSimulationData()

    positives = filter(lambda p: p[1] > 0, sd)

    #print "positives: {}".format(positives)
    print "count of positives: {}".format(len(positives))

    xs, ys = zip(*sd)
    plt.subplot(212)
    plt.plot(xs, ys, 'bo', xs, ys, 'r-')
    plt.title("positives with obstructing cars, car count: {}".format(carCount))
    plt.ylabel("sensor reading")
    plt.xlabel("x position in meters")
    plt.axis([roadIv[0]-(roadIvLen/10.0), roadIv[1]+(roadIvLen/10.0), -.5, 1.5])
    plt.grid(True)
    plt.show()


def visualizeSpectrum2():
    # units: the units of length and time are arbitrary, but must of course be
    # consistent, so, once you choose e.g. meter and seconds, stick with it.

    # Define the section of road
    road = [0, 400]

    # Define the locations of physical dividers as intervals
    pDiv = [[100,190],[210,300]]
    physDiv = indicator(pDiv)

    # Minimum GPS error
    gpsSigma = 6

    # Moving average length (i.e. the number of bins used) and the bin length
    convLens = [50, 100]
    convBinLen = .1

    # this is the extension of the road to diminish the boundary effects
    ext = max(3*gpsSigma, max(convLens)*convBinLen/2.0)
    extRoad = [road[0] - ext, road[1] + ext]

    # Information for histogram plot
    histBinCount = (road[1] - road[0])/convBinLen
    normed = True
    histcolor = "#3F5D7D"

    # Start time of car trace, time steps in car trace
    timeStart = 0
    timeDelta = 1

    # Sensor accuracy (fractional percentage)
    binomialP = .9

    # number of cars used for simulation
    #carCount = 10**5
    carCount = 3000

    startTime = time()

    cars = map(lambda i: Car(BoundedMarkovChain(
                                 50/3.6,.1/3.6,30/3.6,65/3.6),
                             extRoad,
                             timeStart,
                             timeDelta,
                             PdfGivenTrueVal.binomial(binomialP)),
               range(carCount))

    sim = OneLaneSimulation(cars, physDiv, gpsSigma)
    sd = sim.getSimulationData()
    mAvExt, sdHistExt = movingAvgWoExtDensityHist(sd, extRoad, convLens[0], convBinLen)
    mAv = [i for i in mAvExt if road[0] <= i[0] and i[0] <= road[1]]
    #sdHist = [i for i in sdHistExt if road[0] <= i[0] and i[0] <= road[1]]
    mAvx, mAvy = zip(*mAv)

    sig1xs = [p[0] for p in sd if p[1] > 0]
    sdHistXs, sdHistYs = zip(*sdHistExt)

    fig = plt.figure(figsize = (15, 5))

    # plot sd (simData) without histogram lines (linewidth = 0)
    fig.add_subplot(1, 3, 1)
    plt.grid(True)
    plt.hist(sig1xs,
             bins=np.linspace(road[0], road[1], histBinCount),
             linewidth=0,
             normed = True,
             color = 'darkslateblue')
    #plt.plot(mAvx, mAvy/(sum(mAvy)*convBinLen),color='r',lw=1,zorder=5)

    # plot sd (simData) with histogram lines (linewidth = default)
    fig.add_subplot(1, 3, 2)
    plt.grid(True)
    plt.hist(sig1xs,
             bins=np.linspace(road[0], road[1], histBinCount),
             #linewidth=0,
             normed = True,
             color = 'darkslateblue')
    #plt.plot(mAvx, mAvy/(sum(mAvy)*convBinLen),color='r',lw=1,zorder=5)

    # plot sdHistExt
    fig.add_subplot(1, 3, 3)
    plt.grid(True)
    plt.plot(sdHistXs, sdHistYs, color = 'darkslateblue', linestyle = '-')
    plt.plot(mAvx, mAvy/(sum(mAvy)*convBinLen),color='r',lw=1,zorder=5)

    plt.show()

def visualizeSpectrum():
    # units: the units of length and time are arbitrary, but must of course be
    # consistent, so, once you choose e.g. meter and seconds, stick with it.

    # Define the section of road
    road = [0, 2000]

    # Define the locations of physical dividers as intervals
    pDiv = [[500,800],[1200,1500]]
    physDiv = indicator(pDiv)

    # Minimum GPS error
    gpsSigma = 7

    # Moving average length (i.e. the number of bins used) and the bin length
    convLens = [100, 2000]
    convBinLen = .1

    # this is the extension of the road to diminish the boundary effects
    ext = max(3*gpsSigma, max(convLens)*convBinLen/2.0)
    extRoad = [road[0] - ext, road[1] + ext]

    # Information for histogram plot
    histBinCount = (road[1] - road[0])/convBinLen
    normed = True
    histcolor = "#3F5D7D"

    # Start time of car trace, time steps in car trace
    timeStart = 0
    timeDelta = 1

    # Sensor accuracy (fractional percentage)
    binomialP = .9

    # number of cars used for simulation
    #carCount = 10**5
    carCount = 3000

    startTime = time()

    #cars = map(lambda i: Car(lambda s:\
    #                             5/3.6*sin(2*pi*i/carCount)+50/3.6,
    #                         extRoad,
    #                         timeStart,
    #                         timeDelta,
    #                         PdfGivenTrueVal.binomial(binomialP)),
    #           range(carCount))
    cars = map(lambda i: Car(BoundedMarkovChain(
                                 50/3.6,.1/3.6,30/3.6,65/3.6),
                             extRoad,
                             timeStart,
                             timeDelta,
                             PdfGivenTrueVal.binomial(binomialP)),
               range(carCount))

    sim = OneLaneSimulation(cars, physDiv, gpsSigma)
    sd = sim.getSimulationData()

    for spCol in (1, 2):
        mAvExt, sdHistExt = movingAvgWoExt(sd, extRoad, convLens[spCol-1], convBinLen)
        mAv = [i for i in mAvExt if road[0] <= i[0] and i[0] <= road[1]]
        sdHist = [i for i in sdHistExt if road[0] <= i[0] and i[0] <= road[1]]

        inds = inferFuzzyIndicators(mAv)

        # Remove the plot frame lines. They are unnecessary chartjunk.
        ax = plt.subplot(2, 2, spCol)
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False)    

        # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
        plt.tick_params(axis="both", which="both", bottom="on", top="off",    
                        labelbottom="on", left="off", right="off", labelleft="on")    

        # Ensure that the axis ticks only show up on the bottom and left of the plot.    
        # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
        ax.get_xaxis().tick_bottom()    

    
        # Plot histogram of physical divider probability 
        plt.hist([p[0] for p in sd if p[1] > 0], 
                 bins=np.linspace(road[0], road[1], histBinCount),
                 histtype='stepfilled', 
                 normed=normed,
                 color=histcolor,
                 linewidth=0,
                 zorder=4)

        # Plot the moving average of the histogram
        mAvx, mAvy = zip(*mAv)
        plt.plot(mAvx, mAvy/(sum(mAvy)*convBinLen),color='k',lw=1,zorder=5)

        # Plot lines that indicate where the line crosses 1/2 the max of the moving average
        # and add text above that indicates the distance from the actual divider
        print "len(inds): {}".format(len(inds))
        print "len(pDiv): {}".format(len(pDiv))
        for ind in range(0,len(inds)):
            plt.plot([inds[ind][0],inds[ind][0]],
                     [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                     color = (44/ 255., 160/ 255., 44/ 255.),
                     lw=1.5,linestyle='-',zorder=5)
            plt.text(inds[ind][0],1.28/sum([i[1]-i[0] for i in pDiv]),
                     '%4.2f'%(inds[ind][0]-pDiv[ind][0]),
                      verticalalignment='bottom',
                      horizontalalignment='center',
                      color=(44/ 255., 160/ 255., 44/ 255.), 
                      fontsize=8)
            plt.plot([inds[ind][1],inds[ind][1]],
                     [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                     color = (214/ 255., 39/ 255., 40/ 255.),
                     lw=1.5,linestyle='-',zorder=5)
            plt.text(inds[ind][1],1.28/sum([i[1]-i[0] for i in pDiv]),
                     '%4.2f'%(inds[ind][1]-pDiv[ind][1]),
                      verticalalignment='bottom',
                      horizontalalignment='center',
                      color=(214/ 255., 39/ 255., 40/ 255.), 
                      fontsize=8)

        # Plot simulation specifications
        plt.text((road[1]-road[0])/100,1.45/sum([i[1]-i[0] for i in pDiv]),
                 'BW = %4.1f m' % (convBinLen*convLens[spCol-1]),
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 color='k', 
                 fontsize=10)
        plt.text((road[1]-road[0]),1.45/sum([i[1]-i[0] for i in pDiv]),
                 '$\sigma$(sensor) = %4.1f%%' % ((1.0-binomialP)*100),
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 color='k', 
                 fontsize=10)
        plt.text((road[1]-road[0])/100,1.625/sum([i[1]-i[0] for i in pDiv]),
                 'n = %d' % (carCount),
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 color='k', 
                 fontsize=10)

        # Add patch in background to indicate the actual physical divider
        for ind in pDiv:
            ax.add_patch(patches.Rectangle((ind[0],0),   # (x,y)
                                           (ind[1]-ind[0]),          # width
                                           1.25/sum([i[1]-i[0] for i in pDiv]), # height
                         facecolor=(0.7,0.7,0.7),lw=0,
                         zorder=3))
                          
        # Only print x label if the bottom row
        xticks=np.linspace(road[0],road[1],6,dtype=int)                        
        plt.xlabel("Along-track distance (m)", fontsize=14,fontname='Helvetica')
        plt.xticks(xticks, [str(x) for x in xticks], fontsize=10)    
   
        # Provide tick lines across the plot to help your viewers trace along    
        # the axis ticks. Make sure that the lines are light and small so they    
        # don't obscure the primary data lines.    
        yticks=np.arange((1.5/sum([i[1]-i[0] for i in pDiv]))/5.0,
                       1.5/sum([i[1]-i[0] for i in pDiv]),
                      (1.5/sum([i[1]-i[0] for i in pDiv]))/5.0)

        # Only print y label if the first column
        for y in yticks:
            plt.plot([road[0], road[1]], [y,y], ":", lw=0.75, color="black", alpha=0.6)    
        if (spCol==1): 
            plt.yticks(yticks, [str(x) for x in yticks], fontsize=12)    
            plt.ylabel("Probability", fontsize=14,fontname='Helvetica')
        else: 
            plt.yticks([])    

        # Set axis properties
        plt.axis([road[0], road[1], 0, 1.75/sum([i[1]-i[0] for i in pDiv])])

        # plot spectra
        plt.subplot(2, 2, 2 + spCol)
        ftMa = fft(mAvy)
        sdHistX, sdHistY = zip(*sdHist)
        ftH = fft(sdHistY)
        plt.grid(True)
        plt.xlim([0, 50])
        plt.plot(np.absolute(ftMa), 'b-', np.absolute(ftH), 'r-')
            
    endTime = time()
    print "computation time: {}".format(endTime - startTime)
    plt.show()


def pdfOfDetectedDividers():
    road = [0, 1000]

    # Define the locations of physical dividers as intervals
    pDiv=[[400, 600]]
    physDiv = indicator(pDiv)

    # GPS error
    gpsSigma = 10

    # Moving average length (i.e. the number of bins used) and the bin length
    # the convolution length in meter is then convLen * convBinLen
    convLen = 1000
    convBinLen = .01

    # Start time of car trace, time steps in car trace
    timeStart = 0
    timeDelta = .1

    # Sensor accuracy (fractional percentage)
    binomialP = .9

    # this is the extension of the road to diminish the boundary effects
    ext = max(3*2*gpsSigma, convLen*convBinLen/2.0)
    extRoad = [road[0] - ext, road[1] + ext]

    # number of cars used for simulation
    carCount = 2*10**4

    # number of simulations
    simCount = 50

    def inferDividerBounds():
        #cars = map(lambda i: Car(lambda s:\
        #                             5/3.6*sin(2*pi*float(i)/carCount)+50/3.6,
        #                         extRoad,
        #                         timeStart,
        #                         timeDelta,
        #                         PdfGivenTrueVal.binomial(binomialP)),
        #           range(carCount))
        cars = map(lambda i: Car(BoundedMarkovChain(
                                     50/3.6,.1/3.6,30/3.6,65/3.6),
                                 extRoad,
                                 timeStart,
                                 timeDelta,
                                 PdfGivenTrueVal.binomial(binomialP)),
                   range(carCount))

        sim = OneLaneSimulation(cars,
                                physDiv,
                                gpsSigma)
        sd = sim.getSimulationData()

        mAvExt, sdHistExt = movingAvgWoExt(sd, extRoad, convLen, convBinLen)
        mAv = [i for i in mAvExt if road[0] <= i[0] and i[0] <= road[1]]

        bounds = inferFuzzyIndicators(mAv)[0]
        boundsRounded = np.round(bounds, 2)
        print boundsRounded
        return boundsRounded

    startTime = time()
    multipleInds = [inferDividerBounds() for i in range(simCount)]
    endTime = time()

    # compute sd and mean
    print multipleInds
    print "##########"
    divStarts, divEnds = zip(*multipleInds)
    sdDivStarts = np.round(np.std(divStarts), 2)
    mDivStarts = np.round(np.mean(divStarts), 2)
    sdDivEnds = np.round(np.std(divEnds), 2)
    mDivEnds = np.round(np.mean(divEnds), 2)
    print "divStarts: {}".format(divStarts)
    print "divEnds: {}".format(divEnds)
    print "std of divider start: {}".format(sdDivStarts)
    print "std of divider end: {}".format(sdDivEnds)

    # plot the multiple simulations
    fig = plt.figure(figsize = (13, 5))
    fig.suptitle("road: [{}, {}], pd = [{}, {}], sensor err: {}, gps err: {}"\
                     ", conv. len = {}, conv. bin len = {}, cars: {}, "\
                     "sims = {}".format(road[0], road[1], pDiv[0][0],
                                        pDiv[0][1], binomialP, gpsSigma,
                                        convLen, convBinLen, carCount,
                                        simCount),
                 fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = .85, left = .05, right = .95)
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.title(r'divider start: $\sigma = {}, \mu = {}$'.format(sdDivStarts, mDivStarts))
    #plt.hist(np.array(divStarts) - pDiv[0][0], color = "darkslateblue", range = (-.5, .5))
    print "to plot: {}".format(np.array(divStarts) - pDiv[0][0])
    plt.hist(np.array(divStarts) - pDiv[0][0], color = "slateblue", alpha = .7)
    plt.axvline(x = 0, color = "seagreen", linewidth = 4, alpha = .7)
    plt.axvline(x = mDivStarts-pDiv[0][0], color = "salmon", linewidth = 4, alpha = .7)
    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.title(r'divider end: $\sigma = {}, \mu = {}$'.format(sdDivEnds, mDivEnds))
    #plt.hist(np.array(divEnds) - pDiv[0][1], color = "darkslateblue", range = (-.5, .5))
    plt.hist(np.array(divEnds) - pDiv[0][1], color = "slateblue", alpha = .7)
    plt.axvline(x = 0, color = "seagreen", linewidth = 4, alpha = .7)
    plt.axvline(x = mDivEnds-pDiv[0][1], color = "salmon", linewidth = 4, alpha = .7)

    print "time: {}".format(endTime - startTime)
    plt.show()


def boundarySdOverCarCount():
    road = [0, 1000]

    # Define the locations of physical dividers as intervals
    pDiv=[[400, 600]]
    physDiv = indicator(pDiv)

    # GPS error
    gpsSigma = 5

    # Moving average length (i.e. the number of bins used) and the bin length
    # the convolution length in meter is then convLen * convBinLen
    convLen = 1000
    convBinLen = .01

    # Start time of car trace, time steps in car trace
    timeStart = 0
    timeDelta = .1

    # Sensor accuracy (fractional percentage)
    binomialP = .9

    # this is the extension of the road to diminish the boundary effects
    ext = max(3*2*gpsSigma, convLen*convBinLen/2.0)
    extRoad = [road[0] - ext, road[1] + ext]

    # number of cars used for simulation
    carCountsMult = 10**3
    carCounts = np.array([4, 7, 10, 13, 16, 19, 22]) * carCountsMult

    # number of simulations
    simCount = 50

    def computeStds(carCount):
        def inferDividerBounds():
            #cars = map(lambda i: Car(lambda s:\
            #                             5/3.6*sin(2*pi*float(i)/carCount)+50/3.6,
            #                         extRoad,
            #                         timeStart,
            #                         timeDelta,
            #                         PdfGivenTrueVal.binomial(binomialP)),
            #           range(carCount))
            cars = map(lambda i: Car(BoundedMarkovChain(
                                         50/3.6,.1/3.6,30/3.6,65/3.6),
                                     extRoad,
                                     timeStart,
                                     timeDelta,
                                     PdfGivenTrueVal.binomial(binomialP)),
                       range(carCount))

            sim = OneLaneSimulation(cars,
                                    physDiv,
                                    gpsSigma)
            sd = sim.getSimulationData()

            mAvExt, sdHistExt = movingAvgWoExt(sd, extRoad, convLen, convBinLen)
            mAv = [i for i in mAvExt if road[0] <= i[0] and i[0] <= road[1]]

            bounds = inferFuzzyIndicators(mAv)[0]
            boundsRounded =  np.round(bounds, 2)
            print boundsRounded
            return boundsRounded

        startTime = time()
        # compute divider bounds simCount times
        multipleInds = [inferDividerBounds() for i in range(simCount)]

        # compute sd
        divStarts, divEnds = zip(*multipleInds)
        mDivStarts = np.mean(divStarts)
        sdDivStarts = np.std(divStarts)
        mDivEnds = np.mean(divEnds)
        sdDivEnds = np.std(divEnds)
        endTime = time()
        print "########## carCount: {} ############".format(carCount)
        print "computing time: {}".format(endTime - startTime)
        print "mean of divider start: {}".format(mDivStarts)
        print "std of divider start: {}".format(sdDivStarts)
        print "mean of divider start: {}".format(mDivEnds)
        print "std of divider end: {}".format(sdDivEnds)
        print "####################################"
        
        return mDivStarts, sdDivStarts, mDivEnds, sdDivEnds
        
    startTime = time()
    moments = map(computeStds, carCounts)
    msStarts, sdsStarts, msEnds, sdsEnds = zip(*moments)
    endTime = time()

    # print and plot moments over carCount
    sdsDf = pd.DataFrame({'car count': carCounts,
                          'starts mean': np.round(msStarts, 2),
                          'starts sd': np.round(sdsStarts, 2),
                          'ends mean': np.round(msEnds, 2),
                          'ends sd': np.round(sdsEnds, 2),
                        })
    cols = sdsDf.columns.tolist()
    cols = [cols[0], cols[3], cols[4], cols[1], cols[2]]
    sdsDf = sdsDf[cols]

    print "-----------------------"
    print "sim count: {}".format(simCount)
    print "div: [{}, {}]".format(pDiv[0][0], pDiv[0][1])
    print "sensor err: {}".format(binomialP)
    print "gps sd: {}".format(gpsSigma)
    print "conv bin len: {}, conv len: {}".format(convBinLen, convLen)
    print "----"
    print sdsDf
    print "-----------------------"
    print "computation time: {}".format(endTime - startTime)

    fig = plt.figure(figsize = (14, 5))
    fig.suptitle("div: [{}, {}], sensor err: {}, gps sd: {}, "\
                     "conv. bin len: {}, conv. len: {}, simulations = {}".\
                     format(pDiv[0][0], pDiv[0][1], binomialP,
                         gpsSigma, convBinLen, convLen, simCount),
                 fontsize=14, fontweight='bold')
    # fig.subplots_adjust: adjust the top to make more space beneath the
    # suptitle, adjust the left and right to remove that ugly empty space on
    # both sides of the plot window.
    # Scale is from 0 to 1 so top = 1 would be at the very top of the figure,
    # left = 0 at the very left of the figure and so on.
    fig.subplots_adjust(top = .85, left = .05, right = .95)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_xlabel("car counts ({})".format(carCountsMult))
    plt.grid(True)
    plt.title("means of starts")
    plt.plot(carCounts/carCountsMult, msStarts, 'g-')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_xlabel("car counts ({})".format(carCountsMult))
    plt.grid(True)
    plt.title("means of ends")
    plt.plot(carCounts/carCountsMult, msEnds, 'b-')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_xlabel("car counts ({})".format(carCountsMult))
    plt.grid(True)
    plt.title("standard deviations")
    plt.plot(carCounts/carCountsMult, sdsStarts, 'g-', carCounts/carCountsMult, sdsEnds, 'b-')
    plt.show()


def gaussianConvolutionExtrems():
    iv = [-10, 30]
    supp1 = [5, 15]
    supp2 = [9.5, 10.5]
    sigma1 = 1
    sigma2 = 1
    pd1 = indicator([supp1])
    pd2 = indicator([supp2])

    xs1, ys1 = convGaussWithPd(supp1, sigma1, iv, 500)
    xs2, ys2 = convGaussWithPd(supp2, sigma2, iv, 500)

    fig = plt.figure(figsize = (10,5))
    fig.subplots_adjust(top = .85)
    fig.suptitle("Convolution of physical divider with Gaussian",
            fontsize = 14, fontweight = 'bold')

    fig.add_subplot(121)
    plt.grid(True)
    plt.title(r'$\sigma = {}$'.format(sigma1))
    plt.xlim([0, 20])
    plt.ylim([-.1, 1.1])
    plt.axhline(0, color = 'black')
    plt.plot(xs1, [pd1(x) for x in xs1], 'b-')
    plt.plot(xs1, ys1, 'g-')
    plt.axhline(.5, color = 'salmon', alpha = .5)
    
    fig.add_subplot(122)
    plt.grid(True)
    plt.title(r'$\sigma = {}$'.format(sigma2))
    plt.xlim([0, 20])
    plt.ylim([-.1, 1.1])
    plt.axhline(0, color = 'black')
    plt.plot(xs2, [pd2(x) for x in xs2], 'b-')
    plt.plot(xs2, ys2, 'g-')
    plt.axhline(.5, color = 'salmon', alpha = .5)
    
    plt.show()


def howGpsErrorLeadsToConvolution():
    iv = [0, 20]
    supp = [7, 18]
    sigma = 2.0
    mu = 6.0
    gFun = lambda x: 1/sigma*norm.pdf((x-mu)/sigma)
    pd = indicator([supp])

    xsPd = np.arange(iv[0], iv[1], .1)
    ysPd = [pd(x) for x in xsPd]

    xsG = xsPd
    ysG = [4*gFun(x) for x in xsG]

    fig = plt.figure(figsize = (5, 4))
    fig.subplots_adjust(top = .85)
    fig.suptitle("GPS error and physical divider",
            fontsize = 14, fontweight = 'bold')

    #plt.grid(True)
    plt.ylim([-.1, 1.1])
    plt.plot(xsPd, ysPd, linestyle = '-', color = 'slateblue')
    plt.plot(xsG, ysG, linestyle = '-', color = 'seagreen')
    plt.axvline(x = mu, color = 'salmon', alpha = .5)
    plt.axhline(y = 0, color = 'black', alpha = .5)
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    
    plt.show()


def sensorFpr():
    # units: the units of length and time are arbitrary, but must of course be
    # consistent, so, once you choose e.g. meter and seconds, stick with it.
    
    # Define the section of road
    road = [0, 2000]

    # Define the locations of physical dividers as intervals
    pDiv = [[100,500],[1500,1900]]
    physDiv = indicator(pDiv)

    # Minimum GPS error
    gpsSigma = 5

    # Moving average length (i.e. the number of bins used) and the bin length
    convLen = 500
    convBinLen = .1

    # Information for histogram plot
    histBinCount = (road[1] - road[0])/convBinLen
    normed = True
    histcolor = "#3F5D7D"

    # Start time of car trace, time steps in car trace
    timeStart = 0
    timeDelta = .1

    # Sensor accuracy (fractional percentage)
    binomialPs = [.99, .95, .9, .8, .7]

    # this is the extension of the road to diminish the boundary effects
    ext = max(4*gpsSigma, convLen*convBinLen/2.0)
    extRoad = [road[0] - ext, road[1] + ext]

    # some subplots (sp: SubPlot)
    fig = plt.figure()
    fig.suptitle("physDivs: {}, conv. bw. = {}, conv. bin len = {}".format(
                                        pDiv, convLen*convBinLen, convBinLen),
                 fontsize = 14, fontweight = 'bold')
    fig.subplots_adjust(top = .9, left = .05, right = .95)
    spRows = (1, 2, 3)
    startTime = time()
    for spRow in spRows:
        #carCount = 5 * 10**(spRow)
        carCount = 2 * 10**(spRow-1)

        spCols = range(1, len(binomialPs)+1)
        for spCol in spCols:
            cars = map(lambda i: Car(BoundedMarkovChain(
                                         50/3.6,.1/3.6,30/3.6,65/3.6),
                                     extRoad,
                                     timeStart,
                                     timeDelta,
                                     PdfGivenTrueVal.binomial(
                                         binomialPs[spCol-1])),
                       range(carCount))
            sim = OneLaneSimulation(cars,
                                    physDiv,
                                    gpsSigma)
            sd = sim.getSimulationData()

            mAvExt, sdHistExt = movingAvgWoExt(sd, extRoad, convLen, convBinLen)
            mAv = [i for i in mAvExt if road[0] <= i[0] and i[0] <= road[1]]

            inds = inferFuzzyIndicators(mAv)

            # get a region between physical dividers and then compute the fpr
            # from data in this region
            if len(inds) == len(pDiv):
                fprRg = findFprRegion(inds, gpsSigma, 50)
                if fprRg is not None:
                    fpr = getFprInRegion(sd, fprRg)
                    fprRgCenter = fprRg[0] + (fprRg[1] - fprRg[0])/2
                else:
                    print "fprRg is None"
                    fpr = None
            else:
                fpr = None

            # Remove the plot frame lines. They are unnecessary chartjunk.
            ax = plt.subplot(max(spRows), max(spCols), max(spCols)*(spRow-1) + spCol)
            ax.spines["top"].set_visible(False)    
            ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(False)    

            # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
            plt.tick_params(axis="both", which="both", bottom="on", top="off",    
                            labelbottom="on", left="off", right="off", labelleft="on")    

            # Ensure that the axis ticks only show up on the bottom and left of the plot.    
            # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
            ax.get_xaxis().tick_bottom()    

            # Plot histogram of physical divider probability 
            sightingSites = [p[0] for p in sd if p[1] > 0]
            plt.hist(sightingSites,
                     bins=np.linspace(road[0], road[1], histBinCount),
                     histtype='stepfilled',
                     normed=normed,
                     color=histcolor,
                     linewidth=0 ,
                     zorder=4)

            # Plot the moving average of the histogram
            mAvx, mAvy = zip(*mAv)
            plt.plot(mAvx, mAvy/(sum(mAvy)*convBinLen),color='k',lw=1,zorder=5)

            # Plot lines that indicate where the line crosses 1/2 the max of the moving average
            # and add text above that indicates the distance from the actual divider
            if len(inds) != len(pDiv):
                print "WARNING: there are {} pds but {} had been detected".format(len(pDiv), len(inds))
            else:
                for ind in range(0,len(inds)):
                    plt.plot([inds[ind][0],inds[ind][0]],
                             [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                             color = (44/ 255., 160/ 255., 44/ 255.),
                             lw=1.5,linestyle='-',zorder=5)
                    plt.text(inds[ind][0],1.28/sum([i[1]-i[0] for i in pDiv]),
                             '%4.2f'%(inds[ind][0]-pDiv[ind][0]),
                              verticalalignment='bottom',
                              horizontalalignment='center',
                              color=(44/ 255., 160/ 255., 44/ 255.), 
                              fontsize=8)
                    plt.plot([inds[ind][1],inds[ind][1]],
                             [0,1.25/sum([i[1]-i[0] for i in pDiv])],
                             color = (214/ 255., 39/ 255., 40/ 255.),
                             lw=1.5,linestyle='-',zorder=5)
                    plt.text(inds[ind][1],1.28/sum([i[1]-i[0] for i in pDiv]),
                             '%4.2f'%(inds[ind][1]-pDiv[ind][1]),
                              verticalalignment='bottom',
                              horizontalalignment='center',
                              color=(214/ 255., 39/ 255., 40/ 255.), 
                              fontsize=8)

            # Plot simulation specifications
            plt.text((road[1]-road[0])/100,1.45/sum([i[1]-i[0] for i in pDiv]),
                     '$\sigma$(GPS) = %4.1f m' % (gpsSigma),
                     verticalalignment='center',
                     horizontalalignment='left',
                     color='k', 
                     fontsize=10)
            plt.text((road[1]-road[0]),1.45/sum([i[1]-i[0] for i in pDiv]),
                     'fpr = %.0f%%' % ((1.0-binomialPs[spCol-1])*100),
                     verticalalignment='bottom',
                     horizontalalignment='right',
                     color='k', 
                     fontsize=10)
            plt.text((road[1]-road[0])/100,1.625/sum([i[1]-i[0] for i in pDiv]),
                     'n = %d' % (carCount),
                     verticalalignment='bottom',
                     horizontalalignment='left',
                     color='k', 
                     fontsize=10)
            if fpr is not None:
                plt.text(fprRgCenter, 1.1/sum([i[1]-i[0] for i in pDiv]),
                         'fpr = %.1f%%' % (fpr*100),
                         verticalalignment='bottom',
                         horizontalalignment='center',
                         color='g', 
                         fontsize=10)
 
            # Add patch in background to indicate the actual physical divider
            for ind in pDiv:
                ax.add_patch(patches.Rectangle((ind[0],0),   # (x,y)
                                               (ind[1]-ind[0]),          # width
                                               1.25/sum([i[1]-i[0] for i in pDiv]), # height
                             facecolor=(0.7,0.7,0.7),lw=0,
                             zorder=3))
                              
            # Only print x label if the bottom row
            xticks=np.linspace(road[0],road[1],6,dtype=int)                        
            if (spRow==max(spRows)): 
                plt.xlabel("Along-track distance (m)", fontsize=14,fontname='Helvetica')
            plt.xticks(xticks, [str(x) for x in xticks], fontsize=10)    
       
            # Provide tick lines across the plot to help your viewers trace along    
            # the axis ticks. Make sure that the lines are light and small so they    
            # don't obscure the primary data lines.    
            yticks=np.round(np.arange((1.5/sum([i[1]-i[0] for i in pDiv]))/5.0,
                                       1.5/sum([i[1]-i[0] for i in pDiv]),
                                      (1.5/sum([i[1]-i[0] for i in pDiv]))/5.0),
                            4)
 
            # Only print y label if the first column
            for y in yticks:
                plt.plot([road[0], road[1]], [y,y], ":", lw=0.75, color="black", alpha=0.6)    
            if (spCol==1): 
                plt.yticks(yticks, [str(x) for x in yticks], fontsize=12)    
                plt.ylabel("Probability", fontsize=14,fontname='Helvetica')
            else: 
                plt.yticks([])    
 
            # Set axis properties
            plt.axis([road[0], road[1], 0, 1.75/sum([i[1]-i[0] for i in pDiv])])

    endTime = time()
    print "computation time: {}".format(endTime - startTime)
    plt.show()


def findFprRegion(inds, gpsSigma, minSize):
    gaps = [ [inds[i][1], inds[i+1][0]] for i in range(len(inds)-1)]
    gapSizes = map(lambda g: g[1] - g[0] - 2*3*gpsSigma, gaps)
    maxInd = max(zip(range(len(gapSizes)), gapSizes), key = lambda p: p[1])
    if maxInd[1] <= minSize:
        ret = None
    else:
        ret = gaps[maxInd[0]]
    return ret


def getFprInRegion(data, fprRg, confidence):
    dataInReg = filter(lambda p: fprRg[0] < p[0] and p[0] < fprRg[1], data)
    positives = filter(lambda p: p[1] > 0, dataInReg)
    total = len(dataInReg)
    posCount = len(positives)
    negCount = total - posCount
    print "posCount: {}, negCount: {}".format(posCount, negCount)
    fpr = float(len(positives))/len(dataInReg)
    ci = beta(posCount + 1, negCount + 1).interval(confidence)
    return fpr, ci

###############################################################################

if __name__ == "__main__":
    visualizeSimulation()
    #visualizeSpectrum()
    #visualizeSpectrum2()
    #visualizeSecondLaneSimulation()
    #visualizeInfluenceOfSensorError()
    #pdfOfDetectedDividers()
    #boundarySdOverCarCount()
    #visualizeBandwidthEffect()
    #gaussianConvolutionExtrems()
    #howGpsErrorLeadsToConvolution()
    #sensorFpr()
    #findFprRegion([[100, 500], [700, 800], [1000, 1200], [1500, 1900]], 5, 290)
    # fpr = 2/8 = .25
    #print getFprInRegion([
    #        [ 5, 0],
    #        [15, 0],
    #        [15, 0],
    #        [16, 1],
    #        [16, 1],
    #        [18, 0],
    #        [18, 0],
    #        [18, 0],
    #        [18, 0],
    #        [22, 1],
    #    ], [10, 20], .9)
