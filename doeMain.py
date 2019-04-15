from importlib import import_module
import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import shutil

# ******************************************************************************************************************** #
# Initial Version: Created by Abhishek R (cws:ra2)

# ******************************************************************************************************************** #
class doeMaster(object):
    """
        Main Class For all DOEs.
    """
    
    # **************************************************************************************************************** #
    def __init__(self):
        """
        Intialize the method available to doeMaster class.

        """
        self.listPackages() # Set the list of packages supported
        self.getInputDistributionTypes() # Set the input Distribution Type supported
        self.nInputs = None #
        self.nSamples = None

    # **************************************************************************************************************** #
    def listPackages(self):
        """
        Can be used to view all the eligible packages.
        :return: List of Available Packages
        """

        listSourcePackages = []
        try:
            import pyDOE
            listSourcePackages.append('pyDOE')
        except ImportError, e:
            pass
        try:
            import diversipy
            listSourcePackages.append('diversipy')
        except ImportError:
            pass

        self.listSourcePackages = listSourcePackages
        return self.listSourcePackages

    # **************************************************************************************************************** #
    def setPackage(self, packageName):
        """
        Select the wrapper class corresponding to the package name.
        :param packageName:
            Provide package name, use listPackages() to get the list of available packages
        :return:
            object of the Wrapper Class corresponding to the input package name, error string incase of invalid package

        """
        if str(packageName).lower() == 'pydoe':
            wrapperObj = pyDOEMaster()
        elif str(packageName).lower() == 'diversipy':
            wrapperObj = diversipyMaster()
        else:
            wrapperObj = 'invalid package'
        return wrapperObj

    # **************************************************************************************************************** #
    def lhsCorrScore(self, inputMatrix):
        """
        This Function Calculate the correlation between each columns of the input matrix.
        Calculates RHO as a score of correlation

        :param input_matrix:
            Input Matrix for which the score has to be calculated.
        :return:
            rho,maxPairwiseCorr,corrMatrix
        """

        # Drop if any discrete columns exists
        nCols = len(inputMatrix[0, :])  # No of Columns of data in the given matrix

        discreteColIdx = []  # Stores the index of discrete columns
        contColIdx = []  # Stores the index of continuous columns

        # Loop Over each Cols and find out which are discrete.
        # Identification of discrete is that first value should be a string.
        # Better logic can be implemented but not in need of it right now.
        for i in range(nCols):

            # Column data.
            if type(inputMatrix[0, i]) == str:  # if string type set as discrete column
                discreteColIdx.append(i)
            else:
                contColIdx.append(i)  # Else continous column

        inputMatrix = inputMatrix[:, contColIdx]

        # Calculate the correlation matrix
        corrMatrix = np.corrcoef(inputMatrix, rowvar=False)

        # Get the No of Columns of the input matrix
        nParams = len(inputMatrix[0, :])

        # Loop over column pair.
        numerator = 0
        for i in range(1, nParams):
            for j in range(0, i):
                numerator = numerator + corrMatrix[i, j] ** 2
        denominator = nParams * (nParams - 1) / 2

        # Calculating the RHO value.
        rho = np.sqrt(numerator / denominator)

        # Get maximum pairwise correlation
        di = np.diag_indices(nParams)
        corrMatrixDup = corrMatrix
        for i in di[0]:
            corrMatrixDup[i, i] = np.nan

        maxPairwiseCorr = np.nanmax(corrMatrixDup.ravel())

        print '*** RHO ***'
        print rho
        print '*** Max Pairwise Correlation ***'
        print maxPairwiseCorr
        print '*** Correlation Table ***'
        print corrMatrixDup
        return rho, maxPairwiseCorr, corrMatrix

    # **************************************************************************************************************** #
    def lhsDistScore(self, inputMatrix):
        """
        This Function Calculate the L1 and L2 distrance between each row of the input matrix.
        :param inputMatrix:
            Input Matrix for which the score has to be calculated.
        :return:
            mean L1 distance,mean L2 distance
        """

        # Calculate the nSamples in the given matrix
        nRows = len(inputMatrix[:, 0])


        # Drop if any discrete columns exists
        nCols = len(inputMatrix[0, :])  # No of Columns of data in the given matrix

        discreteColIdx = []  # Stores the index of discrete columns
        contColIdx = []  # Stores the index of continuous columns

        # Loop Over each Cols and find out which are discrete.
        # Identification of discrete is that first value should be a string.
        # Better logic can be implemented but not in need of it right now.
        for i in range(nCols):

            # Column data.
            if type(inputMatrix[0, i]) == str:  # if string type set as discrete column
                discreteColIdx.append(i)
            else:
                contColIdx.append(i)  # Else continous column

        inputMatrix = inputMatrix[:,contColIdx]

        distArrayL1 = []
        distArrayL2 = []
        for i in range(1, nRows):
            for j in range(0, i):

                p1 = inputMatrix[i, :]
                p2 = inputMatrix[j, :]

                distL1 = 0
                distL2 = 0
                for d in range(len(p1)):
                    distL1 = distL1 + abs(p1[d] - p2[d])
                    distL2 = distL2 + (p1[d] - p2[d]) ** 2
                distL2 = np.sqrt(distL2)
                distArrayL1.append(distL1)
                distArrayL2.append(distL2)

        di = np.unique(distArrayL1)
        ji = []
        for i in di:
            ji.append(distArrayL1.count(i))

        p = 1
        phi = 0

        for i in distArrayL1:
            phi = phi + i ** (-p)
        phi = phi ** (1 / p)

        # print('D0 = ',di[0])
        # print('J0 = ',ji[0])
        # print('Phi = ',phi)
        print '*** L1 Distance ***'
        print 'Mean = ',np.mean(distArrayL1)
        print 'Max = ', max(distArrayL1)
        print 'Min = ', min(distArrayL1)
        print '*** L2 Distance ***'
        print 'Mean = ', np.mean(distArrayL2)
        print 'Max = ', max(distArrayL2)
        print 'Min = ', min(distArrayL2)

        return np.mean(distArrayL1), np.mean(distArrayL2)

    # **************************************************************************************************************** #
    def getInputDistributionTypes(self):
        """
        :return:
            List of all supported distribution
        """
        inputDistributionsList = pd.DataFrame(columns=["Name","Type","Keyword","Parameters","Default"])
        inputDistributionsList["Name"] = ["Uniform","Normal"]
        inputDistributionsList["Type"] = ["Continuous","Continuous"]
        inputDistributionsList["Keyword"] = ["uniform","norm"]
        inputDistributionsList["Parameters"] = ["['uniform',minValue,maxValue]","['norm',loc,scale]"]
        inputDistributionsList["Defaults"] = ["['uniform',0,1]","['norm',0,1]"]


        self.inputDistributionsList = inputDistributionsList
        return self.inputDistributionsList

    # **************************************************************************************************************** #
    def getInputDistributionFunction(self,params):
        """
        This create a ppf function for the specified distribution and return it, this can then be multiplied with
        the matrix column to convert it.
        :param params:
            each param should be of format ['type',p1,p2..] p1,p2 corresponds to the input of 'type' of distribution
            you select.
        :return:
            ppf function.
        """

        # Default Function
        ppfFunction = import_module('scipy.stats.distributions').uniform().ppf

        type = params[0]
        inputDistributionsList = self.inputDistributionsList

        # If type of distribution is not amoung the support return unit ppf.
        if not(type in inputDistributionsList["Keyword"].tolist()):
            return ppfFunction

        moduleName = 'scipy.stats.distributions'
        distModule = import_module(moduleName)

        # Depending upon the distribution create the ppf function.
        if type == 'uniform':
            if len(params) == 3:
                print 'Uniform Distribution Applied with min:' + str(params[1]) + ' and max :' + str(params[2])
                ppfFunction = distModule.uniform(params[1], params[2] - params[1]).ppf
        if type == 'norm':
            if len(params) == 3:
                print 'Uniform Distribution Applied with mean:' + str(params[1]) + ' and std :' + str(params[2])
                ppfFunction = distModule.norm(loc=params[1], scale=params[2]).ppf
            else:
                print 'Uniform Distribution Applied with mean: 0 and std : 1'
                ppfFunction = distModule.norm().ppf

        return ppfFunction

    # **************************************************************************************************************** #
    def applyDistributionToMatrix(self,inMatrix,inDistribution):
        """
        Takes in the input unitMatrix and multiple with the respective distribution given by inDistribution.
        :param inMatrix:
        input unit hypercube matrix
        :param inDistribution:
            list of list each sublist if cooresponding the the ith input and contains the necessary information
            about the input
        :return:
            return the updated matrix wrt input distribution provided.
        """
        
        firstRow = inMatrix[0,:]
        numericCols = [index for index,value in enumerate(firstRow) if type(value) != str]                
        nInputs = len(numericCols)
        if np.min(inMatrix[:,numericCols]) < 0.0 or np.max(inMatrix[:,numericCols]) >1.0:
            errMsg = 'Input Matrix should be a unit matrix (with values between 0-1)'
            return inMatrix
        # Length of inDistribution should be equal to the no of inputs.
        if len(inDistribution) != nInputs:
            errMsg = "No of Distribution does not match the no of inputs."
            return inMatrix

        outMatrix = inMatrix.copy()
        for index,num_col in enumerate(numericCols):
            ppfFunction = self.getInputDistributionFunction(inDistribution[index])
            outMatrix[:,num_col] = ppfFunction(np.array(outMatrix[:,num_col],dtype='float64'))

        return outMatrix


    # **************************************************************************************************************** #
    def sliceDOE(self,inMatrix):
        """
            This function takes the inMatrix seperates continuous variables and discrete variables.
            For each unique value of discrete columns stores the subset matrix of continuous variables in a dictionary
            and returns it.
            NOTE: Discrete columns is identified if they have string value.
            Columns with numeric value 1,2,3 will not be treated ad discrete. It has to be '1','2','3'

        :param inMatrix:
            Input matrix containing discrete variables which needs to be sliced.

        :return:
            dictionary containing slices wrt each discrete value
        """

        nCols = len(inMatrix[0, :])  # No of Columns of data in the given matrix

        discreteColIdx = []  # Stores the index of discrete columns
        contColIdx = []  # Stores the index of continuous columns

        # Loop Over each Cols and find out which are discrete.
        # Identification of discrete is that first value should be a string.
        # Better logic can be implemented but not in need of it right now.
        for i in range(nCols):

            # Column data.
            colData = inMatrix[0, i]  # first value
            if type(inMatrix[0, i]) == str:  # if string type set as discrete column
                discreteColIdx.append(i)
            else:
                contColIdx.append(i)  # Else continous column

        slicedDOE = {}
        if len(discreteColIdx) <=0 :
            print 'No Discrete columns found to slice'
            return slicedDOE

        # For each Discrete column slice the matrix and store it in dictionary.
        for dCol in discreteColIdx:
            colData = np.array(inMatrix[:, dCol])
            uniqVals = np.unique(colData)
            for uniqVal in uniqVals:
                rowIds, = np.where(colData == uniqVal)
                slicedDOE[uniqVal] = inMatrix[rowIds, :][:, contColIdx]

        return slicedDOE

    # **************************************************************************************************************** #
    def plotMatrix(self,inputMatrix,addGrid=False,nNewPoints = None,figsize=(20,20)):
        """
            Function to be used to view the DOE distribution in 2D plane (wrt each column pairs)
        :param inputMatrix: Input Matrix which needs to be plotted. In case of sliced DOE it should be dictionary
            containing matrix for each slice.
        :param addGrid: Add grid only works for unit hypercube. Set true to add it
        :param nNewPoints: Setting this to numeric value will color the n points from the end of the array to red color
        :param figsize: provide a tuple for plot size.
        :return: None

        """

        def getXYPairs(nParams):
            xy_pair = []
            for i in range(1, nParams):
                for j in range(0, i):
                    xy_pair.append([i, j])
            return xy_pair

        if type(inputMatrix) == dict:
            nSlices = len(inputMatrix.keys())
            nParams = len(inputMatrix[inputMatrix.keys()[0]][0, :])

        else:
            nSlices = 1
            curMatrix = inputMatrix
            nParams = len(inputMatrix[0, :])

        if type(figsize) == tuple:
            plotSize =  figsize
        else:
            plotSize = (20,20)

        f, axarr = plt.subplots(nParams * nSlices, nParams, figsize=plotSize, sharex='col', sharey='row')
        for slice in range(nSlices):

            if type(inputMatrix) == dict:
                curMatrix = inputMatrix[inputMatrix.keys()[slice]]

            # Drop if any discrete columns exists
            nParams = len(curMatrix[0, :])  # No of Columns of data in the given matrix
            nSamples = len(curMatrix[:, 0])
            discreteColIdx = []  # Stores the index of discrete columns
            contColIdx = []  # Stores the index of continuous columns
            if type(nNewPoints) == int and nNewPoints<nSamples:
                color = ['blue'] * (nSamples-nNewPoints) + ['red'] *nNewPoints
            else:
                color = ['blue'] * nSamples

            # Loop Over each Cols and find out which are discrete.
            # Identification of discrete is that first value should be a string.
            # Better logic can be implemented but not in need of it right now.
            for i in range(nParams):

                # Column data.
                if type(curMatrix[0, i]) == str:  # if string type set as discrete column
                    discreteColIdx.append(i)
                else:
                    contColIdx.append(i)  # Else continous column

            curMatrix = curMatrix[:,contColIdx]

            xyPairs = getXYPairs(nParams)

            for index, xy in enumerate(xyPairs):
                x, y = xy
                # Do not get confused here.
                # x,y represent row,col in plot : but with respect to curMatrix they are swapped =>
                # this is because x-axis all rows in a given columns and vice verse
                # if its still confusing print the below line to check.
                #print 'Row=',x+1,'Column=',y+1,'Xdata=',y+1,'Ydata=',x+1
                pltXid = slice*nParams + x
                axarr[pltXid, y].scatter(curMatrix[:, y], curMatrix[:, x],color=color)
                if addGrid:
                    axarr[pltXid, y].set_yticks(np.round(np.arange(0, 1, 1.0 / nSamples),decimals=3), minor=False)
                    axarr[pltXid, y].set_xticks(np.round(np.arange(0, 1, 1.0 / nSamples),decimals=3), minor=False)
                    axarr[pltXid, y].grid()

            # Hiding the other plots
            xyPairsHide = xyPairs
            xyDiag = np.diag_indices(nParams)
            for ind in xyDiag[0]:
                xyPairsHide.append([ind,ind])

            # Set the axis to off
            for y,x in xyPairsHide:
                pltXid = slice * nParams + x
                axarr[pltXid, y].axis('off')

            for ind in range(nParams):
                pltXid = slice * nParams + ind
                axarr[pltXid, 0].set_ylabel('Param-' + str(ind + 1))
                axarr[-1, ind].set_xlabel('Param-' + str(ind + 1))

        f.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.show()

    # **************************************************************************************************************** #
    def addNpointsLHS(self,inMatrix, m=1, method='random'):
        """
        Add 'n' new points to the existing LHS matrix.
        NOTE:
            Assumes inMatrix to be a unit Hypercube i.e. all parameters are from 0-1
        :param m:
            How many new points do you want.
        :param method:
            'random' : choose a random value inside the bin
            'center' : choose the mid point of the bin
        :return:
        """
        # Validations.
        if not method in ['random', 'center']:
            print 'Selected Method is invalid'
            return inMatrix
        if type(m) != int:
            print '"m" should be of type Integer'
            return inMatrix
        if m <= 0:
            print '"m" should be greater than 0'
            return inMatrix

        nRows = len(inMatrix[:, 0])
        nCols = len(inMatrix[0, :])

        # These are basically columnid/rowid taken at random.
        colVec = np.argsort(np.random.uniform(size=nCols))
        rowVec = np.argsort(np.random.uniform(size=nRows + m))

        # This new matrix will hold the new points.
        outMatrix = np.empty([nRows + m, nCols])
        outMatrix[:] = np.nan

        # Loop over each parameter at a time.
        for colId in colVec:
            newRow = -1

            # Loop over each bin.
            for rowId in rowVec:

                # Lower/Upper bound value of the new Bin
                lowerBound = float(rowId) / (nRows + m)
                upperBound = float(rowId + 1) / (nRows + m)

                # Compare all given point for the current paramater.
                # If there is no existing point in the new bin range then we add a new point.
                gtLowerBound = lowerBound <= inMatrix[:, colId]
                ltUpperBound = upperBound > inMatrix[:, colId]
                if not any(np.logical_and(gtLowerBound, ltUpperBound)):
                    newRow = newRow + 1

                    if method == 'random':
                        newPoint = np.random.uniform(low=lowerBound, high=upperBound, size=1)
                    else:
                        newPoint = lowerBound+ (upperBound-lowerBound)/2
                    outMatrix[newRow, colId] = newPoint

        # Drop All NANs
        outMatrix = outMatrix[~np.isnan(outMatrix).any(axis=1), :]
        print outMatrix
        if len(outMatrix)>=m:
            for i in range(len(outMatrix[0,:])):
                np.random.shuffle(outMatrix[:, i])
            print outMatrix
            print len(outMatrix)
            outMatrix = np.concatenate((inMatrix, outMatrix[:m]), axis=0)
        else:
            print 'Could not generate any new points'
            print outMatrix
            outMatrix = inMatrix

        # return
        return outMatrix

# ******************************************************************************************************************** #
class pyDOEMaster(doeMaster):
    """
        Child Class of doeMaster, for python Package pyDOE
    """

    # **************************************************************************************************************** #
    def __init__(self):
        """
        Initialize the pyDOEMaster Class
        """
        self.inModule = import_module('pyDOE')
        super(pyDOEMaster, self).__init__()

    # **************************************************************************************************************** #
    def lhs(self,nInputs=None,nSamples=None, inputDistribution = None,criterion = 'corr'):
        """

        :param nInputs:
            No of inputs/Parameters for creating the LHS
        :param nSamples:
            No of sample points for creating the LHS
        :param inputDistribution:
            Default 'None' : all input are create in range of 0-1 [unit Hypercube]
            You can also choose different input distributions example [['uniform',0,10]] ranging from 0-10 with equal probability.
            Check function getInputDistributionTypes() to find all options
        :param criterion:
            default : 'corr'
            'c': center the points within the sampling intervals
            'm': maximize the minimum distance between points, but place the point in a randomized location within its interval
            'cm': same as 'm' but centered within the intervals
            'corr': minimize the maximum correlation coefficient
        :return:
            return the lhs matrix
        """

        # Import Function Responsible for Creating DOE.
        doeFunction = self.inModule.lhs

        # Validation Steps.
        # N-Inputs and N-Samples have to be validated.
        if nInputs == None:
            nInputs = self.nInputs

        if nSamples == None:
            nSamples = self.nSamples

        if type(nInputs) != int or type(nSamples) !=int:
            print "N-Inputs/N-Samples should be specified numeric values"
            return []
        elif nInputs <=0 or nSamples <=0:
            print "N-Inputs/N-Samples should be greater than 0"
            return []
        if inputDistribution != None:
            if type(inputDistribution[0]) != list:
                print 'inputDistribution should be of type list of list, example [["uniform,0,1"]]'
                return []

        self.nInputs = nInputs
        self.nSamples = nSamples

        # Validate all Parameter which goes into the function.
        valid_params = ['c','m','cm','corr']
        if not (criterion in valid_params):
            print 'Invalid Value for "criterion", Valid values are : ' + valid_params
            return []

        outMatrix = doeFunction(nInputs,samples=nSamples,criterion=criterion)

        if inputDistribution != None:
                outMatrix = self.applyDistributionToMatrix(outMatrix,inputDistribution)

        return outMatrix

# ******************************************************************************************************************** #
class diversipyMaster(doeMaster):
    """
        Child Class of doeMaster, for python Package diversipy
    """

    # **************************************************************************************************************** #
    def __init__(self):
        """
        Initilize diversipyMaster Class
        """
        self.inModule = import_module('diversipy')
        super(diversipyMaster, self).__init__()

    # **************************************************************************************************************** #
    def convertCoordinateToUnitHypercube(self,inMatrix,method='center'):
        """
            diversipyMaster libraries always return the LHS as coordinate based values.
            We need to convert then to actual values from 0-1.
        :param inMatrix:
            inMatrix is lhs matrix with coordinates corresponding to the bins
        :param method:
            For each bin how should the point be selected. You have 3 options:
            1. 'center' : select the mid point of each bin
            2. 'random' : select random point inside each bin
            3. 'max_distance' : select the point such that all bins are far apart.
        :return:
            matrix with updated values.
        """

        # Convert the matrix into a unit hypercube based on selected method.
        # Converting to unit hypercube is must to multiple with any given input distribution.
        outMatrix = inMatrix
        if method == 'center':
            # 1. 'center' : select the mid point of each bin
            outMatrix = self.inModule.centered_lhs(outMatrix)
        elif method == 'random':
            # 2. 'random' : select random point inside each bin
            outMatrix = self.inModule.perturbed_lhs(outMatrix)
        elif method == 'max_distance':
            # 3. 'max_distance' : select the point such that all bins are far apart.
            outMatrix = self.inModule.edge_lhs(outMatrix)

        return outMatrix

    # **************************************************************************************************************** #
    def lhd_matrix(self,nInputs=None,nSamples=None, inputDistribution = None,selectPointMethod = 'center'):
        """
            Calls the lhs_matrix function from diversipy package to create the LHS
            :param nInputs:
            No of inputs/Parameters for creating the LHS
        :param nSamples:
            No of sample points for creating the LHS
        :param inputDistribution:
            Default 'None' : all input are create in range of 0-1 [unit Hypercube]
            You can also choose different input distributions example [['uniform',0,10]] ranging from 0-10 with equal probability.
            Check function getInputDistributionTypes() to find all options
        :param selectPointMethod:
            For each bin how should the point be selected. You have 3 options:
            1. 'center' : select the mid point of each bin
            2. 'random' : select random point inside each bin
            3. 'max_distance' : select the point such that all bins are far apart.

        :return:
            return the matrix of LHS create by the method.
        """

        # Import Function Responsible for Creating DOE.
        doeFunction = self.inModule.lhd_matrix

        # Validation Steps.
        # N-Inputs and N-Samples have to be validated.
        if nInputs == None:
            nInputs = self.nInputs

        if nSamples == None:
            nSamples = self.nSamples

        if type(nInputs) != int or type(nSamples) != int:
            print "N-Inputs/N-Samples should be specified numeric values"
            return []
        elif nInputs <=0 or nSamples <=0:
            print "N-Inputs/N-Samples should be greater than 0"
            return []

        # Validate other input parameters
        if not (selectPointMethod in ['center', 'random', 'max_distance']):
            print "selectPointMethod should be either 'center'/'random'/'max_distance'"
            return []
        if inputDistribution != None:
            if type(inputDistribution[0]) != list:
                print 'inputDistribution should be of type list of list, example [["uniform,0,1"]]'
                return []

        # Set the class object cause they survived validation
        self.nInputs = nInputs
        self.nSamples = nSamples

        # calls the doe creation function and create the matrix
        outMatrix = doeFunction(nSamples,nInputs )

        # The resultant matrix from about step have values of bins, we need to create a unit Hypercube.
        outMatrix = self.convertCoordinateToUnitHypercube(outMatrix,method = selectPointMethod)

        # Convert the unit matrix according to the input distribution
        if inputDistribution != None:
            outMatrix = self.applyDistributionToMatrix(outMatrix, inputDistribution)

        # Return output matrix
        return outMatrix

    # **************************************************************************************************************** #
    def improved_lhd_matrix(self, nInputs=None, nSamples=None, inputDistribution=None,
                            selectPointMethod = 'center', num_candidates=100, target_value=None):
        """
            calls the improved_lhs_matrix function from the diversipy library
        :param nInputs:
            No of inputs/Parameters for creating the LHS
        :param nSamples:
            No of sample points for creating the LHS
        :param inputDistribution:
            Default 'None' : all input are create in range of 0-1 [unit Hypercube]
            You can also choose different input distributions example [['uniform',0,10]] ranging from 0-10 with equal probability.
            Check function getInputDistributionTypes() to find all options
        :param selectPointMethod:
            For each bin how should the point be selected. You have 3 options:
            1. 'center' : select the mid point of each bin
            2. 'random' : select random point inside each bin
            3. 'max_distance' : select the point such that all bins are far apart.
        :param num_candidates:
            default is 100.
            The number of random candidates considered for every point to be added to the LHD
        :param target_value:
            default is None
            The distance a candidate should ideally have to the already chosen points of the LHD.
        :return:
            lhs Matrix
        """

        # Import Function Responsible for Creating DOE.
        doeFunction = self.inModule.improved_lhd_matrix

        # Validation Steps.
        # N-Inputs and N-Samples have to be validated.
        if nInputs == None:
            nInputs = self.nInputs

        if nSamples == None:
            nSamples = self.nSamples

        if type(nInputs) != int or type(nSamples) != int:
            print "N-Inputs/N-Samples should be specified numeric values"
            return []
        elif nInputs <=0 or nSamples <=0:
            print "N-Inputs/N-Samples should be greater than 0"
            return []

        # validate other parameters
        if type(num_candidates) !=int:
            print "num_candidates should be numeric"
            return []
        if target_value != None and type(target_value) != float:
            print "target_value should be float/None"
            return []
        if not (selectPointMethod in ['center','random','max_distance']):
            print "selectPointMethod should be either 'center'/'random'/'max_distance'"
            return []

        if inputDistribution != None:
            if type(inputDistribution[0]) != list:
                print 'inputDistribution should be of type list of list, example [["uniform,0,1"]]'
                return []

        # assign to class variables cause they survived validation
        self.nInputs = nInputs
        self.nSamples = nSamples

        # Create the LHS matrix by calling the respective fucntions.
        outMatrix = doeFunction(nSamples,nInputs ,num_candidates= num_candidates,target_value= target_value,
                                dist_matrix_function=None)

        # The matrix created from the above method return the values as bin coordinates, convert then to unit Hypercube.
        outMatrix = self.convertCoordinateToUnitHypercube(outMatrix, method=selectPointMethod)

        # Convert the unit hypercube to select input distribution.
        if inputDistribution != None:
            outMatrix = self.applyDistributionToMatrix(outMatrix, inputDistribution)

        # Return the output matrix.
        return outMatrix

    def slicedLHS(self, nInputs=None, nSamples=None, nSlice=2, inputDistribution=None,selectPointMethod = 'center',
                  num_candidates=100, target_value=None):
        """
            This Method creates sliced LHS using diversipy liraries

        :param nInputs:
            No of inputs/Parameters for continuous type to be created
        :param nSamples:
            No of sample points per sliced LHS
        :param nSlice:
            No of Slices you want to create.
        :param inputDistribution:
            Default 'None' : all input are create in range of 0-1 [unit Hypercube]
            You can also choose different input distributions example [['uniform',0,10]] ranging from 0-10 with equal probability.
            Check function getInputDistributionTypes() to find all options
        :param selectPointMethod:
            For each bin how should the point be selected. You have 3 options:
            1. 'center' : select the mid point of each bin
            2. 'random' : select random point inside each bin
            3. 'max_distance' : select the point such that all bins are far apart.
        :param num_candidates:
            default is 100.
            The number of random candidates considered for every point to be added to the LHD
        :param target_value:
            default is None
            The distance a candidate should ideally have to the already chosen points of the LHD.
        :return:
            A dictionary with sliced LHS matrix
            examples {'Slice_1': matrix,'Slice_2': matrix ...}
        """

        # Input validation
        if type(nSlice) != int:
            print 'nSlice should be of type int'
            return []
            if nSlice <= 1:
                print 'nSlice should be greater than 1'
                return []
        if nSamples == None:
            nSamples = self.nSamples

        if type(nSamples) != int:
            print "N-Samples should be specified numeric values"
            return []
        elif nSamples <= 0:
            print "N-Samples should be greater than 0"
            return []

        if inputDistribution != None:
            if type(inputDistribution[0]) != list:
                print 'inputDistribution should be of type list of list, example [["uniform,0,1"]]'
                return []

        lhsMatrix = self.improved_lhd_matrix(nInputs=nInputs, nSamples=nSamples * nSlice, inputDistribution=None,
                        selectPointMethod=selectPointMethod, num_candidates=num_candidates, target_value=target_value)

        sliceMatrix = {}
        lhsMatrixDub = lhsMatrix
        for i in range(1, nSlice+1):
            if len(lhsMatrixDub[:, 0]) > nSamples:
                curSlice = self.inModule.psa_select(lhsMatrixDub, nSamples)
                sliceMatrix['Slice_' + str(i)] = curSlice
                mainMatrix = set([tuple(row) for row in lhsMatrixDub])
                curSliceMatrix = set([tuple(row) for row in curSlice])
                lhsMatrixDub = np.array(list(mainMatrix.symmetric_difference(curSliceMatrix)))
                if inputDistribution !=None:
                    curSlice = self.applyDistributionToMatrix(curSlice,inputDistribution)
            else:
                curSlice = lhsMatrixDub
                if inputDistribution !=None:
                    curSlice = self.applyDistributionToMatrix(curSlice,inputDistribution)
                sliceMatrix['Slice_' + str(i)] = curSlice

        # Return the sliced matrix
        return sliceMatrix

if __name__ == '__main__':
    obj = doeMaster()
    lhsMatrix = obj.setPackage('pyDOE').lhs(nSamples=10, nInputs=2)
    
