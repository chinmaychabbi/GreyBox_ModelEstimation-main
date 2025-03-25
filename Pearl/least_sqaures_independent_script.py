## Importing external modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from casadi import *
import pickle
import pandas as pd
import tensorflow as tf
import datetime
import math
import time
import tracemalloc
import copy
import scipy


# =============================================================================
# User Inputs
# =============================================================================
Type_Data_Source = 2  # 1 - PNNNL Prototypes , 2 - Simulated 4State House Model

Type_System_Model = 2 # 1 - One State Model , 2 - Two State Model , 4 - Four State Model

Shoter_ResultsPath = True  # True - Results are stored in shorter path , False - Results are stored in the appropriate directory

Short_ResultsFolder = 'Results_1'

DateTime_Tuple = ('08/01/2017', '08/07/2017')

FileResolution_Minutes = 10

Theta_Initial = [ 0.05, 0.05, 10000000, 10000000, 0.5, 0.5]  # [R_zw, R_wa, C_z, C_w, A_1, A_2]

# Plot Size parameters
Plot_Width = 15
Plot_Height = 10

## Data Source dependent User Inputs
if (Type_Data_Source  == 1):  # PNNL Prototype Data

    Simulation_Name = "test1"

    Total_Aggregation_Zone_Number = 5

    ## User Input: Aggregation Unit Number ##
    # Aggregation_UnitNumber = 1
    # Aggregation_UnitNumber = 2

    # Aggregation Zone NameStem Input
    Aggregation_Zone_NameStem = 'Aggregation_Zone'

    ## Providing Proper Extensions depending on Type of Filter/Smoother Utilized and IMPROVEMENT is needed wrt type of data and model type (NINAD'S WORK)
    
    GBModel_Key = 'LS_DS_B' + '_SSM_' + str(Type_System_Model)

elif (Type_Data_Source  == 2):  # Simulated 4 State House Thermal Model Data

    Simulated_HouseThermalData_Filename = "PecanStreet_Austin_NSRDB_Gainesville_2017_Fall_3Months.mat"

    ## Providing Proper Extensions depending on Type of Filter/Smoother Utilized and IMPROVEMENT is needed wrt type of data and model type (NINAD'S WORK)
    GBModel_Key = 'LS_DS_H' + '_SSM_' + str(Type_System_Model)

## Basic Computation

# Computing ts in seconds
ts = FileResolution_Minutes*60

#training folder name
Training_FolderName = GBModel_Key

# =============================================================================
# Initialization
# =============================================================================
 
PHVAC = np.zeros((1,1)) #power consumed by AC
 
PHVAC1 = np.zeros((1,1))
 
 
# =============================================================================
# Data Access: Dependent on Data Source
# =============================================================================
 
for kk in range(Total_Aggregation_Zone_Number):
 
    kk = kk + 1
 
    Aggregation_UnitNumber = kk
 
    print("Current Unit Number: " + str(kk))
 
    if (Type_Data_Source  == 1):  # PNNL Prototype Data
 
        # =============================================================================
        # Getting Required Data from Sim_ProcessedData
        # =============================================================================
 
        # Getting Current File Directory Path
        Current_FilePath = os.path.dirname(__file__)
 
        # Getting Folder Path
        Sim_ProcessedData_FolderPath_AggregatedTestTrain = os.path.join(Current_FilePath, '..', '..', '..', 'Results',
                                                                        'Processed_BuildingSim_Data', Simulation_Name,
                                                                        'Sim_TrainingTestingData')
        Sim_ProcessedData_FolderPath_Regression = os.path.join(Current_FilePath, '..', '..', '..', 'Results',
                                                            'Processed_BuildingSim_Data', Simulation_Name,
                                                            'Sim_RegressionModelData')
 
        # LOOP: Output Generation for Each Aggregated Zone
 
        # Creating Required File Names
 
        Aggregation_DF_Test_File_Name = 'Aggregation_DF_Test_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'
 
        Aggregation_DF_Train_File_Name = 'Aggregation_DF_Train_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'
 
        ANN_HeatInput_Test_DF_File_Name = 'ANN_HeatInput_Test_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'
 
        ANN_HeatInput_Train_DF_File_Name = 'ANN_HeatInput_Train_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'
 
        PHVAC_Regression_Model_File_Name = 'QAC_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk)
 
        # Get Required Files from Sim_AggregatedTestTrainData_FolderPath    
        AggregatedTest_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Test_File_Name), "rb")
        AggregatedTest_DF = pickle.load(AggregatedTest_Dict_File)
 
        AggregatedTrain_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Train_File_Name), "rb")
        AggregatedTrain_DF = pickle.load(AggregatedTrain_Dict_File)
 
        PHVAC_Regression_Model_File_Path = os.path.join(Sim_ProcessedData_FolderPath_Regression, PHVAC_Regression_Model_File_Name)
        PHVAC_Regression_Model = tf.keras.models.load_model(PHVAC_Regression_Model_File_Path)
 
        # Get Required Files from Sim_RegressionModelData_FolderPath
        ANN_HeatInput_Test_DF_File = open(os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Test_DF_File_Name),
                                            "rb")
        ANN_HeatInput_Test_DF = pickle.load(ANN_HeatInput_Test_DF_File)
 
        ANN_HeatInput_Train_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Train_DF_File_Name), "rb")
        ANN_HeatInput_Train_DF = pickle.load(ANN_HeatInput_Train_DF_File)
 
       
        # =============================================================================
        # Basic Computation
        # =============================================================================
 
        # Getting DateTime Data
        DateTime_Train = AggregatedTrain_DF['DateTime']
        DateTime_Test = AggregatedTest_DF['DateTime']
 
        # Resetting
        ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
 
        # Joining Train/Test Aggregated_DF
        Aggregated_DF = pd.concat([AggregatedTrain_DF.reset_index(drop=True), AggregatedTest_DF.reset_index(drop=True)])
        Aggregated_DF.sort_values("DateTime", inplace=True)
        Aggregated_DF.reset_index(drop=True, inplace=True)
 
        # Joining Train/Test ANN_HeatInput_DF
        ANN_HeatInput_Train_DF.insert(0, 'DateTime', DateTime_Train)
        ANN_HeatInput_Test_DF.insert(0, 'DateTime', DateTime_Test)
        ANN_HeatInput_DF = pd.concat([ANN_HeatInput_Train_DF.reset_index(drop=True), ANN_HeatInput_Test_DF.reset_index(drop=True)])
        ANN_HeatInput_DF.sort_values("DateTime", inplace=True)
        ANN_HeatInput_DF.reset_index(drop=True, inplace=True)
 
        # Getting Data within Start and End Dates provided by user
        # Add DateTime to Aggregation_DF
        DateTime_List = Aggregated_DF['DateTime'].tolist()
 
        # Getting Start and End Dates for the Dataset
        StartDate_Dataset = DateTime_List[0]
        EndDate_Dataset = DateTime_List[-1]
 
        # Getting the File Resolution from DateTime_List
        DateTime_Delta = DateTime_List[1] - DateTime_List[0]
 
        FileResolution_Minutes = DateTime_Delta.seconds/60
 
            # Getting Start and End Date
        StartDate = datetime.datetime.strptime(DateTime_Tuple[0],'%m/%d/%Y')
        EndDate = datetime.datetime.strptime(DateTime_Tuple[1],'%m/%d/%Y')
 
        # User Dates Corrected
        StartDate_Corrected = datetime.datetime(StartDate.year,StartDate.month,StartDate.day,0,int(FileResolution_Minutes),0)
        EndDate_Corrected = datetime.datetime(EndDate.year,EndDate.month,EndDate.day,23,60-int(FileResolution_Minutes),0)
 
        Counter_DateTime = -1
 
        # Initializing DateRange List
        DateRange_Index = []
 
        # FOR LOOP:
        for Element in DateTime_List:
            Counter_DateTime = Counter_DateTime + 1
 
            if (Element >= StartDate_Corrected and Element <= EndDate_Corrected):
                DateRange_Index.append(Counter_DateTime)
 
        # Getting Train and Test Dataset
        Aggregated_DF = copy.deepcopy(Aggregated_DF.iloc[DateRange_Index,:])
        ANN_HeatInput_DF = copy.deepcopy(ANN_HeatInput_DF.iloc[DateRange_Index,:])
 
        Aggregated_DF.reset_index(drop=True, inplace=True)
        ANN_HeatInput_DF.reset_index(drop=True, inplace=True)
 
 
        # Computing QZic and QZir Train
 
        # Initialization
        QZic_Train = []
        QZir_Train = []
        QZic_Test = []
        QZir_Test = []
        QSol1_Test = []
        QSol1_Train = []
        QSol2_Test = []
        QSol2_Train = []
        QAC_Test = []
        QAC_Train = []
 
        # FOR LOOP: Getting Summation
        for ii in range(ANN_HeatInput_DF.shape[0]):
            # print(ii)
            QZic_Train_1 = ANN_HeatInput_DF['QZic_P'][ii][0] + ANN_HeatInput_DF['QZic_L'][ii][0] + \
                            ANN_HeatInput_DF['QZic_EE'][ii][0]
            QZir_Train_1 = ANN_HeatInput_DF['QZir_P'][ii][0] + ANN_HeatInput_DF['QZir_L'][ii][0] + \
                            ANN_HeatInput_DF['QZir_EE'][ii][0] + ANN_HeatInput_DF['QZivr_L'][ii][0]
            QZic_Train.append(QZic_Train_1)
            QZir_Train.append(QZir_Train_1)
 
            QSol1_Train_1 = ANN_HeatInput_DF['QSol1'][ii][0]
            QSol2_Train_1 = ANN_HeatInput_DF['QSol2'][ii][0]
            # QAC_Train_1 = ANN_HeatInput_DF['QAC'][ii][0]
            QAC_Train_1 = Aggregated_DF['QHVAC_X'].iloc[ii]
 
            QSol1_Train.append(QSol1_Train_1)
            QSol2_Train.append(QSol2_Train_1)
            QAC_Train.append(QAC_Train_1)
 
        ANN_HeatInput_DF.insert(2, 'QZic', QZic_Train)
        ANN_HeatInput_DF.insert(2, 'QZir', QZir_Train)
        ANN_HeatInput_DF.insert(2, 'QSol1_Corrected', QSol1_Train)
        ANN_HeatInput_DF.insert(2, 'QSol2_Corrected', QSol2_Train)
        ANN_HeatInput_DF.insert(2, 'QAC_Corrected', QAC_Train)
 
        ANN_HeatInput_DF.reset_index(drop=True, inplace=True)
        Aggregated_DF.reset_index(drop=True, inplace=True)
 
        # Creating Common DF
        Common_Data_DF = pd.concat([Aggregated_DF[[ 'DateTime', 'Zone_Air_Temperature_', 'Site_Outdoor_Air_Drybulb_Temperature_']], ANN_HeatInput_DF[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir', 'QAC_Corrected']]], axis=1)
 
        # Creating Training and Testing Split
        Common_Data_DF_Train = Common_Data_DF.iloc[0:math.floor((1-Test_Split)*len(Common_Data_DF)),:]
       
        Test_Start_Index = math.floor((1-Test_Split)*len(Common_Data_DF)) + 1
 
        Common_Data_DF_Test = Common_Data_DF.iloc[Test_Start_Index:,:]
       
        # Debugger
        # plt.figure()
        # plt.plot(Common_Data_DF['QAC_Corrected'])
        #plt.show()
       
        ## Creating Y_Measurement and U_Measurement - Dependent on the Model Type
        if (Type_System_Model == 1):  # One State Model
 
            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((6,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = Common_Data_DF_Train['Zone_Air_Temperature_'].iloc[ii]
 
                # Getting Disturbances
                T_a = Common_Data_DF_Train['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[ii]
                Q_sol1 = Common_Data_DF_Train['QSol1_Corrected'].iloc[ii]
                Q_sol2 = Common_Data_DF_Train['QSol2_Corrected'].iloc[ii]
                Q_Zic = Common_Data_DF_Train['QZic'].iloc[ii]
                Q_Zir = Common_Data_DF_Train['QZir'].iloc[ii]
 
                # Getting Control
                Q_ac = Common_Data_DF_Train['QAC_Corrected'].iloc[ii]
 
                # Updating Y_Measurement
                Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))
 
                # Updating U_Measurement
                U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement
            Y_Measurement_Train = Y_Measurement_Train[:,1:]
            U_Measurement_Train = U_Measurement_Train[:,:-1]
 
            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((6,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = Common_Data_DF_Test['Zone_Air_Temperature_'].iloc[ii]
 
                # Getting Disturbances
                T_a = Common_Data_DF_Test['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[ii]
                Q_sol1 = Common_Data_DF_Test['QSol1_Corrected'].iloc[ii]
                Q_sol2 = Common_Data_DF_Test['QSol2_Corrected'].iloc[ii]
                Q_Zic = Common_Data_DF_Test['QZic'].iloc[ii]
                Q_Zir = Common_Data_DF_Test['QZir'].iloc[ii]
 
                # Getting Control
                Q_ac = Common_Data_DF_Test['QAC_Corrected'].iloc[ii]
 
                # Updating Y_Measurement
                Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))
 
                # Updating U_Measurement
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement
            Y_Measurement_Test = Y_Measurement_Test[:,1:]
            U_Measurement_Test = U_Measurement_Test[:,:-1]
 
        elif (Type_System_Model == 2):  # Two State Model
 
            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((6,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = Common_Data_DF_Train['Zone_Air_Temperature_'].iloc[ii]
 
                # Getting Disturbances
                T_a = Common_Data_DF_Train['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[ii]
                Q_sol1 = Common_Data_DF_Train['QSol1_Corrected'].iloc[ii]
                Q_sol2 = Common_Data_DF_Train['QSol2_Corrected'].iloc[ii]
                Q_Zic = Common_Data_DF_Train['QZic'].iloc[ii]
                Q_Zir = Common_Data_DF_Train['QZir'].iloc[ii]
 
                # Getting Control
                Q_ac = Common_Data_DF_Train['QAC_Corrected'].iloc[ii]
 
                # Updating Y_Measurement
                Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))
 
                # Updating U_Measurement
                U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement
            Y_Measurement_Train = Y_Measurement_Train[:,1:]
            U_Measurement_Train = U_Measurement_Train[:,:-1]
 
            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((6,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = Common_Data_DF_Test['Zone_Air_Temperature_'].iloc[ii]
 
                # Getting Disturbances
                T_a = Common_Data_DF_Test['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[ii]
                Q_sol1 = Common_Data_DF_Test['QSol1_Corrected'].iloc[ii]
                Q_sol2 = Common_Data_DF_Test['QSol2_Corrected'].iloc[ii]
                Q_Zic = Common_Data_DF_Test['QZic'].iloc[ii]
                Q_Zir = Common_Data_DF_Test['QZir'].iloc[ii]
 
                # Getting Control
                Q_ac = Common_Data_DF_Test['QAC_Corrected'].iloc[ii]
 
                # Updating Y_Measurement
                Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))
 
                # Updating U_Measurement
                U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement
            Y_Measurement_Test = Y_Measurement_Test[:,1:]
            U_Measurement_Test = U_Measurement_Test[:,:-1]    
   
 
    elif (Type_Data_Source  == 2):  # Simulated 4 State House Thermal Model Data
 
        # =============================================================================
        # Getting Required Data from Sim_ProcessedData
        # =============================================================================
 
        # Getting Current File Directory Path
        Current_FilePath = os.path.dirname(__file__)
 
        # Getting Folder Path
        Sim_HouseThermalData_FilePath = os.path.join(Current_FilePath, '..', '..', '..', 'Data',
                                                                        'HouseThermalModel_Data', 'PostProcessedFiles',
                                                                        'LargeHouse', Simulated_HouseThermalData_Filename)
       
 
 
        ## Read Dictionary from .mat File
        Common_Data_Dict = scipy.io.loadmat(Sim_HouseThermalData_FilePath)
 
        ## Reading ActuaL Data
        Common_Data_Array = Common_Data_Dict[Simulated_HouseThermalData_Filename.split('.')[0]]
 
        # =============================================================================
        # Basic Computation
        # =============================================================================
 
        # Getting Start and End Date
        StartDate = datetime.datetime.strptime(DateTime_Tuple[0],'%m/%d/%Y')
        EndDate = datetime.datetime.strptime(DateTime_Tuple[1],'%m/%d/%Y')
 
        # User Dates Corrected
        StartDate_Corrected = datetime.datetime(StartDate.year,StartDate.month,StartDate.day,0,int(FileResolution_Minutes),0)
        EndDate_Corrected = datetime.datetime(EndDate.year,EndDate.month,EndDate.day,23,60-int(FileResolution_Minutes),0)
 
        Counter_DateTime = -1
 
        # Initializing DateRange List
        DateRange_Index = []
 
        # FOR LOOP:
        for Element in DateTime_List:
            Counter_DateTime = Counter_DateTime + 1
 
            if (Element >= StartDate_Corrected and Element <= EndDate_Corrected):
                DateRange_Index.append(Counter_DateTime)
 
        # Getting Train and Test Dataset
        Common_Data_Array_DateCorrected = Common_Data_Array[DateRange_Index,:]
 
        # Creating Training and Testing Split
        Common_Data_Array_DateCorrected_Train = Common_Data_Array_DateCorrected[0:math.floor((1-Test_Split)*Common_Data_Array_DateCorrected.shape[0]),:]
       
        Test_Start_Index = math.floor((1-Test_Split)*Common_Data_Array_DateCorrected.shape[0]) + 1
 
        Common_Data_Array_DateCorrected_Test = Common_Data_Array_DateCorrected[Test_Start_Index:,:]
 
        ## Reading DateTime/States(k+1)/States(k)/Input(k) - Training
        DateTime_Array_Train = Common_Data_Array_DateCorrected_Train[:,0:4]
        States_k_1_Array_Train = Common_Data_Array_DateCorrected_Train[:,4:8]
        States_k_Array_Train = Common_Data_Array_DateCorrected_Train[:,8:12]
        Inputs_k_Array_Train = Common_Data_Array_DateCorrected_Train[:,12:16]
 
        ## Reading DateTime/States(k+1)/States(k)/Input(k) - Testing
        DateTime_Array_Test = Common_Data_Array_DateCorrected_Test[:,0:4]
        States_k_1_Array_Test = Common_Data_Array_DateCorrected_Test[:,4:8]
        States_k_Array_Test = Common_Data_Array_DateCorrected_Test[:,8:12]
        Inputs_k_Array_Test = Common_Data_Array_DateCorrected_Test[:,12:16]        
 
        ## Creating Y_Measurement and U_Measurement - Dependent on the Model Type
        if (Type_System_Model == 1):  # One State Model
 
            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((6,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = States_k_1_Array_Train[ii,1]
 
                # Getting Disturbances
                T_a = Inputs_k_Array_Train[ii,0]
                Q_in = Inputs_k_Array_Train[ii,3]
                Q_ac = Inputs_k_Array_Train[ii,4]
                Q_venti = Inputs_k_Array_Train[ii,5]
                Q_infil = Inputs_k_Array_Train[ii,6]
                Q_sol = Inputs_k_Array_Train[ii,7]
 
                # # Updating Y_Measurement
                # Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))
 
                # # Updating U_Measurement
                # U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Train = Y_Measurement_Train[:,1:]
            # U_Measurement_Train = U_Measurement_Train[:,:-1]
 
            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((6,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = States_k_1_Array_Test[ii,1]
 
                # Getting Disturbances #add according to state
                T_a = Inputs_k_Array_Test[ii,0]
                Q_in = Inputs_k_Array_Test[ii,3]
                Q_ac = Inputs_k_Array_Test[ii:,4]
                Q_venti = Inputs_k_Array_Test[ii,5]
                Q_infil = Inputs_k_Array_Test[ii,6]
                Q_sol = Inputs_k_Array_Test[ii,7]
 
                # Updating Y_Measurement
                # Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))
 
                # # Updating U_Measurement
                # U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Test = Y_Measurement_Test[:,1:]
            # U_Measurement_Test = U_Measurement_Test[:,:-1]
 
        elif (Type_System_Model == 2):  # Two State Model
 
            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((7,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = States_k_1_Array_Train[ii,1]
 
                # Getting Disturbances
                T_sol_w = Inputs_k_Array_Train[ii,1]
                T_a = Inputs_k_Array_Train[ii,0]
                Q_in = Inputs_k_Array_Train[ii,3]
                Q_ac = Inputs_k_Array_Train[ii,4]
                Q_venti = Inputs_k_Array_Train[ii,5]
                Q_infil = Inputs_k_Array_Train[ii,6]
                Q_sol = Inputs_k_Array_Train[ii,7]
 
                # Updating Y_Measurement
                # Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))
 
                # # Updating U_Measurement
                # U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_sol_w, T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Train = Y_Measurement_Train[:,1:]
            # U_Measurement_Train = U_Measurement_Train[:,:-1]
 
            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((7,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = States_k_1_Array_Test[ii,1]
 
                # Getting Disturbances
                T_sol_w = Inputs_k_Array_Test[ii,1]
                T_a = Inputs_k_Array_Test[ii,0]
                Q_in = Inputs_k_Array_Test[ii,3]
                Q_ac = Inputs_k_Array_Test[ii,4]
                Q_venti = Inputs_k_Array_Test[ii,5]
                Q_infil = Inputs_k_Array_Test[ii,6]
                Q_sol = Inputs_k_Array_Test[ii,7]
 
                # Updating Y_Measurement
                # Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))
 
                # # Updating U_Measurement
                # U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_sol_w, T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Test = Y_Measurement_Test[:,1:]
            # U_Measurement_Test = U_Measurement_Test[:,:-1]
 
 
        elif (Type_System_Model == 4):  # Four State Model #Pandas data frame with column name that was read from .mat file
 
            # Creating Y_Measurement and U_Measurement for Training
            Y_Measurement_Train = np.zeros((1,1))
            U_Measurement_Train = np.zeros((8,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = States_k_1_Array_Train[ii,1]
 
                # Getting Disturbances
                T_sol_w = Inputs_k_Array_Train[ii,1]
                T_sol_r = Inputs_k_Array_Train[ii,2]
                T_am = Inputs_k_Array_Train[ii,0]
                Q_in = Inputs_k_Array_Train[ii,3]
                Q_ac = Inputs_k_Array_Train[ii,4]
                Q_venti = Inputs_k_Array_Train[ii,5]
                Q_infil = Inputs_k_Array_Train[ii,6]
                Q_sol = Inputs_k_Array_Train[ii,7]
 
                # Updating Y_Measurement
                # Y_Measurement_Train = np.hstack((Y_Measurement_Train, np.reshape(np.array([T_z]), (1,1))))
 
                # # Updating U_Measurement
                # U_Measurement_Train = np.hstack((U_Measurement_Train, np.reshape(np.array([T_sol_w, T_sol_r, T_a, Q_int, Q_ac, Q_venti, Q_infil, Q_sol]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Train = Y_Measurement_Train[:,1:]
            # U_Measurement_Train = U_Measurement_Train[:,:-1]
 
            # Creating Y_Measurement and U_Measurement for Testing
            Y_Measurement_Test = np.zeros((1,1))
            U_Measurement_Test = np.zeros((7,1))
 
            for ii in range(Aggregated_DF.shape[0]):
 
                # Getting State Measurement
                T_z = States_k_1_Array_Test[ii,1]
 
                # Getting Disturbances
                T_sol_w = Inputs_k_Array_Test[ii,1]
                T_sol_r = Inputs_k_Array_Test[ii,2]
                T_am = Inputs_k_Array_Test[ii,0]
                Q_in = Inputs_k_Array_Test[ii,3]
                Q_ac = Inputs_k_Array_Test[ii,4]
                Q_venti = Inputs_k_Array_Test[ii,5]
                Q_infil = Inputs_k_Array_Test[ii,6]
                Q_sol = Inputs_k_Array_Test[ii,7]
 
                # Updating Y_Measurement
                # Y_Measurement_Test = np.hstack((Y_Measurement_Test, np.reshape(np.array([T_z]), (1,1))))
 
                # # Updating U_Measurement
                # U_Measurement_Test = np.hstack((U_Measurement_Test, np.reshape(np.array([T_sol_w, T_sol_r, T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]), (6,1))))
 
            # Removing first elements from Y_Measurement and U_Measurement - Not Required here
            # Y_Measurement_Test = Y_Measurement_Test[:,1:]
            # U_Measurement_Test = U_Measurement_Test[:,:-1]  
        
        # Creating data for optimization problem
        # Getting State Measurement
        y = States_k_1_Array_Train[:,1]

        # Getting Disturbances
        T_sol_w = Inputs_k_Array_Train[:,1]
        T_sol_r = Inputs_k_Array_Train[:,2]
        T_am = Inputs_k_Array_Train[:,0]
        Q_in = Inputs_k_Array_Train[:,3]
        Q_ac = Inputs_k_Array_Train[:,4]
        Q_venti = Inputs_k_Array_Train[:,5]
        Q_infil = Inputs_k_Array_Train[:,6]
        Q_sol = Inputs_k_Array_Train[:,7]
   
    # =============================================================================
    # Creating Sim_ANNModelData Folder
    # =============================================================================
 
    if (Shoter_ResultsPath == False):
 
        # Making Additional Folders for storing Aggregated Files
        Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..', '..', 'Results',
                                                            'Processed_BuildingSim_Data')
           
        Sim_ANNModelData_FolderName = 'Sim_GB_ModelData'      
   
 
        # Checking if Folders Exist if not create Folders
        if (
                os.path.isdir(
                    os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName))):
 
            # Folders Exist
            z = None
 
        else:
 
            os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName))
 
        # Make the Training Folder
        os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName, Training_FolderName))
 
        # Creating Sim_RegressionModelData Folder Path
        Sim_ANNModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name,
                                                Sim_ANNModelData_FolderName, Training_FolderName)
   
    elif (Shoter_ResultsPath == True):
   
        ## Shorter Path
       
        Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..', '..')
 
        Sim_ANNModelData_FolderName = Short_ResultFolder
 
        # Make the Training Folder
        os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName, Training_FolderName))
 
        # Creating Sim_RegressionModelData Folder Path
        Sim_ANNModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath,
                                                Sim_ANNModelData_FolderName, Training_FolderName)
        
    # =============================================================================
        # Estimation
    # =============================================================================
    
    if (Type_Data_Source  == 1):  # PNNL Prototype Data
    
        if (Type_System_Model == 1):  # One State
    
            None
    
        elif (Type_System_Model == 2): # Two State
    
            None
    
    elif (Type_Data_Source  == 2): # 4th Order ODE House Thermal Model
    
        if (Type_System_Model == 1):  # One State
    
            ## Initial Setup

            # State/Parameter/Output Dimensions
            State_n = 1
            Parameter_n = 5
            Output_n = 1
            Input_n= 6

            # Initial Filter stae mean/covariance - as one state
            T_ave_ini_model = 22. #room temp

            #parameters
            R_win_ini_model = 9.86
            C_in_ini_model = 1000
            C_1_ini_model = 1000
            C_2_ini_model = 1000
            C_3_ini_model = 1000

            # State Covariance
            P_model = 1

            # Filter process/measurement noise covariances
            Q_model = 0.01 #state dynamics-next state uncertainity
            R_model = 0.01 #output uncerratinity - always scalor, PQ based on states

            # Creating Infinity
            Infinity = np.inf


            ## Creating Optimization Variables

            # State Variables
            T_ave = SX.sym('T_ave',N+1,1)

            #Output Variable
            y_l = SX.sym('y_l',N,1)

            ## Getting total time steps
            N = y.shape[0]

            # Parameter Variables
            R_win = SX.sym('R_win',1,1)
            C_in = SX.sym('C_in',1,1)
            C_1 = SX.sym('C_1',1,1)
            C_2 = SX.sym('C_2',1,1)
            C_3 = SX.sym('C_3',1,1)

            # System Matrix
            A_matrix = SX.sym('A_matrix',State_n,State_n)

            A_matrix[0,0] = -1/(R_win*C_in)


            # System Constants
            C_matrix = DM(1,State_n)
            C_matrix[:,:] = np.reshape(np.array([1]), (1,State_n)) #np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

            #Creating input matrix, i.e B
            B_matrix = SX.sym('B_matrix',State_n,Input_n)

            B_matrix[0,0] = 1/(R_win*C_in)
            B_matrix[0,1] = C_1/C_in
            B_matrix[0,2] = C_2/C_in
            B_matrix[0,3] = C_3/C_in
            B_matrix[0,4] = 1/C_in
            B_matrix[0,5] = 1/C_in

            ## Constructing the Cost Function

            # Cost Function Development
            CostFunction = 0

            ## Constructing the Constraints

            # Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
            T_ave_lb = []
            T_ave_ub = []

            y_lb = []
            y_ub = []

            R_win_lb = [0]
            R_win_ub = [Infinity]
            C_in_lb = [0]
            C_in_ub = [Infinity]
            C_1_lb = [0]
            C_1_ub = [Infinity]
            C_2_lb = [0]
            C_2_ub = [Infinity]
            C_3_lb = [0]
            C_3_ub = [Infinity]




            Eq_x_lb = []
            Eq_y_lb = []

            Eq_x_ub = []
            Eq_y_ub = []

            Eq_x = []
            Eq_y = []

            # FOR LOOP: For each time step
            for ii in range(N):

                # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
                CostFunction += (y_l[ii]-y[ii])**2

                ## State/Covariance Equations - Formulation

                # Creating State Vector
                x_k_1 = SX.sym('x_k_1',State_n,1)
                x_k = SX.sym('x_k',State_n,1)

                x_k_1[0,0] = T_ave[ii+1]

                x_k[0,0] = T_ave[ii]

                #Creating input vector - U

                U_vector = DM(Input_n,1)
                U_vector[:,:] = np.reshape(np.array([T_am[ii],Q_in[ii],Q_ac[ii],Q_sol[ii],Q_venti[ii],Q_infil[ii]]), (Input_n,1))

                # State Equation
                x_Eq = -x_k_1 + x_k + ts*(A_matrix @ x_k + B_matrix @ U_vector)
                y_Eq = C_matrix @ x_k #D matrix only for feed forward system


                # Adding current equations to Equation List
                Eq_x += [x_Eq[0,0]] #[1,0] for 2 state
                Eq_y += [y_Eq[0,0]] #always scalor for all states


                # Adding Equation Bounds, [0,0] for 2 equations
                Eq_x_lb += [0]
                Eq_x_ub += [0]

                Eq_y_lb += [0]
                Eq_y_ub += [0]


                # Adding Variable Bounds
                T_ave_lb += [-Infinity]
                T_ave_ub += [Infinity]

                y_lb += [-Infinity]
                y_ub += [Infinity]

            ## Adding Variable Bounds - For (N+1)th Variable
            T_ave_lb += [-Infinity]
            T_ave_ub += [Infinity]

            ## Constructing NLP Problem

            # Creating Optimization Variable: x
            x = vcat([R_win,C_in,C_1,C_2,C_3,T_ave,y])

            # Creating Cost Function: J
            J = CostFunction

            # Creating Constraints: g
            g = vertcat(*Eq_x, *Eq_y)

            # Creating NLP Problem
            NLP_Problem = {'f': J, 'x': x, 'g': g}

            ## Constructiong NLP Solver
            NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

            ## Solving the NLP Problem

            # Creating Initial Variables
            T_ave_ini = T_ave_ini_model*np.ones((N+1,)).tolist()
            R_win_ini = R_win_ini_model*np.ones((1,)).tolist()
            C_in_ini = C_in_ini_model*np.ones((1,)).tolist()
            C_1_ini = C_1_ini_model*np.ones((1,)).tolist()
            C_2_ini = C_2_ini_model*np.ones((1,)).tolist()
            C_3_ini = C_3_ini_model*np.ones((1,)).tolist()
            y_ini = y_ini_model*np.ones((N,)).tolist()

            x_initial = vertcat(*R_win_ini, *C_in_ini, *C_1_ini, *C_2_ini, *C_3_ini, *T_ave_ini, *y_ini)

            # Creating Lower/Upper bounds on Variables and Equations
            x_lb = vertcat(*R_win_lb, *C_in_lb, *C_1_lb, *C_2_lb, *C_3_lb, *T_ave_lb, *y_lb)

            x_ub = vertcat(*R_win_ub, *C_in_ub, *C_1_ub, *C_2_ub, *C_3_ub, *T_ave_ub, *y_ub)

            G_lb = vertcat(*Eq_x_lb, *Eq_y_lb)

            G_ub = vertcat(*Eq_x_ub, *Eq_y_ub)

            # Solving NLP Problem
            NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

            #----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
            #--------------------------------------------------------------------------------------------------------------------------------------------------------#

            ## Getting the Solutions
            NLP_Sol = NLP_Solution['x'].full().flatten()

            R_win_sol = NLP_Sol[0]
            C_in_sol = NLP_Sol[1]
            C_1_sol = NLP_Sol[2]
            C_2_sol = NLP_Sol[3]
            C_3_sol = NLP_Sol[4]
            T_ave_sol = NLP_Sol[5:(N+1)+5]
            y_sol = NLP_Sol[(N+1)+5:(N+1)+5+N]
    
        elif (Type_System_Model == 2): # Two State
    
            ## Initial Setup

            # State/Parameter/Output Dimensions
            State_n = 2
            Parameter_n = 7
            Output_n = 1
            Input_n= 7

            # Initial Filter stae mean/covariance - as one state
            T_ave_ini_model = 22. #room temp
            T_wall_ini_model = 24.

            #parameters
            R_win_ini_model = 9.86
            R_w2_ini_model = 9.86
            C_in_ini_model = 1000
            C_1_ini_model = 1000
            C_2_ini_model = 1000
            C_3_ini_model = 1000
            C_w_ini_model = 1000

            # State Covariance
            P_model = 1

            # Filter process/measurement noise covariances
            Q_model = 0.01 #state dynamics-next state uncertainity
            R_model = 0.01 #output uncerratinity - always scalor, PQ based on states

            # Creating Infinity
            Infinity = np.inf


            ## Creating Optimization Variables

            # State Variables
            T_ave = SX.sym('T_ave',N+1,1)
            T_wall = SX.sym('T_ave',N+1,1)

            #Output Variable
            y_l = SX.sym('y_l',N,1)

            ## Getting total time steps
            N = y.shape[0]

            # Parameter Variables
            R_win = SX.sym('R_win',1,1)
            R_w2 = SX.sym('R_w2',1,1)
            C_in = SX.sym('C_in',1,1)
            C_1 = SX.sym('C_1',1,1)
            C_2 = SX.sym('C_2',1,1)
            C_3 = SX.sym('C_3',1,1)
            C_w = SX.sym('C_w',1,1)

            # System Matrix
            A_matrix = SX.sym('A_matrix',State_n,State_n)

            A_matrix[0,0] = (-1/(R_w2*C_in)) - (1/(R_win*C_in))
            A_matrix[0,1] = 1/(R_win*C_in)
            A_matrix[1,0] = -1/(R_w2*C_in)
            A_matrix[1,1] = -2/(R_w2*C_w)

            # System Constants
            C_matrix = DM(1,State_n)
            C_matrix[:,:] = np.reshape(np.array([1,0]), (1,State_n)) #np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

            #Creating input matrix, i.e B
            B_matrix = SX.sym('B_matrix',State_n,Input_n)

            B_matrix[0,0] = 0
            B_matrix[0,1] = 1/(R_win*C_in)
            B_matrix[0,2] = C_1/C_in
            B_matrix[0,3] = C_2/C_in
            B_matrix[0,4] = 0
            B_matrix[0,5] = 1/C_in
            B_matrix[0,6] = 1/C_in
            B_matrix[1,0] = 1/(R_w2*C_w)
            B_matrix[1,1] = 0
            B_matrix[1,2] = 0
            B_matrix[1,3] = 0
            B_matrix[1,4] = C_3/C_w
            B_matrix[1,5] = 0
            B_matrix[1,6] = 0

            ## Constructing the Cost Function

            # Cost Function Development
            CostFunction = 0

            ## Constructing the Constraints

            # Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
            T_ave_lb = []
            T_ave_ub = []

            T_wall_lb = []
            T_wall_ub = []

            y_lb = []
            y_ub = []

            R_win_lb = [0]
            R_win_ub = [Infinity]
            R_w2_lb = [0]
            R_w2_ub = [Infinity]
            C_in_lb = [0]
            C_in_ub = [Infinity]
            C_1_lb = [0]
            C_1_ub = [Infinity]
            C_2_lb = [0]
            C_2_ub = [Infinity]
            C_3_lb = [0]
            C_3_ub = [Infinity]
            C_w_lb = [0]
            C_w_ub = [Infinity]


            Eq_x_lb = []
            Eq_y_lb = []

            Eq_x_ub = []
            Eq_y_ub = []

            Eq_x = []
            Eq_y = []

            # FOR LOOP: For each time step
            for ii in range(N):

                # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
                CostFunction += (y_l[ii]-y[ii])**2

                ## State/Covariance Equations - Formulation

                # Creating State Vector
                x_k_1 = SX.sym('x_k_1',State_n,1)
                x_k = SX.sym('x_k',State_n,1)

                x_k_1[0,0] = T_ave[ii+1]
                x_k_1[1,0] = T_wall[ii+1]

                x_k[0,0] = T_ave[ii]
                x_k[1,0] = T_wall[ii]

                #Creating input vector - U

                U_vector = DM(Input_n,1)
                U_vector[:,:] = np.reshape(np.array([T_sol_w[ii], T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infil[ii]]), (Input_n,1))

                # State Equation
                x_Eq = -x_k_1 + x_k + ts*(A_matrix @ x_k + B_matrix @ U_vector)
                y_Eq = C_matrix @ x_k #D matrix only for feed forward system


                # Adding current equations to Equation List
                Eq_x += [x_Eq[1,0]] #[1,0] for 2 state
                Eq_y += [y_Eq[0,0]] #always scalor for all states


                # Adding Equation Bounds, [0,0] for 2 equations
                Eq_x_lb += [0]
                Eq_x_ub += [0]

                Eq_y_lb += [0]
                Eq_y_ub += [0]


                # Adding Variable Bounds
                T_ave_lb += [-Infinity]
                T_ave_ub += [Infinity]

                T_wall_lb += [-Infinity]
                T_wall_ub += [Infinity]

                y_lb += [-Infinity]
                y_ub += [Infinity]

            ## Adding Variable Bounds - For (N+1)th Variable
            T_ave_lb += [-Infinity]
            T_ave_ub += [Infinity]

            T_wall_lb += [-Infinity]
            T_wall_ub += [Infinity]

            ## Constructing NLP Problem

            # Creating Optimization Variable: x
            x = vcat([R_win,R_w2,C_in,C_1,C_2,C_3,C_w,T_ave,T_wall,y])

            # Creating Cost Function: J
            J = CostFunction

            # Creating Constraints: g
            g = vertcat(*Eq_x, *Eq_y)

            # Creating NLP Problem
            NLP_Problem = {'f': J, 'x': x, 'g': g}

            ## Constructiong NLP Solver
            NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

            ## Solving the NLP Problem

            # Creating Initial Variables
            T_ave_ini = T_ave_ini_model*np.ones((N+1,)).tolist()
            T_wall_ini = T_ave_ini_model*np.ones((N+1,)).tolist()
            R_win_ini = R_win_ini_model*np.ones((1,)).tolist()
            R_w2_ini = R_win_ini_model*np.ones((1,)).tolist()
            C_in_ini = C_in_ini_model*np.ones((1,)).tolist()
            C_1_ini = C_1_ini_model*np.ones((1,)).tolist()
            C_2_ini = C_2_ini_model*np.ones((1,)).tolist()
            C_3_ini = C_3_ini_model*np.ones((1,)).tolist()
            C_w_ini = C_3_ini_model*np.ones((1,)).tolist()
            y_ini = y_ini_model*np.ones((N,)).tolist()

            x_initial = vertcat(*R_win_ini, *R_w2_ini, *C_in_ini, *C_1_ini, *C_2_ini, *C_3_ini, *C_w_ini, *T_ave_ini, *T_wall_ini, *y_ini)

            # Creating Lower/Upper bounds on Variables and Equations
            x_lb = vertcat(*R_win_lb, *R_w2_lb, *C_in_lb, *C_1_lb, *C_2_lb, *C_3_lb, *C_w_lb, *T_ave_lb, *T_wall_lb, *y_lb)

            x_ub = vertcat(*R_win_ub, *R_w2_ub, *C_in_ub, *C_1_ub, *C_2_ub, *C_3_ub, *C_w_ub, *T_ave_ub, *T_wall_ub, *y_ub)

            G_lb = vertcat(*Eq_x_lb, *Eq_y_lb)

            G_ub = vertcat(*Eq_x_ub, *Eq_y_ub)

            # Solving NLP Problem
            NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

            #----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
            #--------------------------------------------------------------------------------------------------------------------------------------------------------#

            ## Getting the Solutions
            NLP_Sol = NLP_Solution['x'].full().flatten()

            R_win_sol = NLP_Sol[0]
            R_w2_sol = NLP_Sol[1]
            C_in_sol = NLP_Sol[2]
            C_1_sol = NLP_Sol[3]
            C_2_sol = NLP_Sol[4]
            C_3_sol = NLP_Sol[5]
            C_w_sol = NLP_Sol[6]
            T_ave_sol = NLP_Sol[7:(N+1)+7]
            T_wall_sol = NLP_Sol[(N+1)+7:(N+1)+7+(N+1)]
            y_sol = NLP_Sol[(N+1)+7+(N+1):(N+1)+7+(N+1)+N]
    
        elif (Type_System_Model == 4): # Four State

            ## Initial Setup

            # State/Parameter/Output Dimensions
            State_n = 4
            Parameter_n = 12
            Output_n = 1
            Input_n = 8

            # Initial Filter stae mean/covariance - as one state
            T_ave_ini_model = 22. #room temp
            T_wall_ini_model = 24.
            T_attic_ini_model = 22.
            T_im_ini_model = 22.

            #parameters

            R_win_ini_model = 9.86
            R_w2_ini_model = 9.86
            R_attic_ini_model = 9.86
            R_im_ini_model = 9.86
            R_roof_ini_model = 9.86
            C_in_ini_model = 1000
            C_w_ini_model = 1000
            C_attic_ini_model = 1000
            C_im_ini_model = 1000
            C_1_ini_model = 1000
            C_2_ini_model = 1000
            C_3_ini_model = 1000

            # Creating Infinity
            Infinity = np.inf


            ## Creating Optimization Variables

            # State Variables
            T_ave = SX.sym('T_ave',N+1,1)
            T_wall = SX.sym('T_ave',N+1,1)
            T_attic = SX.sym('T_attic',N+1,1)
            T_im = SX.sym('T_im',N+1,1)

            #Output Variable
            y_l = SX.sym('y_l',N,1)

            ## Getting total time steps
            N = y.shape[0]

            # Parameter Variables
            R_win = SX.sym('R_win',1,1)
            R_w2 = SX.sym('R_w2',1,1)
            R_attic = SX.sym('R_attic',1,1)
            R_im = SX.sym('R_im',1,1)
            R_roof = SX.sym('R_roof',1,1)

            C_in = SX.sym('C_in',1,1)
            C_w = SX.sym('C_w',1,1)
            C_attic = SX.sym('C_attic',1,1)
            C_im = SX.sym('C_im',1,1)
            C_1 = SX.sym('C_1',1,1)
            C_2 = SX.sym('C_2',1,1)
            C_3 = SX.sym('C_3',1,1)

            # System Matrix
            A_matrix = SX.sym('A_matrix',State_n,State_n)

            A_matrix[0,0] = (-1/C_in(1/(R_w2) + 1/R_attic + 1/R_im + 1/R_win))
            A_matrix[0,1] = 1/(C_in*(R_w2))
            A_matrix[0,2] = 1/(C_in*R_attic)
            A_matrix[0,3] = 1/(C_in*R_im)
            A_matrix[1,0] = 1/(C_w*(R_w2))
            A_matrix[1,1] = -2/(C_w*(R_w2))
            A_matrix[1,2] = 0
            A_matrix[1,3] = 0
            A_matrix[2,0] = 1/(C_attic*R_attic)
            A_matrix[2,1] = 0
            A_matrix[2,2] = -2/(C_attic*R_attic)
            A_matrix[2,3] = 0
            A_matrix[3,0] = 1/(C_im*R_im)
            A_matrix[3,1] = 0
            A_matrix[3,2] = 0
            A_matrix[3,3] = -1/(C_im*R_im)

            # System Constants
            C_matrix = DM(1,State_n)
            C_matrix[:,:] = np.reshape(np.array([1,0,0,0]), (1,State_n)) #np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

            #Creating input matrix, i.e B
            B_matrix = SX.sym('B_matrix',State_n,Input_n)

            B_matrix[0,0] = 0
            B_matrix[0,1] = 0
            B_matrix[0,2] = 1/(C_in*R_win)
            B_matrix[0,3] = C_1/C_in
            B_matrix[0,4] = C_2/C_in
            B_matrix[0,5] = 0
            B_matrix[0,6] = 1/C_in
            B_matrix[0,7] = 1/C_in


            B_matrix[1,0] = 0
            B_matrix[1,1] = 1/(C_w*R_w2)
            B_matrix[1,2] = 0
            B_matrix[1,3] = 0
            B_matrix[1,4] = 0
            B_matrix[1,5] = 0
            B_matrix[1,6] = 0
            B_matrix[1,7] = 0

            B_matrix[2,0] = 1/(C_attic*R_roof)
            B_matrix[2,1] = 0
            B_matrix[2,2] = 0
            B_matrix[2,3] = 0
            B_matrix[2,4] = 0
            B_matrix[2,5] = 0
            B_matrix[2,6] = 0
            B_matrix[2,7] = 0


            B_matrix[3,0] = 0
            B_matrix[3,1] = 0
            B_matrix[3,2] = 0
            B_matrix[3,3] = 0
            B_matrix[3,4] = 0
            B_matrix[3,5] = C_3/C_im
            B_matrix[3,6] = 0
            B_matrix[3,7] = 0

            ## Constructing the Cost Function

            # Cost Function Development
            CostFunction = 0

            ## Constructing the Constraints

            # Initializing Lower-Upper Bounds for State/Parameters/Intermediate variables and the Equations
            T_ave_lb = []
            T_ave_ub = []

            T_wall_lb = []
            T_wall_ub = []

            T_attic_lb = []
            T_attic_ub = []

            T_im_lb = []
            T_im_ub = []

            y_lb = []
            y_ub = []

            R_win_lb = [0]
            R_win_ub = [Infinity]

            R_w2_lb = [0]
            R_w2_ub = [Infinity]

            C_in_lb = [0]
            C_in_ub = [Infinity]

            C_1_lb = [0]
            C_1_ub = [Infinity]

            C_2_lb = [0]
            C_2_ub = [Infinity]

            C_3_lb = [0]
            C_3_ub = [Infinity]

            C_w_lb = [0]
            C_w_ub = [Infinity]

            R_attic_lb = [0]
            R_aattic_ub = [Infinity]

            R_im_lb = [0]
            R_im_ub = [Infinity]

            R_roof_lb = [0]
            R_roof_ub = [Infinity]

            C_attic_lb = [0]
            C_attic_ub = [Infinity]

            C_im_lb = [0]
            C_im_ub = [Infinity]


            Eq_x_lb = []
            Eq_y_lb = []

            Eq_x_ub = []
            Eq_y_ub = []

            Eq_x = []
            Eq_y = []

            # FOR LOOP: For each time step
            for ii in range(N):

                # Computing Cost Function: e_l_T * S_inv * e_l + log(S)
                CostFunction += (y_l[ii]-y[ii])**2

                ## State/Covariance Equations - Formulation

                # Creating State Vector
                x_k_1 = SX.sym('x_k_1',State_n,1)
                x_k = SX.sym('x_k',State_n,1)

                x_k_1[0,0] = T_ave[ii+1]
                x_k_1[1,0] = T_wall[ii+1]
                x_k_1[2,0] = T_attic[ii+1]
                x_k_1[3,0] = T_im[ii+1]

                x_k[0,0] = T_ave[ii]
                x_k[1,0] = T_wall[ii]
                x_k[2,0] = T_attic[ii]
                x_k[3,0] = T_im[ii]

                #Creating input vector - U

                U_vector = DM(Input_n,1)
                U_vector[:,:] = np.reshape(np.array([T_sol_r[ii], T_sol_w[ii], T_am[ii], Q_in[ii], Q_ac[ii], Q_sol[ii], Q_venti[ii], Q_infil[ii]]), (Input_n,1))

                # State Equation
                x_Eq = -x_k_1 + x_k + ts*(A_matrix @ x_k + B_matrix @ U_vector)
                y_Eq = C_matrix @ x_k #D matrix only for feed forward system


                # Adding current equations to Equation List
                Eq_x += [x_Eq[1,0,0,0]]
                Eq_y += [y_Eq[0,0]]


                # Adding Equation Bounds, [0,0] for 2 equations
                Eq_x_lb += [0]
                Eq_x_ub += [0]

                Eq_y_lb += [0]
                Eq_y_ub += [0]


                # Adding Variable Bounds
                T_ave_lb += [-Infinity]
                T_ave_ub += [Infinity]

                T_wall_lb += [-Infinity]
                T_wall_ub += [Infinity]

                T_attic_lb += [-Infinity]
                T_attic_ub += [Infinity]

                T_im_lb += [-Infinity]
                T_im_ub += [Infinity]

                y_lb += [-Infinity]
                y_ub += [Infinity]

            ## Adding Variable Bounds - For (N+1)th Variable
            T_ave_lb += [-Infinity]
            T_ave_ub += [Infinity]

            T_wall_lb += [-Infinity]
            T_wall_ub += [Infinity]

            T_attic_lb += [-Infinity]
            T_attic_ub += [Infinity]

            T_im_lb += [-Infinity]
            T_im_ub += [Infinity]

            ## Constructing NLP Problem

            # Creating Optimization Variable: x
            x = vcat([R_win, R_w2, R_attic, R_im, R_roof, C_in, C_w, C_attic, C_im, C_1, C_2, C_3, T_ave, T_wall, T_attic,T_im,y])

            # Creating Cost Function: J
            J = CostFunction

            # Creating Constraints: g
            g = vertcat(*Eq_x, *Eq_y)

            # Creating NLP Problem
            NLP_Problem = {'f': J, 'x': x, 'g': g}

            ## Constructiong NLP Solver
            NLP_Solver = nlpsol('nlp_solver', 'ipopt', NLP_Problem)

            ## Solving the NLP Problem

            # Creating Initial Variables
            T_ave_ini = T_ave_ini_model*np.ones((N+1,)).tolist()
            T_wall_ini = T_ave_ini_model*np.ones((N+1,)).tolist()
            R_win_ini = R_win_ini_model*np.ones((1,)).tolist()
            R_w2_ini = R_win_ini_model*np.ones((1,)).tolist()
            C_in_ini = C_in_ini_model*np.ones((1,)).tolist()
            C_1_ini = C_1_ini_model*np.ones((1,)).tolist()
            C_2_ini = C_2_ini_model*np.ones((1,)).tolist()
            C_3_ini = C_3_ini_model*np.ones((1,)).tolist()
            C_w_ini = C_3_ini_model*np.ones((1,)).tolist()
            y_ini = y_ini_model*np.ones((N,)).tolist()

            x_initial = vertcat(*R_win_ini, *R_w2_ini, *R_attic_ini, *R_im_ini, *R_roof_ini, *C_in_ini, *C_w_ini, *C_attic_ini, *C_im_ini, *C_1_ini, *C_2_ini, *C_3_ini, *T_ave_ini, *T_wall_ini, *y_ini)

            # Creating Lower/Upper bounds on Variables and Equations
            x_lb = vertcat(*R_win_lb, *R_w2_lb, *R_attic_lb, *R_im_lb, *R_roof_lb, *C_in_lb, *C_w_lb, *C_attic_lb, *C_im_lb, *C_1_lb, *C_2_lb, *C_3_lb, *T_ave_lb, *T_wall_lb, *y_lb)

            x_ub = vertcat(*R_win_ub, *R_w2_ub, *R_attic_ub, *R_im_ub, *R_roof_ub, *C_in_ub, *C_w_ub, *C_attic_ub, *C_im_ub, *C_1_ub, *C_2_ub, *C_3_ub, *T_ave_ub, *T_wall_ub, *y_ub)

            G_lb = vertcat(*Eq_x_lb, *Eq_y_lb)

            G_ub = vertcat(*Eq_x_ub, *Eq_y_ub)

            # Solving NLP Problem
            NLP_Solution = NLP_Solver(x0 = x_initial, lbx = x_lb, ubx = x_ub, lbg = G_lb, ubg = G_ub)

            #----------------------------------------------------------------Solution Analysis ----------------------------------------------------------------------#
            #--------------------------------------------------------------------------------------------------------------------------------------------------------#

            ## Getting the Solutions
            NLP_Sol = NLP_Solution['x'].full().flatten()

            R_win_sol = NLP_Sol[0]
            R_w2_sol = NLP_Sol[1]
            R_attic_sol = NLP_Sol[2]
            R_im_sol = NLP_Sol[3]
            R_roof_sol = NLP_Sol[4]
            C_in_sol = NLP_Sol[5]
            C_w_sol = NLP_Sol[6]
            C_attic_sol = NLP_Sol[7]
            C_im_sol = NLP_Sol[8]
            C_1_sol = NLP_Sol[9]
            C_2_sol = NLP_Sol[10]
            C_3_sol = NLP_Sol[11]
            T_ave_sol = NLP_Sol[12:(N+1)+12]
            T_wall_sol = NLP_Sol[((N+1)+12):(N+1)+12+(N+1)]
            T_attic_sol = NLP_Sol[(N+1)+12+(N+1):(N+1)+12+(N+1)+(N+1)]
            T_im_sol = NLP_Sol[(N+1)+12+(N+1)+(N+1):(N+1)+12+(N+1)+(N+1)+(N+1)]
            y_sol = NLP_Sol[(N+1)+12+(N+1)+(N+1)+(N+1)+12+(N+1)+(N+1)+(N+1)+N]
