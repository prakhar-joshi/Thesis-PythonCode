#!/usr/bin/env python
# coding: utf-8

# In[422]:


#-------------------------------------------------------------------------#
########----------- WT Gearbox Reliabilty Analysis Tool -----------########
#-------------------------------------------------------------------------#

# 3 Main sections- 1.relibility analysis model, 2.RIF, 3.Cost Model, 4. GUI


# In[423]:


import numpy as np
from graphviz import Digraph
# import pandas as pd  #if you want to use the code snippet to save the GUI table data as csv
import PySimpleGUI as sg
import matplotlib.pyplot as plt


# In[424]:


#-------------------------------------------------------------------------#
########----------- 1. Reliblity Analysis -----------########
#-------------------------------------------------------------------------#


# Relibilty estimation is done by two methods namely quantitative and qualitative
# Qualitative and Quantitative options are two options to the user
# most functinos do the same thing but are defined differently for both the options
# Quantitative structure and data are from Kang 2019 and qualitative is from Bhardwaj


# In[425]:


################# Defining the FTA ###############
# Variables lbls and lbls_name are the code names(optional) and names of the Fault tree elements elements
# order of lbls is important start from top level, end at basic events
# Matrix variable is a matrix of 0 and 1 of dimensions (n x n), n = no. of lbls
# n rows and n coloumns are in the order mentioned in lbls 
# 1 in Matrix depicts directly dependent lower elements
# basic_index is the array of indices indicating the basic faults out of lbls

def creatingArray_Quantitative():
    #     lbls = np.array(["F","M1","M2","M3","B4","B5","B6","B7","I8","B9","B10","I11","B12","B13","B14","B15","B16","B17","B18","B19"])
    lbls_name = np.array(["Gearbox Failure","Abnormal Gear","Bearing Fault","Lubrication System Failure","Abnormal Filter",
                         "Poor Quality of lubrication oil","Contamination","Abnormal Vibration","Tooth Wear","Scuffing","Gear Pitting",
                         "Cracks in Gear","Corrosion of Pins","Abrasive Wear","Surface Fatigue","Gear Tooth Deterioration",
                         "Gear Teeth Offset"])
    
    temp=len(lbls_name)
    #creating matrix one array at a time
    f0=np.zeros(temp) #f0 has dependencies with 1st,2nd and 3rd element(counting starts with 0)
    f0[1]=f0[2]=f0[3]=1
    
    m1=np.zeros(temp)
    m1[7]=m1[8]=m1[9]=m1[10]=m1[11]=1

    m2=np.zeros(temp)
    m2[7]=m2[12]=m2[13]=m2[14]=1
    
    m3=np.zeros(temp)
    m3[4]=m3[5]=m3[6]=1

    i8=np.zeros(temp)
    i8[15]=i8[16]=1

    #     i11=np.zeros(temp)
    #     i11[17]=i11[18]=i11[19]=1
    
    # concatenate the arrays in a matrix form
    matrix=np.zeros((temp,temp))
    matrix[0,:]=f0
    matrix[1,:]=m1
    matrix[2,:]=m2
    matrix[3,:]=m3
    matrix[8,:]=i8
    #     matrix[11,:]=i11
    #     matrix[16,:]=i16
    
    basic_index=np.array([4,5,6,7,9,10,11,12,13,14,15,16])
    for x in basic_index:
        matrix[x,x]=1

    # basic indices are represented as only one '1' in the matrix referring to the same column and row location
    
    return lbls_name, matrix, basic_index


# In[426]:


def creatingArray_Qualitative():
    lbls_name = np.array(["Gearbox_Failure","Gear", "Bearing", "Lubrication System" , "Housing" , "Shaft",
                              "Gear teeth fail","Gear teeth Slip","Abnormal Gear noise",
                              "Loss of bearing Function","Abnormal bearing Noise","Bearing Ring Creep","bearing Misalignment","Overheating/Seizure",
                              "Loss of Lubrication",
                              "Fracture in Housing","Leakage",
                              "Fatigue and Fracture","Bending/Deflection","Surface Finish Degradation", "Shaft Missalignment",
                              "Gear Abrasive Wear","Gear Fretting Corrosion","Gear Scuffing","Gear Pitting", "Plastic Deformation", "Tooth Shear","Gear Spalling","Tooth Bending Fatigue","Gear Missalignment","Gear Lack of Lubrication",
                              "Bearing Spalling","Bearing Smearing","Brinelling","Flutting","Axial Cracking","Bearing Scuffing","Contact Wear","Bearing Fretting Corrosion","Bearing Vibration","Fracture in bearing groove","Lack of heat removal in bearing",
                              "Failure of oil filter","Poor/altered oil quality","Debris","Lack of heat removal","Inadequate Oil","Pump loss",
                              "Housing Corrosion","Loose Fitting","Housing Vibration","Bolt Failure","Degraded Washers",
                              "Irregular Grooving","Weld defect","Shaft Fretting Corrosion","High Speed","High loads","High temperature","Improper Assembly"
                         ])
    
    basic_index = range(21,60)

    matrix=np.zeros((len(lbls_name)-len(basic_index),len(lbls_name)))
    
    #first row carries the known failure rates of component, ie. gearbox,gears,bearings etc.
    matrix[0][1:6]=[0.0015,0.0910,0.016,0.004,0.0175]
    
    #rows 1 -5 connect subassembly components to failure modes. factors sum upto 1 and indicate the likliness of occurence of that specific failure mode if a fialure is reported in the component
    matrix[1][6:9]=[0.7,0.2,0.1]
    matrix[2][9:14]=[0.6,0.1,0.1,0.1,0.1]
    matrix[3][14]=1
    matrix[4][15:17]=[0.4,0.6]
    matrix[5][17:21]=[0.4,0.3,0.2,0.1]

    # starts connecting failure modes to failure causes, the rows are filled with weights that sum upto 1 
    matrix[6][21:29]=[0.1,0.2,0.25,0.15,0.12,0.08,0.05,0.05]
    matrix[7][25:30]=[0,0,0,0.7,0.3]
    matrix[8][30]=1

    matrix[9][31:39]=[0.08,0.2,0.08,0.07,0.15,0.1,0.2,0.12]
    matrix[10][[36,38,39]]=[0.2,0.3,0.5] 
    matrix[11][37]=1
    matrix[12][39:41]=[0,1]
    matrix[13][41]=1

    matrix[14][42:48]=[0.2,0.15,0.15,0.2,0.15,0.15]

    matrix[15][48:51]=[0.3,0.35,0.35]
    matrix[16][51:53]=[0.5,0.5]

    matrix[17][53:58]=[0.1,0.2,0.15,0.2,0.35]
    matrix[18][[55,58]]=[0.3,0.7]
    matrix[19][56:58]=[0.6,0.4]
    matrix[20][[57,59]]=[0.7,0.3]

    return lbls_name, matrix, basic_index


# In[427]:


# to convert failure rate in per year to failure probability
def failurerate2prob(rate):
    time=20 #20 years  
    prob = 0-np.expm1(-rate*time) # probability of failure    
    # expm1(x) would give exp(x)-1 but more accurately
    return prob
    


# In[428]:


# This part of code generates the prob array(failure rate) with only the given basic event probability mentioned, other values are set to 0
def Prob_Quantitative(matrix):
    #     lbls_name = np.array(["Gearbox Failure","Abnormal Gear","Bearing Fault","Lubrication System Failure","Abnormal Filter",
    #                          "Poor Quality of lubrication oil","Contamination","Abnormal Vibration","Tooth Wear","Glued","Gear Pitting",
    #                          "Cracks in Gear","Corrosion of Pins","Abrasive Wear","Surface Fatigue","Gear Tooth Deterioration",
    #                          "Gear Teeth Offset"])
    #     basic_index=np.array([4,5,6,7,9,10,11,12,13,14,15,16])

    fail_rate=np.array([1.8,1.8,1.44,2.14,0.24,1.3,1.54,12,10,3,0.3,1.3]) # note, this*i is in per hour failure rate
    i=10**-6
    convert=24*365 #converting rate from per hour to per year
        
    fail_rate=fail_rate*i*convert
    
    #      basic_prob=failurerate2prob(fail_rate)
    # basic probability is the probability of basic failure events occuring as given in literature

    temp=np.count_nonzero(matrix,axis=1)

    prob=np.zeros(len(matrix))
    i=0
    for x in range(len(matrix)):
        if temp[x]==1:
            prob[x]=fail_rate[i]
            i=i+1
    # variabile 'prob' is going to be used throughout the rest of the code
    # prob carries the probability of all Fault tree elements
    return prob


# In[429]:


def Prob_Qualitative(matrix):
    
    fail_rate=np.zeros(len(matrix)) #failure rate is in year
    fail_rate[0]=np.sum(matrix[0])
    fail_rate[1:6]=matrix[0][1:6]
    
    for x in range(len(fail_rate)):
        for y in range(6,len(fail_rate)):
            if matrix[x][y] !=0:
                fail_rate[y]=matrix[0][x]*matrix[x][y]
    
    return fail_rate


# In[430]:


# GearboxFTA function creates a tree diagram, uses lbls and matrix 
# def GearboxFTA(lbls_name,matrix,name):
#     dot=Digraph()
#     for x in range(len(matrix)):
#         for y in range(len(matrix[0])):
#             if matrix[x,y] != 0:
#                 if x != y:
#                     dot.edge(str(lbls_name[x]),str(lbls_name[y]))    
#     display(dot)
#     save="finalfigs \"" + name
# #   saves the graph as pdf
#     dot.render(name, view=True)  


# In[431]:


# function to calculate the failure rate of higher level tree elements from the basic event probability
def calc_failure_Quantitative(fail_rate,matrix):
    rev_mat=np.copy(matrix[::-1])
    # reversing the order so that lower level events are evaluated first and their result can be used for higher level
    prob_tem=failurerate2prob(fail_rate) #for calculations, converting failure rate to failure probability
    tem=np.zeros(len(rev_mat))
    for x in range(len(rev_mat)):
        tem=prob_tem*rev_mat[x]  # setting all probabilities other than dependent proababilites as 0
        # reversing order of probability
        prob_tem[len(rev_mat)-x-1]=1-np.prod(1-tem) #probability calculating formula
    prob_final=np.copy(prob_tem)
    prob_final=(0-np.log(1-prob_final))/(20)  #returned as failure rate per year
    return prob_final


# In[432]:


# given the user input change to the failure causes, the weights for the failure causes are updated
# rest of the calucation happens on updated weights, which changes the ratio of weights of failure modes are summed up and can be more or less than 1. This is multiplied with the original failure rate of component to get the new rate
# component failure rates are summed up to get teh gearbox failure rate
def calc_failure_Qualitative(change_input,matrix):
    #     change=np.ones(len(basic))
    #     change= [1.]*len(basic)
    basic=len(matrix[0])-len(matrix)
    change = change_input[-basic:]

    W=matrix[6:21,21:60] # matrix with weights of failure cause, dimension=n(failuremodes)xn(failurecauses)
    A=matrix[1:6,6:21] # matrix with failure mode weights, dimension= n(components)xn(failuremodes)
    F=matrix[0,1:6] # array of failure rates of subassembly(also called as component)
    #     F = [0.0015,0.0910,0.016,0.004,0.0175]
   
    M=np.matmul(W,np.transpose(change)) #Updated weights of failure causes based on user requested change and sumed them up , dimension=n(failuremodes),1
    
    #      P=np.matmul(A,M) #
    P=A*M
    temp=np.transpose(F*np.transpose(P))
    Fail=[]
    for x in temp:
        Fail.append(sum(x))
        
    Fail.insert(0,sum(Fail))

    for x in temp:
        for y in x:
            if y != 0:
                Fail.append(y)
    return Fail


# In[433]:


#-------------------------------------------------------------------------#
########----------- 2. Reliblity Influencing Factors -----------########
#-------------------------------------------------------------------------#
# an interconnect to find the what component properites influence the failure cause
# Adapted from Rahimi & Rausand 2013 and Bhardwaj 2019
# Rifs are different for both methods.hence two differnet functions
# Basic indices are used to connect these to the failure cause


# In[434]:


# making a component->RIF->cause nested class
class cause:
    def __init__(self,RIFname,cause_arr,lbls_name):
        self.name=RIFname #name of RIF
        self.cause=lbls_name[cause_arr] #name of causes corresponding to each RIF
        self.causeindex=cause_arr #index of causes corresponding to each RIF
        self.causechange=np.ones(len(cause_arr)) # can be used to store reqeuested change in failure cause prob
class RIF:
    def __init__(self,component,RIF,lbls_name,cause_mat):
        self.name=component #name of the component
        self.allRIFs=RIF #name of all rifs in a component
        self.RIF=[None]*len(RIF)
        for x in range(len(self.RIF)):
            self.RIF[x]=cause(RIF[x],cause_mat[x],lbls_name) #nests class cause


# In[435]:


def nestingRIFs_Qualitative(lbls_name):
    # components is a list array defining all componets
    components=np.array(["Gears","Gear_Lube","Bearing","Bearing_Lube","Lubrication System","Housing","Shaft","Others"])

    # RIF_x is the names of all RelibilityInfluencingFactors of component x
    # x_causemat are 2D arrays each row contains the indices of basic causes in the FTA
    # Rows corresponds with the different rifs in the  component in the order of RIF_x
    RIF_gear = np.array(["Gear Quality","Gear Maintenance Accessibility"])
    gear_causeMat=[[25,26,27,28,29],[29,30]]

    RIF_gearlube = np.array(["Improper Gear Lubrication", "Gear Lube Contamination"])
    gearlube_causeMat=[[23,24,27,30],[21,22,23,24, 43,44]]
    
    RIF_bearing = np.array(["Bearing Quality","Bearing Maintenance Accessibility"])
    bearing_causeMat=[[35,39,40],[34,40]]

    RIF_bearinglube = np.array(["Improper Bearing Lubrication", "Bearing Lube Contamination"])
    bearinglube_causeMat=[[33,36,37,41],[31,37]]

    RIF_lubesys = np.array(["Lubrication System Quality","Lubrication System Maintainenece Frequency","Lubrication System Maintenance Accessibility"])
    lubesys_causeMat=[[44,46],[42,43],[47]]
    
    RIF_housing = np.array(["Housing Quality","Housing Maintenance Accessibility","Housing Maintainenece Frequency"])
    housing_causeMat=[[49],[49],[51]]
    
    RIF_shaft = np.array(["Shaft Quality","Shaft Maintenance Accessibility"])
    shaft_causeMat=[[59],[53,54]]
    
    RIF_others = np.array(["Wind","External_Vibration", "Temperature" , "Environment"])
    others_causeMat=[[21,28,31,33,36,37,41,56,57],[22,28,29,32,38,39,45,46,50,59],[25,30,35,41,52,58],[22,24,38,43,48,55]]
    
    # allRIFs is array of RIF arrays
    # cause mat is  3d array of individual 2d arrays
    allRIFs=[RIF_gear,RIF_gearlube,RIF_bearing,RIF_bearinglube,RIF_lubesys,RIF_housing,RIF_shaft,RIF_others]
    causemat=[gear_causeMat,gearlube_causeMat,bearing_causeMat,bearinglube_causeMat,lubesys_causeMat,housing_causeMat,shaft_causeMat,others_causeMat]
    
    component= [None]*len(components)
    #     print(len(components))
    #assign values to the component_rif classes    
    for x in range((len(components))):
            component[x]=RIF(components[x],allRIFs[x], lbls_name,causemat[x])
    return component


# In[436]:


# nesting RIFs provides values inside the nested classes
def nestingRIFs_Quantitative(lbls_name):
    # components is a list array defining all componets
    components=np.array(["Bearing","Bearing Lube","Gears","Gear Lube","Others"])

    #     lbls_name = np.array(["Gearbox Failure","Abnormal Gear","Bearing Fault","Lubrication System Failure","Abnormal Filter",
    #                          "Poor Quality of lubrication oil","Contamination","Abnormal Vibration","Tooth Wear","Glued","Gear Pitting",
    #                          "Cracks in Gear","Corrosion of Pins","Abrasive Wear","Surface Fatigue","Gear Tooth Deterioration",
    #                          "Gear Teeth Offset"])
    #     basic_index=np.array([4,5,6,7,9,10,11,12,13,14,15,16])

    
    
    # RIF_x is the names of all RelibilityInfluencingFactors of component x
    # x_causemat are 2D arrays each row contains the indices of basic causes in the FTA
    # Rows corresponds with the different rifs in the  component in the order of RIF_x
    RIF_bearing = np.array(["Bearing_Design","Surface Hardness","Surface Roughness" , "Material Quality"])
    bearing_causeMat=[[14],[13],[13],[14]]

    RIF_bearinglube = np.array(["Grease Quality", "Contamination"])
    bearinglube_causeMat=[[12,13,14],[13]]

    RIF_gear = np.array(["Gear Design","Surface Roughness","Surface Hardness", "Material Quality"])
    gear_causeMat=[[10,11,16],[9,10,15],[10,15],[10,11,16]]

    RIF_gearlube = np.array(["Lubricant Quality","Contamination"])
    gearlube_causeMat=[[9,10,15],[9]]

    RIF_others = np.array(["External Vibration", "Temperature" , "Environment", "Filter_Design"])
    others_causeMat=[[7,9,12],[5],[4,5],[4]]
    
    # allRIFs is array of RIF arrays
    # cause mat is  3d array of individual 2d arrays
    allRIFs=[RIF_bearing,RIF_bearinglube,RIF_gear,RIF_gearlube,RIF_others]
    causemat=[bearing_causeMat,bearinglube_causeMat,gear_causeMat,gearlube_causeMat,others_causeMat]
    
    component= [None]*len(components)
    
    #assign values to the component_rif classes    
    for x in range((len(components))):
            component[x]=RIF(components[x],allRIFs[x], lbls_name,causemat[x])
    
    return component


# In[437]:


######## Create a visal chart betwen failure causes and RIF #########
#similar to gearbox fta function uses dot diagrm to create a visual network of component-RIF and corresponding causes
# uses the nested class data
# from graphviz import Digraph
# import pylab
# lbls_name, matrix, basic_index = creatingArray_Quantitative()
# component=nestingRIFs_Quantitative(lbls_name)
# # component=nestingRIFs_Qualitative()
# dot2=Digraph()
# for x in range(len(component)):
#     x=2
#     for y in range(len(component[x].allRIFs)):
#         dot2.edge(component[x].name,component[x].allRIFs[y])
#         for z in range(len(component[x].RIF[y].cause)):
#             dot2.edge(component[x].allRIFs[y],component[x].RIF[y].cause[z])
#     break
# display(dot2)
# filename = dot2.render(filename='img/g1')

# pylab.savefig('filename.png')



# In[438]:


#-------------------------------------------------------------------------#
########----------- 3. Cost of Failure -----------########
#-------------------------------------------------------------------------#


# In[439]:


# gear,bearing,lube stuff are the 3 subassemblies considered for failure and costs are attached
# takes failure rate in order- gear,bearing,lubrication
# only considers these 3


#----IMP---#
# the costs are onshore costs since cost distribution from 2012 paper are for onshore paper #

def Cost_calc(fail_rate):
    # the order is gear,bearing,lube
    cost_component=np.array([445000,144000,4000]) # in usd/failure from onshore installations in 2010 USA
    cost_crane=np.array([300000,300000,0]) # in usd/event from 2012 NREL paper
    cost_labor=np.array([18000,8000,500]) # in usd/failure for USA 2012
    
    inflation=1.19169 #Adjust for inflation based on US CPI ---https://www.calculator.net/inflation-calculator.html?cstartingamount1=1&cinyear1=2010&coutyear1=2020&calctype=1&x=96&y=6
    USD2EUR=0.813 #usd to eur in 2020
    
    # energy cost = power generated * cost of energy
    # powergenerated=  #average power = product of capcaity factor and power output
    # https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Electricity_price_statistics    

    cost_energysweden=0.06  #in Eur/kWh for sweden, without taxes from Eurostats(lowest prices in europ)
    capacity_factor=0.41 #from 2018 cost of wind energy for 2.4MW onshore wind turbines
    rated_power=2500 # in kw   

    power_tariff= cost_energysweden*capacity_factor*rated_power #in Eur/hr
    downtime=np.array([260,260,50]) # in hours/failure need to find a better source, right now from tazi
    # note downtime is MTTR + time lost in logistics and oher stuff 
    cost_lossofprod=downtime*power_tariff

    cost_total = cost_lossofprod + ((cost_labor + cost_component+cost_crane)*inflation*USD2EUR) # in Eur/failure adjusted for costs in 2020

    ans=sum(cost_total*fail_rate)
    return ans


# In[440]:


#retunrs the array of costs for pie chart contribution, I was too lay to fix the function cost_calc
def Cost_calc_array(fail_rate):
    cost_component=np.array([445000,144000,4000]) # in usd/failure from onshore installations in 2010 USA
    cost_crane=np.array([300000,300000,0]) # in usd/event from 2012 NREL paper
    cost_labor=np.array([18000,8000,500]) # in usd/failure for USA 2012
    
    inflation=1.19169 #Adjust for inflation based on US CPI ---https://www.calculator.net/inflation-calculator.html?cstartingamount1=1&cinyear1=2010&coutyear1=2020&calctype=1&x=96&y=6
    USD2EUR=0.813 #usd to eur in 2020
    
    # energy cost = power generated * cost of energy
    # powergenerated=  #average power = product of capcaity factor and power output
    # https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Electricity_price_statistics    

    cost_energysweden=0.06  #in Eur/kWh for sweden, without taxes from Eurostats(lowest prices in europ)
    capacity_factor=0.41 #from 2018 cost of wind energy for 2.4MW onshore wind turbines
    rated_power=2500 # in kw   

    power_tariff= cost_energysweden*capacity_factor*rated_power #in Eur/hr
    downtime=np.array([260,260,50]) # in hours/failure need to find a better source, right now from tazi
    # note downtime is MTTR + time lost in logistics and oher stuff 
    cost_lossofprod=downtime*power_tariff

    cost_total = cost_lossofprod + ((cost_labor + cost_component+cost_crane)*inflation*USD2EUR) # in Eur/failure adjusted for costs in 2020
    
    ans=cost_total*fail_rate
    return ans


# In[441]:


#-------------------------------------------------------------------------#
########----------- 4. GUI -----------########
#-------------------------------------------------------------------------#


# In[442]:


########-----------for plots and charts-----------########


# In[443]:


# 'increment' function returns a array of failure rate improvement based on input array of changes 
# 'xaxis' is the percentage change
# 'indices' of basic_causes that are requested to be changed 
def increments(xAxis, indices,prob,matrix):
    Combined_fail=[] #intended to have the updated gearbox failure rate
    initial_fail=(calc_failure_Quantitative(prob,matrix)[0]) #with no changes
    for x in indices:
        failrate_temp=[]
        for y in xAxis:
            prob_temp= np.copy(prob)
            prob_temp[x]=prob_temp[x] * (1+y/100)
            fail=calc_failure_Quantitative(prob_temp,matrix)
            failrate_temp.append(fail[0])
        Combined_fail.append(failrate_temp)
    
    improvement=(Combined_fail-initial_fail)/initial_fail*100 
        
    return improvement


# In[444]:


# 'increment' function returns a array of failure rate improvement based on input array of changes 
# 'xaxis' is the percentage change
# 'indices' of basic_causes that are requested to be changed 
def increments_cost(xAxis, indices,prob,matrix):
    Combined_fail=[] #intended to have the updated gearbox failure rate
    initial_fail=calc_failure_Quantitative(prob,matrix) #with no changes
    initial_cost=Cost_calc(initial_fail[1:4])
    for x in indices:
        failrate_temp=[]
        for y in xAxis:
            prob_temp= np.copy(prob)
            prob_temp[x]=prob_temp[x] * (1+y/100)
            fail=calc_failure_Quantitative(prob_temp,matrix)
            cost=Cost_calc(fail[1:4])
            failrate_temp.append(cost)
        Combined_fail.append(failrate_temp)
    
    improvement=(Combined_fail-initial_cost)/initial_cost*100 
        
    return improvement


# In[445]:


# 'increment' function returns a array of failure rate improvement based on input array of changes 
# 'xaxis' is the percentage change
# 'indices' of basic_causes that are requested to be changed 
def Qincrements(xAxis, indices,prob,matrix):
    Combined_fail=[] #intended to have the updated gearbox failure rate
    initial_fail=prob[0] #with no changes
            
    # loop runs for the number of cause changes called
    for x in indices:
        failrate_temp=[]
        for y in xAxis:
            change=[1]*len(matrix[0])
            if isinstance(x,int):
                change[x] = 1+y/100
            else:
                for z in x:
                    change[z] = 1+y/100
            fail=calc_failure_Qualitative(change,matrix)
            failrate_temp.append(fail[0])
        Combined_fail.append(failrate_temp)
    
    improvement=(Combined_fail-initial_fail)/initial_fail*100 
        
    return improvement


# In[446]:


# 'increment' function returns a array of failure rate improvement based on input array of changes 
# 'xaxis' is the percentage change
# 'indices' of basic_causes that are requested to be changed 
def Qincrements_cost(xAxis, indices,prob,matrix):
    Combined_fail=[] #intended to have the updated gearbox failure rate
    initial_fail=prob[0] #with no changes
    initial_cost=Cost_calc(prob[1:4])
    
    # loop runs for the number of cause changes called
    for x in indices:
        failrate_temp=[]
        for y in xAxis:
            change=[1]*len(matrix[0])
            if isinstance(x,int):
                change[x] = 1+y/100
            else:
                for z in x:
                    change[z] = 1+y/100
            fail=calc_failure_Qualitative(change,matrix)
            cost=Cost_calc(fail[1:4])
            failrate_temp.append(cost)
        Combined_fail.append(failrate_temp)
    
    improvement=(Combined_fail-initial_cost)/initial_cost*100 
    return improvement


# In[447]:


#plots the output from 'increment' fucntion which is taken as values
# xAxis is taken as x and the label is for the legend
# ########--- can also add title ---#########
def PyplotSimple(values,x, labels,naming):
    i=0
    plt.figure(figsize=(10,10))
     
    arr=np.array(values[:,-1])
    max_index=arr.argsort()[-5:][::-1]
    temp_labels=[None]*len(labels)
    #     temp_labels[int(max_index)]=labels[int(max_index)]
    for temp in max_index:
        temp_labels[temp]=labels[temp]
    for y in values:
    #         plt.plot(x,y, label=labels[i])
        plt.plot(x,y, label=temp_labels[i])
        i=i+1

    plt.xlabel(naming[2])
    plt.ylabel(naming[1])
    plt.title(naming[0])
    plt.legend()
    plt.grid()
    #     plt.show()
    fig = plt.gcf()  # get the figure to show
    return fig


# In[448]:


# specifically used to create pie chart of failure rate contribution
# need to change this function if more than 3 assembly componnet exsists

def CostPieChartContribution(prob,matrix, lbls_name):
    import matplotlib.pyplot as plt
    labels = lbls_name[1:4]
    
    failureRate=calc_failure_Quantitative(prob,matrix)
    temp_input=Cost_calc_array(failureRate[1:4])

    # Creating plot
    fig = plt.figure(figsize =(10, 7))
    plt.pie(temp_input, labels = labels)
    
    plt.legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize='small')
    plt.title('Annual Cost due to gearbox failure: '+ str(int(sum(temp_input)))+' Eur')
    plt.grid()

    # show plot
    plt.show()
        
    return fig


# In[449]:


# specifically used to create pie chart of failure rate contribution
# need to change this function if more than 3 assembly componnet exsists

def PieChartContribution(prob,matrix, lbls_name):
    import matplotlib.pyplot as plt
    labels = [None]*4

    # get ratios of contribution for F0, M1,M2 and M3
    # to be used to generate pie charts of contributions     
    ratios_temp=[0]*4 #ratio of failure mode failurer rates
    ratios=[0]*4
    failureRate=calc_failure_Quantitative(prob,matrix)
    temp_input=failureRate
    
    for x in range(0,4):
        ratios_temp[x]= temp_input[list(np.nonzero(matrix[x])[0])]
        ratios[x]= 100*ratios_temp[x]/np.sum(ratios_temp[x])
    
    fracs=ratios
    
    # Make figure and axes
    fig, axs = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    # sets size and number of sub charts
    radii=1 # radii of pies
    # A standard pie plot
    
    for x in range(0,4):
        labels[x] = lbls_name[list(np.nonzero(matrix[x])[0])] # takes label name index from non zero matrix elements

    # properties for all 4 pies, called by axs(x,y).pie , fracs is the ratio and title is called from lbls 
    axs[0,0].pie(fracs[0], autopct='%.0f%%', shadow=True, radius=radii)
    axs[0,0].set_title(lbls_name[0], weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
    legend = axs[0,0].legend(labels[0], loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    axs[1,1].pie(fracs[1], autopct='%.0f%%', shadow=True, radius=radii)
    axs[1,1].set_title(lbls_name[1], weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
    legend = axs[1,1].legend(labels[1], loc=(0.9, .95), labelspacing=0.1, fontsize='small')


    axs[1,0].pie(fracs[2], autopct='%.0f%%', shadow=True, radius=radii)
    axs[1,0].set_title(lbls_name[2], weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
    legend = axs[1,0].legend(labels[2], loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    
    axs[0,1].pie(fracs[3], autopct='%.0f%%', shadow=True, radius=radii)
    axs[0,1].set_title(lbls_name[3], weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
    legend = axs[0,1].legend(labels[3], loc=(0.9, .95), labelspacing=0.1, fontsize='small')
        
    fig.tight_layout()
        
    return fig


# In[450]:


# doesnt work right now properly but intended to show the plots in a seperate figure in the UI
# plotselect basically prints out one of the two plot types, this needs to be iterated in the calling function
# for now vary the plot select from 0 to 1 to see the plot in the output window

def MakingPlots(plotselect):    
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    def delete_figure_agg(figure_agg):
        figure_agg.get_tk_widget().forget()
        plt.close('all')
    #------------------------------------------------------------------------------------#    
    if 3 <= plotselect <=8:
        lbls_name, matrix, basic_index = creatingArray_Quantitative()
        prob=Prob_Quantitative(matrix)
        component=nestingRIFs_Quantitative(lbls_name)
    elif 9 <= plotselect <=13:
        lbls_name, matrix, basic_index = creatingArray_Qualitative()
        prob=Prob_Qualitative(matrix)
        component=nestingRIFs_Qualitative(lbls_name)        
        
        
    indices=[]
    for x in range(len(component)):
        temp_index=[]
        for y in range(len(component[x].allRIFs)):
            temp_index.extend(component[x].RIF[y].causeindex)
        indices.append(temp_index)

    labels2=[]
    for x in component:
        labels2.append(x.name)
    
    xAxis=np.linspace(0,20,20)
    
    #plotnaming gives plot title,y axis label and x axis label respectively
    Plotnaming_rate_cause=['Sensitivity: Failure Rate vs Failure Cause','Gearbox Failure Rate Change(%)','Failure Cause Failure Rate Change(%)']
    Plotnaming_rate_component=['Sensitivity: Failure Rate vs Component Failure','Gearbox Failure Rate Change(%)','Component Failure Rate Change(%)']
    Plotnaming_cost_cause=['Sensitivity: Failure Cost vs Failure Cause','Change in Annual Cost of Gearbox Failure(%)','Failure Cause Failure Rate Change(%)']
    Plotnaming_cost_component=['Sensitivity: Failure Cost vs Component Failure','Change in Annual Cost of Gearbox Failure(%)','Component Failure Rate Change(%)']

    if plotselect == 3:
        Figure = PieChartContribution(prob,matrix, lbls_name)
    #         return PieChartCon/tribution(prob,matrix, lbls_name) 

    elif plotselect == 4:
        PlotPoints=increments(xAxis, basic_index,prob,matrix)
        Figure = PyplotSimple(PlotPoints,xAxis, lbls_name[basic_index],Plotnaming_rate_cause)
    #         return PyplotSimple(PlotPoints,xAxis, lbls_name[basic_index])
    
    elif plotselect == 5:
        PlotPoints2=increments(xAxis, indices,prob,matrix)
        Figure = PyplotSimple(PlotPoints2,xAxis, labels2, Plotnaming_rate_component)
    #         return PyplotSimple(PlotPoints2,xAxis, labels2)

    elif plotselect == 6:
        Figure = CostPieChartContribution(prob,matrix, lbls_name)
    #         return PieChartCon/tribution(prob,matrix, lbls_name)
        
    elif plotselect == 7:
        PlotPoints=increments_cost(xAxis, basic_index,prob,matrix)
        Figure = PyplotSimple(PlotPoints,xAxis, lbls_name[basic_index], Plotnaming_cost_cause)
    #         return PyplotSimple(PlotPoints,xAxis, lbls_name[basic_index])
    
    elif plotselect == 8:
        PlotPoints2=increments_cost(xAxis, indices,prob,matrix)
        Figure = PyplotSimple(PlotPoints2,xAxis, labels2, Plotnaming_cost_component)
    #         return PyplotSimple(PlotPoints2,xAxis, labels2)

    elif plotselect == 9:
        PlotPoints=Qincrements(xAxis, basic_index,prob,matrix)
        Figure = PyplotSimple(PlotPoints,xAxis, lbls_name[basic_index], Plotnaming_rate_cause)
    #         return PyplotSimple(PlotPoints,xAxis, lbls_name[basic_index])
    
    elif plotselect == 10:
        PlotPoints2=Qincrements(xAxis, indices,prob,matrix)
        Figure = PyplotSimple(PlotPoints2,xAxis, labels2, Plotnaming_rate_component)
    #         return PyplotSimple(PlotPoints2,xAxis, labels2)
    elif plotselect == 11:
        PlotPoints=Qincrements_cost(xAxis, basic_index,prob,matrix)
        Figure = PyplotSimple(PlotPoints,xAxis, lbls_name[basic_index], Plotnaming_cost_cause)
    #         return PyplotSimple(PlotPoints,xAxis, lbls_name[basic_index])
    
    elif plotselect == 12:
        PlotPoints2=Qincrements_cost(xAxis, indices,prob,matrix)
        Figure = PyplotSimple(PlotPoints2,xAxis, labels2, Plotnaming_cost_component)
    #         return PyplotSimple(PlotPoints2,xAxis, labels2)
    
    layout = [[sg.Canvas(size=(200,200),key='-CANVAS-')],[sg.Button('Return to Main', font='Any 10',size=(15, 1))]]

    # create the form and show it without the plot
    window = sg.Window('Matplotlib Single Graph', layout, location=(0,0), finalize=True, element_justification='center', font='Helvetica 18')
    figure_agg = None

    draw_figure(window['-CANVAS-'].TKCanvas, Figure)

    while True:
        event, values = window.read()
        if event is None:
            break
        
        if figure_agg:
            # ** IMPORTANT ** Clean up previous drawing before drawing again
            delete_figure_agg(figure_agg)
   
        figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, Figure )  # draw the figure

        if event == 'Return to Main':
            delete_figure_agg(figure_agg)
            window.close()


# In[451]:


########-----------Actual GUI code starts here-----------########


# In[452]:

# Function output_stuff creates the gui that outputs the resultant failure rate improvenet based on user changes requested in
# the functiton getuserinput. An option to save the results is also provided
# outputstaff takes input of user_indices(indices of component,rif,cause chosen by the user) and user_change(change in percentage for all the causes)

def output_windowQuantitative(user_indices , user_change, component,prob,matrix):
    # pysimplegui is used as the addon for GUI
   
   # sg.change_look_and_feel('GreenTan') # give our window a spiffy set of colors
    my_new_theme = {'BACKGROUND': '#65656c',
                    'TEXT': 'black',
                    'INPUT': 'white',
                    'TEXT_INPUT': 'black',
                    'SCROLL': '#c7e78b',
                    'BUTTON': ('white', '#1954a6'),
                    'PROGRESS': ('#01826B', '#D0D0D0'),
                    'BORDER': 1,
                    'SLIDER_DEPTH': 0,
                    'PROGRESS_DEPTH': 0}

    sg.theme_add_new('MyNewTheme', my_new_theme)

    sg.theme('My New Theme')
    
    failureRate=calc_failure_Quantitative(prob,matrix)

    prob_updated=np.copy(prob)
    
    #temparory variable 'a' would be string type and be output on the window
    a='Analysis Type: Quantitative\n'
    
    # loop runs for the number of cause changes called
    for x in range(len(user_change)):
        user=user_indices[x]
        a= a + str(component[user[0]].RIF[user[1]].cause[user[2]] + " is modified by : " + user_change[x] + "%") +'\n'
        # prob_updated changes the requested cause probability by the reuested amount
        prob_updated[component[user[0]].RIF[user[1]].causeindex[user[2]]] = prob_updated[component[user[0]].RIF[user[1]].causeindex[user[2]]] * (1+float(user_change[x])/100)
    
    failureRate_updated=calc_failure_Quantitative(prob_updated,matrix)
    a= a + "Original Gearbox Failure Rate: " + str(round(failureRate[0],3))+ '\n'
    a= a + "Modified Gearbox Failure Rate: " + str(round(failureRate_updated[0],3)) + '\n'
    improvement=(Cost_calc(failureRate[1:4])-Cost_calc(failureRate_updated[1:4])) #net differnece in updated to original cost
    perc_improvement=improvement*100/Cost_calc(failureRate[1:4])
    a= a+ "Net improvement: " + str(round(improvement,3)) + " EUR/turbine/year" + '\n'
    a= a+ "Net percentage improvement in cost: " + str(round(perc_improvement,3)) + " %" + '\n'+ '\n'
    
    # multiline is used for output
    # layout decribes the elements of the GUI
    MLINE_KEY = '-MLINE-'+sg.WRITE_ONLY_KEY
    layout = [  [sg.Multiline(a,size=(60,20), key=MLINE_KEY)],
                [sg.Button('Save Results and Exit'), sg.Button('Disregard Results and Exit')] ]

    #comnd generate and names the window
    window = sg.Window('Calculation of Results', layout, font=('Helvetica', ' 13'), default_button_element_size=(8, 2))

    while True:     # The Event Loop
        event, value = window.read()
        if event is None:
            break
        if event == 'Disregard Results and Exit':            
            window.close()
        if event == 'Save Results and Exit':
            window.close()
            return a
        #if save result selected, the widow closes nd the content of output window 'a' is returned


# In[453]:
# Function output_stuff creates the gui that outputs the resultant failure rate improvenet based on user changes requested in
# the functiton getuserinput. An option to save the results is also provided
# outputstaff takes input of user_indices(indices of component,rif,cause chosen by the user) and user_change(change in percentage for all the causes)

def output_windowQualitative(user_indices , user_change,component,prob,matrix):
    # pysimplegui is used as the addon for GUI
    
    #     sg.change_look_and_feel('GreenTan') # give our window a spiffy set of colors
    my_new_theme = {'BACKGROUND': '#65656c',
                    'TEXT': 'black',
                    'INPUT': 'white',
                    'TEXT_INPUT': 'black',
                    'SCROLL': '#c7e78b',
                    'BUTTON': ('white', '#1954a6'),
                    'PROGRESS': ('#01826B', '#D0D0D0'),
                    'BORDER': 1,
                    'SLIDER_DEPTH': 0,
                    'PROGRESS_DEPTH': 0}

    sg.theme_add_new('MyNewTheme', my_new_theme)

    sg.theme('My New Theme')

    #     prob_updated=np.copy(prob)
    
    change=[1]*len(matrix[0])
    
    #temparory variable 'a' would be string type and be output on the window
    a='Analysis Type: Qualitative\n'
    
    # loop runs for the number of cause changes called
    for x in range(len(user_change)):
        user=user_indices[x]
        a= a + str(component[user[0]].RIF[user[1]].cause[user[2]] + " is modified by : " + user_change[x] + "%") +'\n'
        # prob_updated changes the requested cause probability by the requested amount
        change[component[user[0]].RIF[user[1]].causeindex[user[2]]] = 1+float(user_change[x])/100
        
    failureRate=prob
    failureRate_updated=calc_failure_Qualitative(change,matrix)
    a= a + "Original Gearbox Failure Rate: " + str(round(failureRate[0],3))+ '\n'
    a= a + "Modified Gearbox Failure Rate: " + str(round(failureRate_updated[0],3)) + '\n'    
    improvement=(Cost_calc(failureRate[1:4])-Cost_calc(failureRate_updated[1:4]))
    perc_improvement=improvement*100/Cost_calc(failureRate[1:4])
    a= a+ "Net improvement in cost: " + str(round(improvement,3)) + " EUR/turbine/year" + '\n'
    a= a+ "Net percentage improvement in cost: " + str(round(perc_improvement,3)) + " %" + '\n'+ '\n'

    # multiline is used for output
    # layout decribes the elements of the GUI
    MLINE_KEY = '-MLINE-'+sg.WRITE_ONLY_KEY
    layout = [  [sg.Multiline(a,size=(60,20), key=MLINE_KEY)],
                [sg.Button('Save Results and Exit'), sg.Button('Disregard Results and Exit')] ]

    #comnd generate and names the window
    window = sg.Window('Calculation of Results', layout, font=('Helvetica', ' 13'), default_button_element_size=(8, 2))

    while True:     # The Event Loop
        event, value = window.read()
        if event is None:
            break
        if event == 'Disregard Results and Exit':            
            window.close()
        if event == 'Save Results and Exit':
            window.close()
            return a
        #if save result selected, the widow closes nd the content of output window 'a' is returned

# In[454]:
# GUI to take user input for change basic cause probability
# allows multiple causes to be changed at once
def GetUserInput(component,prob,matrix,analysis_type): #the varibale analysis type is used to select quantitative or qualitative form of data
    #     sg.change_look_and_feel('Light Green 1')
    my_new_theme = {'BACKGROUND': '#65656c',
                    'TEXT': 'black',
                    'INPUT': 'white',
                    'TEXT_INPUT': 'black',
                    'SCROLL': '#c7e78b',
                    'BUTTON': ('white', '#1954a6'),
                    'PROGRESS': ('#01826B', '#D0D0D0'),
                    'BORDER': 1,
                    'SLIDER_DEPTH': 0,
                    'PROGRESS_DEPTH': 0}

    sg.theme_add_new('MyNewTheme', my_new_theme)

    sg.theme('My New Theme')

    # Table generation code using pysimplegui has been taken from https://pysimplegui.trinket.io/demo-programs#/tables/the-table-element
    
    # ------ Some functions to help generate data for the table ------
    def word():
        return ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    def number(max_val=1000):
        return random.randint(0, max_val)
    

    # ------ Make the Table Data ------
    # data = data_forGUI has 3 columns: component, RIF and cause
    # data_index just carries the indices for each row
    data_forGUI=[]
    data_index=[]
    for x in range(len(component)):
        for y in range(len(component[x].allRIFs)):
            for z in range(len(component[x].RIF[y].cause)):
                data_forGUI.append([component[x].name , component[x].allRIFs[y], component[x].RIF[y].cause[z]])
                data_index.append([x,y,z])
    
    data = data_forGUI
    headings = ['Component','RIF','Cause']
    
    # for saving table data in csv file
    #     df = pd.DataFrame(data)
    #     df.to_csv('newdata.csv')


    # KTH colors
    # Primary colour: #1954a6 - dark blue
    # secondary clour: #249fd8 - blue
    # secondary clour: #d85496 - pink
    # secondary clour: #b0c92b - green
    # secondary clour: #65656c - grey

    # ------ Window Layout ------
    # window divided into 2 columns, column1 just has the table, coulmn 2 has the user selections   
    cl1_layout = [[sg.Table(values=data, headings=headings, max_col_width=25, background_color='lightblue',
                        auto_size_columns=True,
                        display_row_numbers=True,
                        justification='right',
                        num_rows=20,
                        alternating_row_color='lightyellow',
                        key='-TABLE-',
                        tooltip='RIFs and Causes')]]
    
    #Table data has an extra column of row no. which would be used to select the causes by the user
    
    #rows and change array hold the user selected values of rows and change respetably
    rows_array=[]
    change_array=[]
    headings_table2=['Chosen Row','Chosen Change','a']

    cl2_layout=[
                [sg.Text('Use Row numbers in adjacent table to select specifc cause and type the corresponding change requested', size=(40,3))],
                [sg.Text('Row No.', size=(15,1)),sg.Text('% Change', size=(20,1))],
                [sg.Input([], size=(15,1), key='rows'),sg.Input([], size=(20,1), key='change')],[sg.Button('Add Another'), sg.Button('Reset')],
                [sg.HorizontalSeparator()],
                [sg.Text('Selected Values: ')],[sg.Text('Selected Rows: ', size=(15,1)),sg.Text('Selected Change', size=(20,1))],
                [sg.Listbox(values=rows_array,size=(10,4), key='-rowDisplay-'), sg.Listbox(values=change_array,size=(20,4), key='-changeDisplay-')],
                [sg.Button('Calculate'), sg.Button('Exit')]
               ]
    # Listboxes shows the user selection
    # 'reset' clears user selection
    # 'Add Another' adds another cause and change value to the selection
    # 'calculate' opens output_stuff function to display the results
    
    
    #for info on failure causes, not done for qualitative yet
    #     Info_Quanti_index=[0,1,0,0,1,2,2,3,4,5,6,6,6,3,5,7,6,10,14,7,11,10,12,11,1,12]
    
    # info box in window2, Info_Quanti_index maps the info for failure causes ie. the table rows and 'Info_Quanti'
    Info_Quanti_index=[0,2,2,0,1,2,0,2,4,5,7,3,4,6,4,6,4,5,7,3,4,6,3,8,3,1,9,10,9,10]    
    Info_Quanti=['Surface Fatigue:\n Surface faitigue leads to formation of tiny fissures under the surface where the highest shear stress occurs. under repetting loading, fissure grows to the surface untill plate shaped particles of material come loose, formings pits. seavere pitting is referred to as spalling or delamination. ' , 
                 'Corrosion of Pins:\n Corrosion mainly contains two types which are moisture corrosion and frictional corrosion. The moisture corrosion happens when a bearing is in contact with moisture (water or acid), while frictional corrosion is activated by relative movements between mating surfaces given certain friction conditions. Friction corrosion include freeting and false brinelling.  ' , 
                 'Abrasive Wear(Bearing):\n Abrasive wear is also called particle wear or three body wear. The most intensive abrasive wear happens when a soft surface is cut by another hard surface. The second reason occurs when a third hard particle abrades another soft surface. The mechanisms for abrasive wear include microcutting, microfracture, pull-out and individual grains. Therefore, lubricant failures and contamination are incentives to cause wind turbine bearings abrasive wear. ' , 
                 'Scuffing:\n Adhesive wear is the transfer of material from one surface to another through welding and tearing. If the lubricant cannot prevent metal-to-metal contact and sliding motion occurs under load, substantial frictional heat will be generated. This can cause momentary micro-welding and tearing as gears mate and separate. It occurs on corresponding tooth tips and roots and has a scuffed, torn, shiny silver or black appearance ' , 
                 'Gear Pitting:\n A problem with pitting can be labeled as either initial, in which the surface is experiencing small pits to destructive, in which the pits are larger in diameter. Initial pitting may be a problem with the gears not fitting together properly. Destructive pitting is typically an issue with surface overload. ' , 
                 'Gear Crack:\n Cracks can occur due to high temperature, high pressure, faituge or a combination of all three. Fatigue occurs over time in response to repetitive loading. Due to the application of load, the gear tooth is subject to bending leading to fatigue. The fatigues in the gear cause the formation of cracks in the root of the gear tooth, leading to the failure of gear tooth. ' , 
                 'Gear tooth detioration(wear):\n This type of where leaves contacts patterns that show the metal has been affected in the addendum and the dedendum area. Issues with inadequate lubrication commonly cause it, but it may also be due to contamination in the lubrication as well. ' , 
                 'Gear teeth offset:\n design and manfuacturing errors can lead to the gears not being perfectly aligned thus increasign the local pressure thus enabling failure',
                 'Abnormal Vibration:\n ',
                 #                  'Excess Temperature:\n Overheating is one of the results of mounting failure generated within the bearing itself which happens on rings, balls and cages. The symptoms are discoloration from gold to blue. The incentives for overheating contain heavy loads, inadequate heat paths and poor cooling systems. As a result, both transmission system bearings and adjustment system bearings can easily be subjected to this kind of failure. The inappropriate treatment of overheating has the potential to damage or even explode the wind turbine. ' , 
                 'Poor quality of lubrication oil:\n Lubricant contamination is another major cause of large-scale wind turbine bearing failures which can dent the bearing raceways. Wear debris, residual particles, dirty lubricants and water are the most common particles, dirty lubricants and water are the most common sources of contaminations ' , 
                 'Abnormal Filter:\n  ' , 'Dirt:\n  ' , 'Abnormal Vibration:\n  ']    


    col1= sg.Column(cl1_layout)
    col2= sg.Column(cl2_layout)
    
    #to show the fault tree of the selected analysis
    if analysis_type == 'Quantitative':
        title= sg.Text('Reliability Analysis(Kang-2019)', font=("Helvetica", 18), size=(55,1))
        FTA = 'KangFTA.png'
        
    else:
        title= sg.Text('Reliability Analysis(Bhardwaj-2019)', font=("Helvetica", 18),size=(60,1))
        FTA = 'BhardwajFTA.png'

    layout= [[title,sg.Button('View Fault Tree')],[sg.HorizontalSeparator()],[col1,col2],
                 [sg.Button('Info'), sg.Text('For More info about failure causes')],
                 [sg.Multiline('',size=(120,5),key='-MlineINFO-')]]
    
    results = ''
    # ------ Create Window ------
    window = sg.Window('RIFs and Causes', layout)

    selectedrow=None
    # ------ Event Loop ------
    # win2_active = False #this if to create the pop up kind of window for Fault tree
    while True:
        event, values = window.read(timeout=100)
        if event is None:
            break
    #       sg.PopupAnimated()

        if event == 'Info':
            selectedrow=values['-TABLE-'][0]
            window['-MlineINFO-'].update(Info_Quanti[Info_Quanti_index[selectedrow]])
        
        if event == 'Add Another':
            rows_array.append(values['rows'])
            change_array.append(values['change'])
            window.Element('-rowDisplay-').Update(rows_array) #update updates the window element
            window.Element('-changeDisplay-').Update(change_array)
            
        if event == 'Reset':
            rows_array=[]
            change_array=[]
            window.Element('-rowDisplay-').Update(rows_array)
            window.Element('-changeDisplay-').Update(change_array)     
        
        if event == 'Calculate':
            indices = []
            for x in range(len(rows_array)):
                indices.append(data_index[int(rows_array[x])])
            outputfunction=eval("output_window"+ analysis_type + "(indices,change_array,component,prob,matrix)")
            if outputfunction is not None:
                results= results + outputfunction
        
        # if event == 'View Fault Tree' and not win2_active:
        #     win2_active = True
            # layoutFTA=[[sg.Image(FTA)]]
            # window2 = sg.Window('Window 2', layoutFTA, grab_anywhere=True)
        
        # if win2_active: #apparaently this is needed for the pop up fault tree
        #     event, values = window2.read(timeout=100)
    #             if event != sg.TIMEOUT_KEY:
    #                 print("win2 ", event)
            if event is None:
                win2_active = False
                window2.close()
        
        if event == 'Exit':
            window.close()          
    # Results returns the 'saved content' of the output windw to be saved and used later
    return results


# In[455]:


def Quantitative_analysis():
    lbls_name, matrix, basic_index = creatingArray_Quantitative()
    prob=Prob_Quantitative(matrix)
    component=nestingRIFs_Quantitative(lbls_name)
    Results = GetUserInput(component,prob,matrix,"Quantitative")
    return Results

def Qualitative_analysis():
    lbls_name, matrix, basic_index = creatingArray_Qualitative()
    prob=Prob_Qualitative(matrix)
    component=nestingRIFs_Qualitative(lbls_name)
    Results = GetUserInput(component,prob,matrix,"Qualitative")
    return Results

# In[456]:


def ResultLog(a):
    import PySimpleGUI as sg
    
    sg.change_look_and_feel('GreenTan') # give our window a spiffy set of colors

    # multiline is used for output
    # layout decribes the elements of the GUI
    MLINE_KEY = '-MLINE-'+sg.WRITE_ONLY_KEY
    layout = [  [sg.Multiline(a,size=(60,25), key=MLINE_KEY)],
                [sg.Button('Exit')] ]

    #comnd generate and names the window
    window = sg.Window('Logged Results', layout, font=('Helvetica', ' 13'), default_button_element_size=(8, 2))

    while True:     # The Event Loop
        event, value = window.read()
        if event is None:
            break
        if event == 'Exit':            
            window.close()
        #if save result selected, the widow closes nd the content of output window 'a' is returned


# In[457]:
# the First window code. ALl the stuff starts from here
sg.change_look_and_feel('GreenTan')

# Primary colour: #1954a6 - dark blue
# secondary clour: #249fd8 - blue
# secondary clour: #d85496 - pink
# secondary clour: #b0c92b - green
# secondary clour: #65656c - grey

my_new_theme = {'BACKGROUND': '#65656c',
                'TEXT': 'white',
                'INPUT': 'white',
                'TEXT_INPUT': 'white',
                'SCROLL': '#c7e78b',
                'BUTTON': ('white', '#1954a6'),
                'PROGRESS': ('#01826B', '#D0D0D0'),
                'BORDER': 1,
                'SLIDER_DEPTH': 0,
                'PROGRESS_DEPTH': 0}

sg.theme_add_new('MyNewTheme', my_new_theme)

sg.theme('My New Theme')

options=['Kang 2019 - Quantitative FTA','Bhardwaj 2019 - Qualitative FTA', 'Logged Output', 
         'Failure Mode Contribution', 'Sensitivity-Failure vs Cause','Sensitivity-Failure vs Component',
        'Failure Cost Contribution','Sensitivity-Cost vs Cause', 'Sensitivity- Cost vs Component',
        'Sensitivity-Failure vs Cause','Sensitivity-Failure vs Component',
        'Sensitivity-Cost vs Cause', 'Sensitivity- Cost vs Component'] #Radio button names

keys=['Kang','Bhardwaj','Output','contri','sensi-1','sensi-2',
      'contri2','sensi-3','sensi-4','sensi-5','sensi-6','sensi-7','sensi-8'] 

information=['This section of the program uses the work of Kang who developed a Fault tree for WT Gearbox failure rate calculations. Gearbox is divided into 5 components- Gears,Gear Lubricant,Bearings,Bearing Lubricant and others. Reliability Influencing Factors(RIF) for these components are identified that influence the failure causes. User can set potential reliabilty improvements in one or a set of failure causes and get the potential change in cost and reliability for the wind turbine',
            'This section of the program uses the work of Bhardaj et al. who developed an FMEA based WT Gearbox realibilty study. Bhardwaj uses failure rates of gearbox components(beainrgs,shaft,gears,etc) calculated by Smolders 2010 and with the help of qualitative weights connectes component failure rate to failure modes and failure causes. Gearbox is divided into 5 components- Gears,Gear Lubricant,Bearings,Bearing Lubricant and others. Reliability Influencing Factors(RIF) for these components are identified that influence the failure causes. User can set potential reliabilty improvements in one or a set of failure causes and get the potential change in cost and reliability for the wind turbine',
            'This section can be used to view saved results from different iterations of the relability analysis carried out by the user.',
            
            'Analysis of data in Kang et al 2019 is carried out and the contribution of failure rates of components and failure modes on the overall failure of gearbox is represented as a pie chart',
            'Analysis of data in Kang et al 2019 is carried out and sensitivity of failure cause improvement against gearbox failure rate is plotted',
            'Analysis of data in Kang et al 2019 is carried out and sensitivity of improvement in component failure rate(gears,gear lube, bearing,bearing lube,others) is presented against gearbox failure rate'
            
             'Analysis of data in Kang et al 2019 is carried out and the contribution of failure rates of components and failure modes on the overall failure of gearbox is represented as a pie chart',
            'Analysis of data in Kang et al 2019 is carried out and sensitivity of failure cause improvement against gearbox failure rate is plotted',
            'Analysis of data in Kang et al 2019 is carried out and sensitivity of improvement in component failure rate(gears,gear lube, bearing,bearing lube,others) is presented against gearbox failure rate'
            
            'Analysis of data in Bhardwaj et al 2019 is carried out and sensitivity of failure cause improvement against gearbox failure rate is plotted',
            'Analysis of data in Bhardwaj et al 2019 is carried out and sensitivity of improvement in component failure rate(gears,gear lube, bearing,bearing lube,others) is presented against gearbox failure rate'
            'Analysis of data in Bhardwaj et al 2019 is carried out and sensitivity of failure cause improvement against gearbox failure rate is plotted',
            'Analysis of data in Bhardwaj et al 2019 is carried out and sensitivity of improvement in component failure rate(gears,gear lube, bearing,bearing lube,others) is presented against gearbox failure rate' 
            ]
    #information is in the order of the radio button options

link=['Kang, J., Sun, L., & Guedes Soares, C. (2019). Fault Tree Analysis of floating offshore wind turbines. Renewable Energy, 133, 1455-1467. https://doi.org/https://doi.org/10.1016/j.renene.2018.08.097',
     'Bhardwaj, U., Teixeira, A. P., & Soares, C. G. (2019). Reliability prediction of an offshore wind turbine gearbox. Renewable Energy, 141, 693-706. https://doi.org/10.1016/j.renene.2019.03.136 ',
      '']
link_index=[0,1,2,0,0,0,0,0,0,1,1,1,1]
    # link and link index together add the refrenced paper in the info popup

frame1 = [*[[sg.R(f'{options[x]}', 1,key=keys[x])] for x in range(3,6)] ] #Kang failure rate plots
frame2 = [*[[sg.R(f'{options[x]}', 1,key=keys[x])] for x in range(6,9)] ] #Kang failure cost plots
frame3 = [*[[sg.R(f'{options[x]}', 1,key=keys[x])] for x in range(9,11)] ] #Bhardwaj failure rate plots
frame4 = [*[[sg.R(f'{options[x]}', 1,key=keys[x])] for x in range(11,13)] ] #Bhardwaj failure cost plots

layout3 = [[sg.Text('This tool provides multiple ways of analysing wind turbine gearbox reliablity')],
           [sg.HorizontalSeparator()],
           [sg.Text('Component level reliability improvement on LCOE')],
           *[[sg.R(f'{options[x]}', 1,key=keys[x])] for x in range(3)],
           [sg.HorizontalSeparator()],
            [sg.Text('Reliablity Data Analysis from Kang 2019')],
            [sg.Frame('Failure Rate', frame1),sg.Frame('Failure Cost', frame2)],       
           [sg.HorizontalSeparator()],
            [sg.Text('Reliablity Data Analysis from Bhardwaj 2019')],
            [sg.Frame('Failure Rate', frame3),sg.Frame('Failure Cost', frame4)]]       

    # ----------- Create actual layout using Columns and a row of Buttons
layout = [[sg.Column(layout3, key='-COL1-')],
          [sg.Text('',size=(20,1)),sg.Button('Info'),sg.Button('Proceed',size=(10,1)), sg.Button('Exit')],
          [sg.Text('',size=(15,1)),sg.Text('For information on any option, click info after selcting a radio button')]   ]

window = sg.Window('WT Gearbox Relaibility Analysis Tool', layout)

Results=''

layout = 3  # The currently visible layout
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    if event == 'Proceed':
        if values[keys[0]] == True:
            Results = Results + Quantitative_analysis()
        if values[keys[1]] == True:
            Results = Results + Qualitative_analysis()
        if values[keys[2]] == True:
            if Results=='':
                ResultLog('No Logged Results')
            else:
                ResultLog(Results)
        for x in range(3,13):
            if values[keys[x]] == True:
                MakingPlots(x)
    if event == 'Info':
        for x in range(len(keys)):
            if values[keys[x]] == True:
                sg.popup(options[x],(information[x]+'\n'+'\n'+link[link_index[x]]))
            
window.close()
