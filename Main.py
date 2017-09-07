#--------------------------------
import xlrd
import numpy as np
import scipy.stats as stats
import math as math

def calcCov(a,b):
  result  = np.cov(a,b)[0,1]

def createMatrix2(unitv, matrix2):
  main = []
  temp = []
  for i in range(0,49):
    temp.append(unitv[i]).append(matrix2[i])
    main.append(temp)
    temp = []
  return main

def BNGraph(child, main):
  #populate these values into one matrix
  onerow = []
  temp   = [0,0]
  major  = [0,0]
  major  = []
  for row in range(0,2):
    for col in range(0,2):
      for index in range(0,49):
        temp[col]=temp[col]+(main[index][row]*main[index][col])
    major.append(temp)
    temp = [0,0]
  #calculated A


  temp2=[]
  #now calculate Y
  y=[0,0]
  beta=[0,0]
  for l in range(0,2):
    for m in range(0,49):
      a=child[m]
      b=main[m][l]
      y[l]=y[l]+a*b
  beta=np.linalg.solve(major,y) 
  intermed=0
  for index in range(0,49):
    intermed = intermed + ( (beta[0]*main[index][0]) + (beta[1]*main[index][1])- child[index] )**2
  intermed = intermed/49
  #sigma square done


  result=0
  for index in range(0,49):
    result=result+-0.5*math.log(intermed*2*22/7) - 0.5*(( (beta[0]*main[index][0]) + (beta[1]*main[index][1])  \
      - child[index] )**2)/intermed
  return result
    #np.linalg.solve(a,y) you get beta
    #sigma square
    #then find the log likelihood

#-------------------------------------------------------------------------------------------------------------
#-----------------------------------------functions over here-------------------------------------------------
workbook = xlrd.open_workbook('/home/omkar/Downloads/university_data.xlsx')
worksheet = workbook.sheet_by_name('university_data')

num_rows = worksheet.nrows - 1
num_cells = worksheet.ncols - 1
curr_row = -1

rank=[]
uni_name=[]
cs_score=[]
res_over=[]
admin_base=[]
tuition=[]
cov_matrix2=([[[None]*4]*4])
unitv=[1]*49
#--------------------------------all declarations done above this line-------------------------------------------------
while curr_row < num_rows:
  curr_row += 1
  row = worksheet.row(curr_row)
  curr_cell = -1
  while curr_cell < num_cells:
    curr_cell += 1
    cell_value = worksheet.cell_value(curr_row, curr_cell)
    if curr_cell==0:
      rank.append(cell_value)
    elif curr_cell==1:
      uni_name.append(cell_value)
    elif curr_cell==2:
      cs_score.append(cell_value)
    elif curr_cell==3:
      res_over.append(cell_value)
    elif curr_cell==4:
      admin_base.append(cell_value)
    elif curr_cell==5:
      tuition.append(cell_value)


#INDIVIDUAL CALCULATIONS:
#mean
mu1=math.ceil(np.mean(cs_score)*100)/100
mu2=np.around(np.mean(res_over), decimals=2)
mu3=np.around(np.mean(admin_base), decimals=2)
mu4=np.around(np.mean(tuition), decimals=2)

#variance
mult = (49.0/48.0)

var1=np.round(np.var(cs_score)*mult, decimals=2)
var2=np.round(np.var(res_over)*mult, decimals=2)
var3=np.round(np.var(admin_base)*mult, decimals=2)
var4=np.round(np.var(tuition)*mult, decimals=2)

#std deviation
sigma1=np.round(np.std(cs_score), decimals=2)
sigma2=np.round(np.std(res_over), decimals=2)
sigma3=np.round(np.std(admin_base), decimals=2)
sigma4=np.round(np.std(tuition), decimals=2)


#THE BAYESIAN NET part:
array_of_fields=['cs_score','res_over','admin_base','tuition']


#CovarianceMatrix-------------------------------------------------------------------------------------
covarianceMat = [[[None]*4]*4]

covarianceMat[0][0] = np.cov(np.vstack([cs_score,cs_score]))[0,1]
covarianceMat[0][1] = np.cov(np.vstack([cs_score,res_over]))[0,1]
covarianceMat[0][2] = np.cov(np.vstack([cs_score,admin_base]))[0,1]
covarianceMat[0][3] = np.cov(np.vstack([cs_score,tuition]))[0,1]

covarianceMat[1][0] = np.cov(np.vstack([cs_score,res_over]))[0,1]
covarianceMat[1][1] = np.cov(p.vstack([res_over,res_over]))[0,1]
covarianceMat[1][2] = np.cov(np.vstack([res_over,admin_base]))[0,1]
covarianceMat[1][3] = np.cov(np.vstack([res_over,tuition]))[0,1]

covarianceMat[2][0] = np.cov(np.vstack([cs_score,admin_base]))[0,1]
covarianceMat[2][1] = np.cov(np.vstack([res_over,admin_base]))[0,1]
covarianceMat[2][2] = np.cov(.vstack([admin_base,admin_base]))[0,1]
covarianceMat[2][3] = np.cov(np.vstack([admin_base,tuition]))[0,1]

covarianceMat[3][0] = np.cov(np.vstack([cs_score,tuition]))[0,1]
covarianceMat[3][1] = np.cov(np.vstack([res_over,tuition]))[0,1]
covarianceMat[3][2] = np.cov(np.vstack([admin_base,tuition]))[0,1]
covarianceMat[3][3] = np.cov(np.vstack([tuition,tuition]))[0,1]


correlation = np.corrcoef([cs_score, res_over, admin_base, tuition])

norm_cs      = stats.norm.pdf(cs_score, loc=mu1, scale=sigma1)
norm_res     = stats.norm.pdf(res_over, loc=mu2, scale=sigma2)
norm_admin   = stats.norm.pdf(admin_base, loc=mu3, scale=sigma3)
norm_tuition = stats.norm.pdf(tuition, loc=mu4, scale=sigma4)

ll1=0
ll2=0
ll3=0
ll4=0

for iter in range (1, 5):
  if(iter==1):
    for x in norm_cs:
      ll1+=np.log(x)
  elif(iter==2):
    for x in norm_res:
      ll2+=np.log(x)
  elif(iter==3):
    for x in norm_admin:
      ll3+=np.log(x)
  elif(iter==4):
    for x in norm_tuition:
      ll4+=np.log(x)

final = ll1+ll2+ll3+ll4
#final

log1_csTOres     = BNGraph(res_over, createMatrix2(unitv, cs_score))
log2_csTOtuit    = BNGraph(tuition, createMatrix2(unitv, tuition))
log3_tuitTOadmin = BNGraph(admin_base, createMatrix2(unitv, tuition))

logLikelihood    = log1_csTOres + log2_csTOtuit + log3_tuitTOadmin + ll1




# OUTPUT IN FORMAT EXPECTED:

print ("UBitName: omkargur")
print ("personNumber: 50207630")
print ("mu1: %.3f"%mu1)
print ("mu2: %.3f"%mu2)
print ("mu3: %.3f"%mu3)
print ("mu4: %.3f"%mu4)

print ("var1: %.3f"%var1)
print ("var2: %.3f"%var2)
print ("var3: %.3f"%var3)
print ("var4: %.3f"%var4)

print ("sigma1: %.3f"%sigma1)
print ('sigma2: %.3f'%sigma2)
print ('sigma3: %.3f'%sigma3)
print ('sigma4: %.3f'%sigma4)

print ("covarianceMat: ")
print (np.asmatrix(covarianceMat))

print ("correlationMat: ") 
print (correlation)

print ("logLikelihood: ")
print (final)

buntygraph=np.matrix('0 1 0 1;0 0 0 0;0 0 0 0;0 0 1 0')
print ("BNgraph")
print (buntygraph)

print ("BNlogLikelihood")
print (logLikelihood)
