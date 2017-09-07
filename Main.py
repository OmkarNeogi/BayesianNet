#--------------------------------
import xlrd
import numpy as np
import scipy.stats as stats
import math as math

def calcCov(a,b):
  oscar=np.cov(a,b)
  result=oscar[0,1]

def createMatrix2(unitv, matrix2):
  main=[]
  temp=[]
  for i in range(0,49):
    temp.append(unitv[i])
    temp.append(matrix2[i])
    main.append(temp)
    temp=[]
  return main

def BNGraph(child, main):
  #populate these values into one matrix
  import numpy as np
  import math as math
  onerow=[]
  temp=[0,0]
  major=[0,0]
  major=[]
  for row in range(0,2):
    for col in range(0,2):
      for index in range(0,49):
        temp[col]=temp[col]+(main[index][row]*main[index][col])
    major.append(temp)
    temp=[0,0]
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
    intermed=intermed+ ( (beta[0]*main[index][0]) + (beta[1]*main[index][1])- child[index] )**2
  intermed=intermed/49
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
cov_matrix2=([[[],[],[],[]],
  [[],[],[],[]],
  [[],[],[],[]],
  [[],[],[],[]]])
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

#mean
mu1=np.mean(cs_score)
mu2=np.mean(res_over)
mu3=np.mean(admin_base)
mu4=np.mean(tuition)

mu1=math.ceil(mu1*100)/100
mu2=np.around(mu2, decimals=2)
mu3=np.around(mu3, decimals=2)
mu4=np.around(mu4, decimals=2)

#variance,
var1=np.var(cs_score)
var2=np.var(res_over)
var3=np.var(admin_base)
var4=np.var(tuition)

var1=var1*(49.0/48.0)
var2=var2*(49.0/48.0)
var3=var3*(49.0/48.0)
var4=var4*(49.0/48.0)

var1=np.round(var1, decimals=2)
var2=np.round(var2, decimals=2)
var3=np.round(var3, decimals=2)
var4=np.round(var4, decimals=2)

#std deviation
sigma1=np.std(cs_score)
sigma2=np.std(res_over)
sigma3=np.std(admin_base)
sigma4=np.std(tuition)

sigma1=np.round(sigma1, decimals=2)
sigma2=np.round(sigma2, decimals=2)
sigma3=np.round(sigma3, decimals=2)
sigma4=np.round(sigma4, decimals=2)

array_of_fields=['cs_score','res_over','admin_base','tuition']


#CovarianceMatrix-------------------------------------------------------------------------------------
covarianceMat=[[[],[],[],[]],  [[],[],[],[]],  [[],[],[],[]],  [[],[],[],[]]]

X12 = np.vstack([cs_score,res_over])
cov_12 = np.cov(X12)

X13=np.vstack([cs_score,admin_base])
cov_13 = np.cov(X13)

X14=np.vstack([cs_score,tuition])
cov_14 =np.cov(X14)

X23 = np.vstack([res_over,admin_base])
cov_23 = np.cov(X23)

X24=np.vstack([res_over,tuition])
cov_24 = np.cov(X24)

X34=np.vstack([admin_base,tuition])
cov_34 =np.cov(X34)

X11=np.vstack([cs_score,cs_score])
X22=np.vstack([res_over,res_over])
X33=np.vstack([admin_base,admin_base])
X44=np.vstack([tuition,tuition])

cov_11=np.cov(X11)
cov_22=np.cov(X22)
cov_33=np.cov(X33)
cov_44=np.cov(X44)

covarianceMat[0][0]=cov_11[0,1]
covarianceMat[0][1]=cov_12[0,1]
covarianceMat[0][2]=cov_13[0,1]
covarianceMat[0][3]=cov_14[0,1]

covarianceMat[1][0]=cov_12[0,1]
covarianceMat[1][1]=cov_22[0,1]
covarianceMat[1][2]=cov_23[0,1]
covarianceMat[1][3]=cov_24[0,1]

covarianceMat[2][0]=cov_13[0,1]
covarianceMat[2][1]=cov_23[0,1]
covarianceMat[2][2]=cov_33[0,1]
covarianceMat[2][3]=cov_34[0,1]

covarianceMat[3][0]=cov_14[0,1]
covarianceMat[3][1]=cov_24[0,1]
covarianceMat[3][2]=cov_34[0,1]
covarianceMat[3][3]=cov_44[0,1]


#PLOT the data points here
# /covarianceMatrix------------------------------------------------------------------------

correlationMat=[[[],[],[],[]],  [[],[],[],[]],  [[],[],[],[]],  [[],[],[],[]]]

correlation=np.corrcoef([cs_score, res_over, admin_base, tuition])

norm_cs=stats.norm.pdf(cs_score, loc=mu1, scale=sigma1)
norm_res=stats.norm.pdf(res_over, loc=mu2, scale=sigma2)
norm_admin=stats.norm.pdf(admin_base, loc=mu3, scale=sigma3)
norm_tuition=stats.norm.pdf(tuition, loc=mu4, scale=sigma4)

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

final=ll1+ll2+ll3+ll4
#final

matrix1=[]
matrix2=[]
matrix3=[]  

matrix1=createMatrix2(unitv, cs_score)
matrix2=createMatrix2(unitv, tuition)

log1_csTOres=BNGraph(res_over, matrix1)
log2_csTOtuit=BNGraph(tuition, matrix1)
log3_tuitTOadmin=BNGraph(admin_base, matrix2)

logLikelihood=log1_csTOres+log2_csTOtuit+log3_tuitTOadmin+ll1

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
