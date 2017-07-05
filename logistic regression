# -*- coding: utf-8 
# logistic regression
#解析txt文件

from numpy import
import operator
def filematrix():
     fr=open('testSet.txt')
     dataMat=[]
     dataLab=[]
     for line in fr.readlines():
         line=line.strip()
         formLine=line.split('\t')
         dataMat.append([1.0,float(formLine[0]),float(formLine[1])])
         dataLab.append(int(formLine[2]))
     return dataMat, dataLab
         
#定义sigmoid函数
def sigmoid(inX):
    from math import exp
    return 1.0/(1+exp(-inX))

#Gredient Method
def gradAscent(dataMat,dataLab):
    dataMat=mat(dataMat)   #list数据类型转化为matrix数据类型 
    dataLab=mat(dataLab).transpose()
    m,n=shape(dataMat)
    alpha=0.001   #leaning  rate
    w=ones((n,1))
    numCycle=500
    for i in range(numCycle):
        h=sigmoid(dataMat*w)
        error=dataLab-h
        w=w+alpha*dataMat.transpose()*error   #update 
    return w
        
def plotBestFit(weight):
    import matplotlib.pyplot as plt
    dataMat,dataLab=filematrix()
    dataArr=array(dataMat)   #transform  to array
    xcord1=[];ycord1=[]   # scatterplot
    xcord2=[];ycord2=[]
    n=shape(dataArr)[0]
    for i in range(n):
        if int(dataLab[i])==1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')   #图形的属性 :red&square
    ax.scatter(xcord2,ycord2,s=30,c='blue')             #default :cycle
    x = arange(-3.0, 3.0, 0.1)                          #draw  the decison boundary
    y = (-float(weight[0])-float(weight[1])*x)/float(weight[2])
    ax.plot(x, y)
plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()     
