# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm in ML, as shown in
examples/src/main/python/ml/kmeans_example.py.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function
from collections import Counter
import math
import sys
import scipy.sparse as sps
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.linalg import Matrices
sc=SparkContext(appName='INF553')
def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])


def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        mattemp=centers[i][1].multiply(p[1])
        deno=(np.sqrt(centers[i][1].power(2).sum()))*(np.sqrt(p[1].power(2).sum()))
        mattemp=(mattemp.sum())/deno
        if (1-mattemp) < closest:
            closest = 1-mattemp
            bestIndex = i
    return bestIndex


if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()
    fil=open(sys.argv[4],'w')
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    data = lines.map(parseVector).cache()
    d=data.collect()
    row=int(d[0])
    col=int(d[1])
    mat=np.zeros((row,col))
    a=[];b=[];c=[]
    for i in xrange(3,len(d)):
        a.append(int(d[i][0])-1);b.append(int(d[i][1])-1);c.append(int(d[i][2]))
    mydict1=dict(Counter(b))
    for i in range(len(b)):
        c[i]=c[i]*(math.log(float(row+1)/(mydict1[b[i]]+1),2))
    sp=sps.csc_matrix((c,(a,b)),shape=(row,col))
    mt=np.asarray(np.sqrt(sp.power(2).sum(axis=1)))
    sp=sps.csc_matrix(sp/mt)
    a=[];b=[];c=[]
    for i in xrange(3,len(d)):
	    a.append(int(d[i][0])-1);b.append(int(d[i][1])-1);c.append(int(d[i][2]))
    mydict1=dict(Counter(b))
    for i in range(len(b)):
	    c[i]=c[i]*(math.log(float(row+1)/(mydict1[b[i]]+1),2))
    sp=sps.csc_matrix((c,(a,b)),shape=(row,col))
    mt=np.asarray(np.sqrt(sp.power(2).sum(axis=1)))
    mat=sps.csc_matrix(sp/mt)
    si=[]
    for i in range(mat.shape[0]):
	    si.append((i,mat[i]))
    data=sc.parallelize(si)
    K = int(sys.argv[2])
    convergeDist = float(sys.argv[3])
    data=data.sortByKey()
    kPoints = data.repartition(1).takeSample(False, K, 1)
    tempDist = 1.0
    while tempDist > convergeDist:
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p[1], 1)))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(
            lambda st: (st[0],(st[1][1], st[1][0] / st[1][1]))).collect()
        tempDist = sum(np.sqrt((kPoints[ik][1]-p[1]).power(2).sum()) for (ik,p) in newPoints)
        for (ik,p) in newPoints:
            kPoints[ik]=p
    for k in kPoints:
        fil.write("%d\n"%k[1].getnnz())

    spark.stop()
