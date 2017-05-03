# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 09:03:14 2017

@author: bhave
"""
from scipy.sparse import csc_matrix
import numpy as np
import math
import heapq as hp
import sys
from collections import Counter

def centroid(sic,c):
    p=0
    for i in sic:
        p+=c[i]
    return p/len(sic) 

file=list(open(sys.argv[1],'r'))
clu=int(sys.argv[2])
doc=int(file[0])
words=int(file[1])
mat=np.zeros((doc,words))

a=[];b=[];c=[]
for i in xrange(3,len(file)):
    l=file[i].strip("\n").split(" ")
    a.append(int(l[0])-1);b.append(int(l[1])-1);c.append(int(l[2]))
mydict1=dict(Counter(b))
for i in range(len(b)):
    c[i]=c[i]*(math.log(float(doc+1)/(mydict1[b[i]]+1),2))
sp=csc_matrix((c,(a,b)),shape=(doc,words))
mt=np.asarray(np.sqrt(sp.power(2).sum(axis=1)))
mat=csc_matrix(sp/mt)
print(mat[24])

s={}
for i in range(mat.shape[0]):
    s[i]=mat[i]
hea=[]
for i in s:
    for j in s:
        if(i<j):
            mattemp=s[i].multiply(s[j])
            deno=(np.sqrt(s[i].power(2).sum()))*(np.sqrt(s[j].power(2).sum()))
            mattemp=(mattemp.sum())/deno
            hp.heappush(hea,(1-mattemp,i,j))
s1={}
iter=0
while iter<doc-clu:
    f=hp.heappop(hea)
    if (s1.has_key(f[1]) or s1.has_key(f[2])):
        continue
    elif(type(f[1])==int and type(f[2])==int):
        s1[f[1]]=None
        s1[f[2]]=None
        gh=(f[1],f[2])
        x=centroid(gh,mat)
        del s[f[1]]
        del s[f[2]]
    else:
        s1[f[1]]=None
        s1[f[2]]=None
        if(type(f[1])==int):
            gh=f[2]+(f[1],)
        elif(type(f[2])==int):
            gh=f[1]+(f[2],)
        else:
            gh=f[1]+f[2]
        del s[f[1]]
        del s[f[2]]
        x=centroid(gh,mat)
    for i in s:
        if not s1.has_key(i):
            mattemp=x.multiply(s[i])
            deno=(np.sqrt(s[i].power(2).sum()))*(np.sqrt(x.power(2).sum()))
            mattemp=(mattemp.sum())/deno
            hp.heappush(hea,(1-mattemp,gh,i))
    s[gh]=x
    iter+=1

for i in s:
    if type(i)==int:
        print i+1
    else:
        stri=''
        for j in sorted(i):
            stri+=str(j+1)+','
        print stri[:-1]