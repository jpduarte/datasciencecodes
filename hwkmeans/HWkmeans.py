#Mina Azhar
#%matplotlib inline
################################################################LIBRARIES###############################################################################

import numpy as np
import scipy.io
import scipy.cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
##########################################################################FUNCTIONS##############################################################
#function get the cluster of the label that is indicated, it sort the data
def getcluster(data,labels,label):
  newdata = []
  i=0
  for point in data:
    if(labels[i]==label):
      newdata.append(data[i,:])
    i=i+1
  return np.asarray(newdata)

#obtain the SSE for each cluster
def SSEclusters(data,centroids,labels):
  i=0
  SSEs= np.zeros(2) 
  i=0
  for point in data:
    SSEs[labels[i]] = SSEs[labels[i]] + np.linalg.norm(data[i,:]-centroids[labels[i],:])
    i=i+1
 
  return SSEs

#bisection kmean algorith
def bisectkmeans(data,k):
  """-> start with two centroids
  -> interate until we reach number of centroids wanted"""
 
  bisection_k = 2
  #kmeans
  kmeans = KMeans(n_clusters=bisection_k, random_state=0).fit(data)
  centroids = kmeans.cluster_centers_
  labels = kmeans.labels_

  #create centroids with same length as k
  centroids_out = np.zeros((k,2))
  
  SSEs = SSEclusters(data,centroids,labels) 

  while (bisection_k<k):
    #choose cluster with larger SSE
    if(SSEs[0]>SSEs[1]):
      data = getcluster(data,labels,0)
      centroids_out[bisection_k-2,:] = centroids[1,:] #safe centroid that is not being used in next iteration
    else:
      data = getcluster(data,labels,0)
      centroids_out[bisection_k-2,:] = centroids[0,:] #safe centroid that is not being used in next iteration
    
    #Kmeans algorith
    kmeans = KMeans(n_clusters=bisection_k, random_state=0).fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    #calculate SSE for next iteration
    SSEs = SSEclusters(data,centroids,labels) 
    bisection_k=bisection_k+1
    #safe last centroids centroid that is not being used, this happen only at the very end of the while loop
    if (bisection_k==k):
      centroids_out[bisection_k-2,:] = centroids[0,:]
      centroids_out[bisection_k-1,:] = centroids[1,:]

  return centroids_out#,labels,inertia,SSEs  

#####################################################################END FUNCTIONS###########################################################################

#####################################################################CODE RUN FROM HERE#########################################################################
#start with the number of K means wanted
k=3

#import data
pathandfile = './TwoDimensionalContinuousData.csv'
target = open( pathandfile, 'r')
datalist = np.loadtxt(pathandfile,skiprows = 1,delimiter=',')


#bisection method
centroids= bisectkmeans(datalist,k)

print("centroids:")
print (centroids)

plt.show()
