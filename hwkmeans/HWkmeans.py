#%matplotlib inline
import numpy as np
import scipy.io
import scipy.cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

pathandfile = './TwoDimensionalContinuousData.csv'
target = open( pathandfile, 'r')
datalist = np.loadtxt(pathandfile,skiprows = 1,delimiter=',')

'''plt.figure(1)
plt.plot(datalist[:,0],datalist[:,1],'*',color='b')
plt.title('Title')'''

'''kmeans = KMeans(n_clusters=3, random_state=0).fit(datalist)

#give the clusters
kmeans.cluster_centers_
#it can predict to which cluster is asigned
#kmeans.predict([200, 200])

plt.figure(1)
plt.plot(datalist[:,0],datalist[:,1],'*',color='b')
centroids = kmeans.cluster_centers_
plt.plot(centroids[:,0],centroids[:,1],'o',color='r')
plt.title('Title')'''

"""
## k-mean with bysection method

Pseudo Code for k-mean bysection method:

Bisection Method:

-> start with two centroids

-> interate until we reach number of centroids wanted

-> refine k-means centroids using conventional k-mean algorith 
"""
def getcluster(data,labels,label):
  newdata = []
  i=0
  for point in data:
    if(labels[i]==label):
      newdata.append(data[i,:])
    i=i+1
  return np.asarray(newdata)

def SSEclusters(data,centroids,labels):
  i=0
  SSEs= np.zeros(2) 
  i=0
  for point in data:
    SSEs[labels[i]] = SSEs[labels[i]] + np.linalg.norm(data[i,:]-centroids[labels[i],:])
    i=i+1
 
  return SSEs

def bisectkmeans(data,k):
  """-> start with two centroids
  -> interate until we reach number of centroids wanted"""
 
  bisection_k = 2
  kmeans = KMeans(n_clusters=bisection_k, random_state=0).fit(data)
  centroids = kmeans.cluster_centers_
  labels = kmeans.labels_
  inertia = kmeans.inertia_
  centroids_out = np.zeros((k,2))
  
  SSEs = SSEclusters(data,centroids,labels) 
  print (type(bisection_k)) 
  while (bisection_k<k):
    #choose cluster with larger SSE
    if(SSEs[0]>SSEs[1]):
      data = getcluster(data,labels,0)
      centroids_out[bisection_k-2,:] = centroids[1,:] #safe centroid that is not being used
    else:
      data = getcluster(data,labels,0)
      centroids_out[bisection_k-2,:] = centroids[0,:] #safe centroid that is not being used
    
    #Kmeans algorith
    kmeans = KMeans(n_clusters=bisection_k, random_state=0).fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    #calculate SSE for next iteration
    SSEs = SSEclusters(data,centroids,labels) 
    bisection_k=bisection_k+1
    #safe last centroids centroid that is not being used
    if (bisection_k==k):
      centroids_out[bisection_k-2,:] = centroids[0,:]
      centroids_out[bisection_k-1,:] = centroids[1,:]

  return centroids_out#,labels,inertia,SSEs  

#start with the number of K means wanted
k=3

#bisection method
centroids= bisectkmeans(datalist,k)

print (centroids)

plt.show()
