"""
Computation of R-square V_inter/V for clustering
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# compute r square for clustering
def r_square(data,centroids,labels,q):
    """rsquare

    R-square is computed from between variance and within variance of clustered data. This score (between 0 and 1) can be used to evaluate clustering

    Args:
        data(numpy.ndarray): m*n matrix of the original data to cluster
        centroids(np.ndarray): q*n matrix of cluster centroids
        labels(nb.ndarray): m*1 array of cluster labels for each instance of data
        q(int): number of clusters
    Returns:
        float: R-square score
   
    """
    v_within = within_variance(data,centroids,labels,q)
    v_between = between_variance(data,centroids,labels,q)
    return v_between/(v_between+v_within)

# compute within variance
def within_variance(data,centroids,labels,q):
    res = 0.0
    for k in range(q):
        # get number of instances inside cluster k
        n_k = (labels==k).sum()

        # select rows of data associated with each cluster
        d_k = data[np.where(labels==k)]
        
        # sum squared distances between each point and its centroid
        sum = 0.0
        for vec_k in d_k:
            sum += np.sum(np.square(vec_k-centroids[k]))

        res += sum
    return res/len(data)


# compute between variance
def between_variance(data,centroids,labels,q):
    center = np.average(data,axis=0)

    res = 0.0
    
    for k in range(q):
        # get number of instances inside cluster k
        n_k = (labels==k).sum()

        # sum squared distances between global centroid and each cluster centroid
        res += n_k * np.sum(np.square(centroids[k]-center))

    return res/len(data)



# compute r square for clustering
def r_square(data,centroids,labels,q):
    v_within = within_variance(data,centroids,labels,q)
    v_between = between_variance(data,centroids,labels,q)
    return v_between/(v_between+v_within)