# Helper functions crucial to manifold_capacity

import autograd.numpy as np  
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

def localLinCenters(XtotT,nb=25,nc=2,method='nn'):
    '''
    Find local linear centers of each manifold 
    Args:
        XtotT: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
                of the space, and P_i is the number of sampled points for the i_th manifold.
        nb: Number of neighbors to consider for each point (see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding)
        nc: Number of components for each manifold (see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding) 

    Returns:
        centers: Array of local linear centers, shaped (N,1)     
    '''

    centers = []
    for i in range(len(XtotT)):
        X_input = XtotT[i].T
        if (X_input.shape[0] < nb):
            nb = X_input.shape[0]//2
        X_lle = embed(X_input,nb,nc)
        # Get solution for X_lle*P = X_input
        c_lle = np.mean(X_lle,axis=0)
        #print("Center in local space:", c_lle.shape)
        if method == 'nn':
            neigh = getNearestNeighbors(X_lle, c_lle,n_neighbors=5)       
            c = np.mean(X_input[neigh,:],axis=0)   
        elif method == 'weighted':
            wnorm = np.diag(getWeights(X_lle, c_lle))
            c = np.matmul(wnorm,X_input).sum(axis=0)
        centers.append(c)
    return centers

# return embedding vectors given X
# X is #samples x #features
# Returns dimension #sampelsxn_components
def embed(X,n_neighbors,n_components):
    scaler = StandardScaler()
    scaler.fit_transform(X)
    embed = LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=n_components,eigen_solver='dense')
    X_transformed = embed.fit_transform(X)
    return X_transformed

# X is (#samples x #features)
def getNearestNeighbors(X,c,n_neighbors=5):
    Xnorm = X-c
    #print(Xnorm.shape) # (#samples x #feat)
    dist = np.linalg.norm(Xnorm,axis=1)
    #print(dist.shape) # number of samples
    ind = sorted(range(len(dist)), key=lambda i: dist[i])[0:n_neighbors]
    return ind 

def getWeights(X,c):
    Xnorm = X-c
    #print(Xnorm.shape) # (#samples x #feat)
    dist = np.linalg.norm(Xnorm,axis=1)
    weight = 1/(dist**2) 
    weight_normed = weight/np.sum(weight)
    return weight_normed 

def diffBetweenManifolds(XtotT,samp_n):
    n = len(XtotT)
    if samp_n == None:
        samp_n = len(XtotT)
    while True:
        samp_index1 = np.random.choice(np.arange(n),samp_n,replace=True)
        samp_index2 = np.random.choice(np.arange(n),samp_n,replace=True)
        if ((samp_index1-samp_index2).any() != 0):
            break

    g1 = samp_index1.tolist()
    g2 = samp_index2.tolist() 

    dcenters = []
    for i in g1:
        diff = XtotT[g1[i]] - XtotT[g2[i]]
        dcenters.append(np.mean(diff,axis=1))

    return dcenters
    
def diffBetweenCentr(centers,samp_n):
    n = len(centers)
    if samp_n == None:
        samp_n = len(centers)
    while True:
        samp_index1 = np.random.choice(np.arange(n),samp_n,replace=True)
        samp_index2 = np.random.choice(np.arange(n),samp_n,replace=True)
        if ((samp_index1-samp_index2).any() != 0):
        #test = samp_index1-samp_index2
        #if (len(np.unique(test))==len(test)):
            break

    g1 = samp_index1.tolist()
    g2 = samp_index2.tolist() 

    d = []
    for i in g1:
        diff = centers[g1[i]] - centers[g2[i]]
        d.append(diff)
    return d 

def sampleDiff(X,samp_n=None,method='all'):
    if method == 'all':
        d = diffBetweenManifolds(X, samp_n)
    elif method == 'center':
        d = diffBetweenCentr(X,samp_n)
    return d 