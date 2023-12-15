'''
Computes the simulation capacity of the general manifold under sparse or dense labels.
Written by Nga Yu Lo 
Based on original code by SueYeon Chung
'''
import random 
import numpy as np
np.random.seed(0)
import warnings
import copy 
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from cvxopt import solvers, matrix
# from my_utils.utils import reduce_feature_space


# X is list of #manifolds shaped (#features,#instances) 
def normalize(X):
    Xori = np.concatenate(X, axis=1) # Shape (N, sum_i P_i)
    globalmean = np.mean(Xori,axis=1,keepdims=True)
    globalstd = np.std(Xori,axis=1,keepdims=True)
    Xnorm = [(Xi - globalmean)/globalstd for Xi in X]
    return Xnorm 

# X is list of #manifolds shaped (#features,#instances) 
def reduce_feature_space(X,n,have_seed=False, NORMALIZE=False):
    N = X[0].shape[0]
    
    if have_seed:
        torch.manual_seed(n)
    M = torch.randn(N,n).to(device)
    M /= torch.Tensor.sqrt(torch.Tensor.sum(torch.Tensor.square(M), axis=0, keepdims=True))
    X = [torch.matmul(torch.transpose(M,0,1), torch.Tensor(np.array(d)).to(device)) for d in X]
    X = [d.cpu().numpy() for d in X]

    if NORMALIZE:
        X = normalize(X)
    return X

def sample_y(p,seed,split):
    '''
    Get random dichotomy for manifolds.

    Args:
        p: Number of manifolds in the system.
        seed: Matrix of shape (1, M) containing the label for each of the M data points. Labels must be +1 or -1
        split: Sparse parameter (fraction of positive labels); 1/P if sparse labeling and 1/2 if dense labeling)

    Returns:
        y: Array of +1/-1 labels 
    '''
    if split == 1/p:
        return sample_y_opt(p, seed)

    else:
        random.seed(seed)
        y = []
        assert(int(p))
        pos = int(split*p)
        y.extend([1]*pos)
        y.extend([-1]*(p-pos))
        random.shuffle(y)
        return y

def sample_y_opt(p,seed):
    y = []
    y.extend([-1]*(p))
    y[seed] = 1
    return y

def find_svm_sep_primal_wb(X, y, w_ini, kappa=0, tolerance=1e-8, flag_wb=1):
    '''
    Written by: Cory Stephenson 

    Finds the optimal separating hyperplane for data X given the dichotomy specified by y.
    The plane is defined by the vector w and is found by minimizing
        1/2 * w.T * w
    Subject to the constraint
        y * (x.T * w + b) >= 1
    For all data points, and an optional bias b.

    Args:
        X: Data matrix of shape (N, M) where N is the number of features, and M is the number of data points.
        y: Matrix of shape (1, M) containing the label for each of the M data points. Labels must be +1 or -1
        flag_wb: Option to include a bias.  Uses a bias if set to 1.

    Returns:
        sep: Whether or not the dichotomy is linearly separable
        w: Weights of the optimal hyperplane
        margin: Size of margin
        flag: Not used. 
        u: Unormalized weights of the optimal hyperplane
        bias: Bias for the separating plane
    '''
    # Configure the solver
    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 1000000
    solvers.options['feastol'] = tolerance
    solvers.options['abstol'] = tolerance
    solvers.options['reltol'] = tolerance

    # Get the shape of X
    M, N = X.shape[1], X.shape[0]
    
    # Verify there are the right number of labels and that they are +/- 1
    assert M == y.shape[1]
    assert all([np.abs(l[0]) == 1 for l in y])

    # Optionally add a constant component to X, otherwise plane is constrained to pass through the origin
    if flag_wb == 1:
        offset = np.ones((1, M))
    else:
        offset = np.zeros((1, M))
    Xb = np.concatenate([X, offset], axis=0)

    # Construct the input to the solver
    # Want to minimize 1/2 * w.T * P * w subject to the constrant that -y * X.T * w <= -1
    # P ignores the component of w that corresponds to offset, the constraint does not.

    # P should be identity with the final component set to zero
    P = np.identity(N + 1)
    P[-1, -1] = 0
    P = matrix(P)

    # q should be zero, (no term like q.T * w)
    q = np.zeros(N + 1)
    q = matrix(q)

    # Specify the constraint.  Ab is -y * X.T, bb is a vector of -1s
    Ab = - y * Xb # (N, M)
    Ab = matrix(Ab.T) # (M, N)
    bb = - np.ones(M)
    bb = matrix(bb)

    # Solve using cvxopt
    output = solvers.qp(P, q, Ab, bb)
    ub = np.array(output['x'])

    # Separate the bias
    u = ub[0:-1, 0]
    b = ub[-1, 0]
    # Normalize the outputs
    u_norm = np.linalg.norm(u)
    b_norm = b/ u_norm
    w = u/u_norm

    # Compute the margin
    Pr = (np.matmul(w.T, X) + b_norm)/np.linalg.norm(w.T)
    margin = np.min(y * Pr )
    # Check seperability
    seperable = np.all(np.sign(Pr) == y)
    return seperable, w, margin, 1, u, b_norm

def is_linear_separable(p,X,seed,split):
    '''
    Determines if the system of manifold is separable on a given dichotomy.

    Args: 
        p: Number of manifolds 
        X: Data matrix of shape (N, M) where N is the number of features, and M is the number of data points.
        seed: Random seed to ensure different dichotomies
        split: Sparse parameter (fraction of positive labels); 1/P if sparse labeling and 1/2 if dense labeling)

    Returns:
        sep: Whether or not the dichotomy is linearly separable
        margin: Size of margin
        bias: Bias for the separating plane
    '''
    manifold_labels = sample_y(p,seed,split)
    y = []
    num_examples = int(X.shape[0]/p)
    for j in range(len(manifold_labels)):
        for i in range(num_examples):
            y.append(manifold_labels[j])
    del(manifold_labels)
    y = np.array(y)
    X = X.T 
    y = y.reshape(1, -1)
    N = X.shape[0]
    try: 
        sep, w, margin, flag, u, bias  = find_svm_sep_primal_wb(X,y,w_ini=np.zeros((N, 1)),tolerance=1e-2,flag_wb=1)
        bias = bias*np.linalg.norm(u)
    except ValueError as e:
        warnings.warn('Could not find solution')
        return 0, None, None
    
    return sep, bias, margin

def measure_capacity(n,p,activations,rep,split,normalize=False):
    '''
    Get probability of separating a dichotomy. 

    Args:
        n: Number of features to reduce to
        p: Number of manifolds 
        activations: Sequence of arrays, shaped (N,M); N is the original number of features and M is the number of examples in the manifold 
        rep: Number of dichotomies to explore. For sparse labeling, rep=p
        split: Sparse parameter (fraction of positive labels); 1/P if sparse labeling and 1/2 if dense labeling)
        normalize: option to normalize after random projection. Default is False. 

    Returns:
        prob: Probability of separating a dichotomy
        marg: Array of margins from SVM solution of given dichotomies. If not separable, the value is Null.  
    '''
    # Reduce feature dimension
    activations_cp = copy.deepcopy(activations)
    reduced_activations = reduce_feature_space(activations_cp,n,NORMALIZE=normalize)
    del(activations_cp)
    X_sub = np.array(reduced_activations)
    del(reduced_activations)
    # X_sub is in the form of (#manifolds,#features,#examples per class)
    
    # Center Data
    # Compute the global mean over all samples
    Xori = np.concatenate(X_sub, axis=1) # Shape (N, sum_i P_i)
    X_origin = np.mean(Xori, axis=1, keepdims=True)
    # Subtract the mean from each manifold
    X0 = [X_sub[i] - X_origin for i in range(p)]

    X = np.concatenate(X0,axis=1).T
    # X is in the form of (#instances,#features)
    
    separated = 0
    seed = 0
    marg = []
    for i in range(rep):
        sep, bias, margin = is_linear_separable(p,X,seed,split)
        # sep, bias, margin = is_linear_separable(n,p,X,seed,split)
        #print(sep,bias)
        separated += sep
        marg.append(margin)
        seed += 1
    prob = separated/rep
    return prob, marg


# activations is a sequence of arrays, shaped (#features,#instances) 
def bisection_search(min_n,max_n,rep,p,activations,split,found=False,error=10**(-4),max_iter=50,normalize=False):
    '''
    Perform bisection search for N_c, the critical number of features. Search stops when the probability of separating a dichotomy is near within tolerance from 0.5. 

    Args:
        min_n: Minimum possible value for N_c
        max_n: Maximum possible value for N_c
        p: Number of manifolds 
        activations: Sequence of arrays, shaped (N,M); N is the original number of features and M is the number of examples in the manifold 
        rep: Number of dichotomies to explore. For sparse labeling, rep=p
        split: Sparse parameter (fraction of positive labels); 1/P if sparse labeling and 1/2 if dense labeling)
        found: Initialize as False
        error: Tolerance parameter to reach desired probability. Default is 10**-4
        max_iter: Number of iterations to perform search. Default is 50.
        normalize: Option to normalize after random projection. Default is False. 

    Returns:
        n: The estimated critical number of features (N_c)
        n_arr (ret_n_arr): Array of all number of features the input was reduced to (i.e. all tested N_c's)
        prob_arr (ret_prob_arr): Array of probability of separation corresponding to the tested N_c's
        margin_arr: Dictionary of array of margin values of separation for tested dichotomies. Each array corresponds to when the input was reduced to the tested N_c.     
        
    '''

    iter = 0
    prob_max, margin_max = measure_capacity(max_n,p,activations,rep,split,normalize=normalize) 
    prob_min, margin_min = measure_capacity(min_n,p,activations,rep,split,normalize=normalize)
    n_arr = [min_n,max_n]
    prob_arr = [prob_min,prob_max]
    n = 0

    # bias and margin arrays are dictionaries where each key is N, 
    # and each item is an array of the biases and margins of all svms constructed
    margin_arr = {}
    margin_arr[f"{max_n}"] = margin_max
    margin_arr[f"{min_n}"] = margin_min

    while found==False and iter<max_iter:
        n_next = int((max_n+min_n)/2)

        if (n_next in n_arr):
            break
        else:
            n = n_next 
            prob, margin = measure_capacity(n,p,activations,rep,split,normalize=normalize) 
            n_arr.append(n)
            prob_arr.append(prob)
            margin_arr[f"{n}"] = margin
            if np.abs(prob - 0.5) < error:
                found = True
                break
            else:
                if (prob_min-0.5)*(prob-0.5) < 0:
                    max_f = n
                    prob_max = prob
                else:
                    min_f = n 
                    prob_min = prob
            iter += 1
    
    if found==True:
        prob, margin = measure_capacity(int(n),p,activations,rep,split) 
        margin_arr[f"{int(n)}"] = margin              
        return n, n_arr, prob_arr, margin_arr
    else:
        try:
            zip_list = sorted(zip(prob_arr,n_arr))
            ret_prob_arr, ret_n_arr = list(zip(*zip_list))
            n = np.interp(0.5,ret_prob_arr,ret_n_arr)
            prob, margin = measure_capacity(int(n),p,activations,rep,split) 
            margin_arr[f"{int(n)}"] = margin
            return n,ret_n_arr,ret_prob_arr, margin_arr

        except ValueError as e:
            warnings.warn('Could not find solution')
            return -1,ret_n_arr,ret_prob_arr, margin_arr

    