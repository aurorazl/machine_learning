from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter

def rbf_kernel_pca(X,gamma,n_components):
    sq_dists = pdist(X,"sqeuclidean")
    mat_sq_dists = squareform(sq_dists)
    K = exp(-gamma*mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N,N))/N
    K = K - one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)
    eigvals,eigvecs = eigh(K)
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    lambdas = [eigvals[-i] for i in range(1,n_components+1)]
    return alphas,lambdas

X,y = make_moons(n_samples=100,random_state=123)
alphas,lambdas= rbf_kernel_pca(X,gamma=15,n_components=1)
x_new = X[25]
x_proj = alphas[25]
def project_x(x_new,X,gamma,alphas,lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k=np.exp(-gamma*pair_dist)
    return k.dot(alphas/lambdas)
x_reproj = project_x(x_new,X,gamma=15,alphas=alphas,lambdas=lambdas)
plt.scatter(alphas[y==0,0],np.zeros((50)),color="red",marker='^',alpha=0.5)
plt.scatter(alphas[y==1,0],np.zeros((50)),color="blue",marker='o',alpha=0.5)
plt.scatter(x_proj,0,color="black",marker='^',s=100,label="original projection of point X[25]")
plt.scatter(x_reproj,0,color="green",marker='x',s=500,label="remapped point X[25]")
plt.legend(scatterpoints=1)
plt.show()