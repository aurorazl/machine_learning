from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import KernelPCA

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
scikit_kpca = KernelPCA(n_components=2,kernel='rbf',gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0,0],X_skernpca[y==0,1],color="red",marker='^',alpha=0.5)
plt.scatter(X_skernpca[y==1,0],X_skernpca[y==1,1],color="blue",marker='o',alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()