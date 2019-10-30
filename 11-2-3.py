import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_blobs
from matplotlib import cm
from sklearn.metrics import silhouette_samples

np.random.seed(123)
variables = ["x","y","z"]
labels = ["ID_0","ID_1","ID_2","ID_3","ID_4"]
X = np.random.random_sample([5,3])*10
df = pd.DataFrame(X,columns=variables,index=labels)

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='complete')
labels = ac.fit_predict(X)
print(labels)
