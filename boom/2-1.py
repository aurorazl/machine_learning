import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)
# print(df.info())
print(df.describe())
import matplotlib.pyplot as plt
df.hist()
plt.show()

