import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

csv_date = """A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,"""
df = pd.read_csv(StringIO(csv_date))
imr = Imputer(missing_values="NaN",strategy="mean",axis=0)
imr = imr.fit(df)
imputed_date = imr.transform(df.values)
print(imputed_date)