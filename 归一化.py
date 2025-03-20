

from sklearn.preprocessing import MinMaxScaler
#最大最小
scaler = MinMaxScaler(feature_range=(a, b))
data_normalized = scaler.fit_transform(data)

from sklearn.preprocessing import StandardScaler
#z-scores
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)