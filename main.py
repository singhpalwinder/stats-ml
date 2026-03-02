from sklearn.preprocessing import MinMaxScaler, StandardScaler





m = [
    [0, 1, 2],
    [2,0,1],
    [1,2,0],
    [-1,-1,-1]
]

scaler = StandardScaler()
res = scaler.fit(m)
print(res)