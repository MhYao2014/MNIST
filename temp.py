import numpy as np


a = np.array([
    [3,4],
    [1,-1],
    [7,3]
])
print()
musk = np.zeros(a[0].shape)

print(a*musk)

one_sample = lambda row: np.diag(row) - np.matmul(row.reshape(-1,1),row.reshape(1,-1))
Print = lambda row: print (row)

b = np.array([
    [[1,1],
     [0,1]],
    [[3,4],
     [5,1]],
    [[1,2],
     [2,1]]
])
c = np.array([[3,7],
              [5,9],
              [11,8]]).reshape(3,1,2)
print(a)
print(np.sum(a,axis=0))
print(np.apply_along_axis(one_sample,axis=1,arr=a))
print(b)
print(c)
print(c*b)
print(np.sum(c*b,axis=2))
print(np.tensordot(c,b,axes=([2],[2]))[range(c.shape[0]),0,range(c.shape[0]),:])
print(np.tensordot(c,b,axes=([2],[2])).shape)


