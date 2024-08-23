import cpu_apf
import numpy as np

# data = np.random.normal(0., 1., size=(1000, 1000)) > 1
data = np.zeros((1000, 1000))
for i in range(300):
    x = np.random.randint(low=0, high=999)
    y = np.random.randint(low=0, high=999)
    data[y, x] = 1
# data[2, 2] = 1
# data[4, 4] = 1
print(data.min())
print(data.max())
print(data.sum())
data_apf, is_empty = cpu_apf.cpu_apf_bool(data)
print(is_empty)
print(data_apf.min())
print(data_apf.max())
print(np.isnan(data_apf).any())
