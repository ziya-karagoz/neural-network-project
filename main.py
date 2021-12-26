import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split

#Data importları
hap = pd.read_csv("./data.csv")
hap.reset_index(drop=True, inplace=True)

x = hap.iloc[:, 2:]
y = hap.iloc[:, 1]
y = np.array([1 if i >= 5 else 0 for i in y]).reshape(149, 1)
      
#  Her Satır bir eğitim örneği, her sütun bir özellik  [X1, X2, X3]
giris_seti = np.array((x.values), dtype=float) # (149, 8)
labels = y # (149, 1)  

np.random.seed(42)
agirliklar = np.random.rand(8,1)
bias = np.random.rand(1)
lr = 0.05 #eğitim hızı (learning rate)

# sigmoid metodları
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_turevi(x):
    return sigmoid(x)*(1-sigmoid(x))


# eğitim döngüsü
for epoch in range(700):
    girisler = giris_seti
    XW = np.dot(girisler, agirliklar)+ bias
    z = sigmoid(XW)
    error = z - labels
    print(error.sum())
    dcost = error
    dpred = sigmoid_turevi(z)
    z_del = dcost * dpred
    girisler = giris_seti.T
    agirliklar = agirliklar - lr*np.dot(girisler, z_del)
    for num in z_del:
        bias = bias - lr*num

bir_data = np.array(x.values[23])
sonuc = sigmoid(np.dot(bir_data, agirliklar) + bias)

# sonucu ekrana yazdırma
print(sonuc)
