# %%  Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

# %%   Load dataset
from sklearn.datasets import load_digits
dataset = load_digits()
dataset.keys()
# %%   divide the dataset into input and target

input = dataset.data
target = dataset.target

# %%   Visualization

ar = np.linspace((random.randrange(10)),(random.randrange(200)),6, dtype=np.int)
ls =  list(ar)

for i in ls:
    img = input[i,:]
    img = img.reshape(8,8)
    plt.figure(figsize=(3,2), num=0)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.title(dataset.target[i])
    plt.show()

# another way

images = dataset.images
for i in range(6):
    plt.figure(figsize=(6,4), num=0)
    plt.subplot(2,3, i+1)
    plt.axis('off')
    plt.imshow(images[i], cmap='gray')
    plt.title(dataset.target[i])

# %%   Split the dataset into training and testing set

from sklearn.model_selection import train_test_split
train_f, test_f, train_t, test_t = train_test_split(input, target, test_size=0.2, random_state=21)

# %%    model fitting

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(train_f, train_t)

# %%   Score

model.score(test_f, test_t)
# %%    Prediction

num = 10
test_ar = test_f[num]
mark = model.predict([test_ar])

image = test_ar.reshape(8,8)
plt.imshow(image, cmap='gray')
plt.title('Digit is {}'.format(mark))
plt.show()

# %%  Save the model

from sklearn.externals import joblib

joblib.dump(model, 'G:\Machine Learning\Classification\Digit Recognition/Digit_Recnogition_App')

