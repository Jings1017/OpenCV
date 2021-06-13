from tensorflow.python.keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
import sys
import numpy as np
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''net = load_model('./model/model-resnet50.h5')

cls_list = ['cats', 'dogs']

image1 = cv.imread('./train/cat.9952.jpg')

img = image.load_img('./train/cat.9952.jpg', target_size=(224, 224))

h_dim = (224, 112)
h_resized = cv.resize(image1, h_dim, interpolation = cv.INTER_AREA)
cv.imwrite('./resized_img/h_resized.jpg', h_resized)

w_dim = (112, 224)
w_resized = cv.resize(image1, w_dim, interpolation = cv.INTER_AREA)
cv.imwrite('./resized_img/w_resized.jpg', w_resized)

b_dim = (112, 112)
b_resized = cv.resize(image1, b_dim, interpolation = cv.INTER_AREA)
cv.imwrite('./resized_img/b_resized.jpg', b_resized)

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
pred = net.predict(x)[0]
top_inds = pred.argsort()[::-1][:5]

max_val = -1
label = ''
for i in top_inds:
    print('  resized    {:.3f}  {}'.format(pred[i], cls_list[i]))
    if(max_val<pred[i]):
        max_val = pred[i]
        label = cls_list[i]

plt.imshow(image1)
plt.title('Class: '+label)
plt.savefig('./resized_img/result')
plt.show()
'''
for i in range(0,1000):
    image1 = cv.imread('./train/cat.'+str(i)+'.jpg')
    image2 = cv.imread('./train/dog.'+str(i)+'.jpg')
    h_dim = (224, 112)
    h_resized1 = cv.resize(image1, h_dim, interpolation = cv.INTER_AREA)
    h_resized2 = cv.resize(image2, h_dim, interpolation = cv.INTER_AREA)
    cv.imwrite('./sample_r/train/cats.'+str(i)+'.jpg', h_resized1)
    cv.imwrite('./sample_r/train/dogs.'+str(i)+'.jpg', h_resized2)

for i in range(0,400):
    image1 = cv.imread('./train/cat.'+str(i)+'.jpg')
    image2 = cv.imread('./train/dog.'+str(i)+'.jpg')
    h_dim = (224, 112)
    h_resized1 = cv.resize(image1, h_dim, interpolation = cv.INTER_AREA)
    h_resized2 = cv.resize(image2, h_dim, interpolation = cv.INTER_AREA)
    cv.imwrite('./sample_r/valid/cats.'+str(i)+'.jpg', h_resized1)
    cv.imwrite('./sample_r/valid/dogs.'+str(i)+'.jpg', h_resized2)

'''height = np.array([92.05,93.70])
left = np.array([1, 2])
labels = ['before_resized', 'after_resized']
plt.bar(left, height,  tick_label=labels)
plt.savefig('./resized_img/compare')
plt.show()'''