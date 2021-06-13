from tensorflow.python.keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
import sys
import numpy as np
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

net = load_model('./model/model-resnet50.h5')

cls_list = ['cats', 'dogs']

file_name = input('input the file name: ')

file_path = './'+file_name
print('path',file_path)

image1 = cv.imread(file_path)

img = image.load_img(file_path, target_size=(224, 224))

h_dim = (224, 112)
h_resized = cv.resize(image1, h_dim, interpolation = cv.INTER_AREA)
cv.imwrite('./result_image/h_resized.jpg', h_resized)

w_dim = (112, 224)
w_resized = cv.resize(image1, w_dim, interpolation = cv.INTER_AREA)
cv.imwrite('./result_image/w_resized.jpg', w_resized)

b_dim = (112, 112)
b_resized = cv.resize(image1, b_dim, interpolation = cv.INTER_AREA)
cv.imwrite('./result_image/b_resized.jpg', b_resized)

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
pred = net.predict(x)[0]
top_inds = pred.argsort()[::-1][:5]

max_val = -1
label = ''
for i in top_inds:
    print('{:.3f}  {}'.format(pred[i], cls_list[i]))
    if(max_val<pred[i]):
        max_val = pred[i]
        label = cls_list[i]

plt.imshow(image1)
plt.title('Class: '+label)
plt.savefig('./result_image/result')
plt.show()


## resized image
### h
h_img = image.load_img('./result_image/h_resized.jpg', target_size=(224, 224))
x = image.img_to_array(h_img)
x = np.expand_dims(x, axis = 0)
pred = net.predict(x)[0]
top_inds = pred.argsort()[::-1][:5]
h_max_val = -1
h_label = ''
for i in top_inds:
    #print('h_resized    {:.3f}  {}'.format(pred[i], cls_list[i]))
    if(h_max_val<pred[i]):
        h_max_val = pred[i]
        h_label = cls_list[i]

'''plt.imshow(h_resized)
plt.title('Class: '+h_label)
plt.show()'''

### w
w_img = image.load_img('./result_image/w_resized.jpg', target_size=(224, 224))
x = image.img_to_array(w_img)
x = np.expand_dims(x, axis = 0)
pred = net.predict(x)[0]
top_inds = pred.argsort()[::-1][:5]
w_max_val = -1
w_label = ''
for i in top_inds:
    #print('w_resized    {:.3f}  {}'.format(pred[i], cls_list[i]))
    if(w_max_val<pred[i]):
        w_max_val = pred[i]
        w_label = cls_list[i]

'''plt.imshow(w_resized)
plt.title('Class: '+w_label)
plt.show()'''

### b
b_img = image.load_img('./result_image/b_resized.jpg', target_size=(224, 224))
x = image.img_to_array(b_img)
x = np.expand_dims(x, axis = 0)
pred = net.predict(x)[0]
top_inds = pred.argsort()[::-1][:5]
b_max_val = -1
b_label = ''
for i in top_inds:
    #print('b_resized    {:.3f}  {}'.format(pred[i], cls_list[i]))
    if(b_max_val<pred[i]):
        b_max_val = pred[i]
        b_label = cls_list[i]

'''plt.imshow(a_resized)
plt.title('Class: '+a_label)
plt.show()'''

'''print('ori: '+str(max_val))
print('h_re: '+str(h_max_val))
print('w_re: '+str(w_max_val))
print('a_re: '+str(a_max_val))'''

'''height = np.array([max_val,h_max_val,w_max_val,b_max_val])
left = np.array([1, 2, 3, 4])
labels = ['origin', 'h_resize', 'w_resize', 'b_resize']
plt.bar(left, height,  tick_label=labels)
plt.savefig('./resized_img/bar_chart')
plt.show()'''
