import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our convnet astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

TRAIN_DIR = 'C:/Users/mfrangos2016/Desktop/pdf-to-jpeg/pdf2jpg-master/Saved/FA'
TEST_DIR = 'C:/Users/mfrangos2016/Desktop/pdf-to-jpeg/pdf2jpg-master/Saved'
IMG_SIZE = 600
LR = 1e-4

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match



def create_train_data():
    index = 0
    training_data = []
    FA = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
          [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0],[0,1],[1,0],
          [1,0],[1,0],[1,0],[0,1],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
          [1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
          [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
          [1,0],[0,1],[0,1],[0,1],[0,1],[1,0],[0,1],[0,1],[0,1],[1,0],
          [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
          [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
          [1,0],[1,0]
            ]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = FA[index] #CREATES THE LABEL [1,0] format
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        y=int((360/800)*IMG_SIZE) #start distance from the top
        x=int((60/800)*IMG_SIZE) #start distance from the left
        h=int((80/800)*IMG_SIZE)  #length of height
        w=int((200/800)*IMG_SIZE) #length of width
        cropedImage = img[y:y+h, x:x+w]
        cropedImage = cv2.resize(cropedImage, (IMG_SIZE,IMG_SIZE))
        index = index + 1
        training_data.append([np.array(cropedImage),np.array(label)])
    shuffle(training_data)

    np.save('train_data.npy', training_data)
    cv2.imshow("Image",cropedImage)
    cv2.waitKey(0) 
    return training_data

######Show images
index=0
for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        y=int((360/800)*IMG_SIZE) #start distance from the top
        x=int((60/800)*IMG_SIZE) #start distance from the left
        h=int((80/800)*IMG_SIZE)  #length of height
        w=int((200/800)*IMG_SIZE) #length of width
        crop = img[y:y+h, x:x+w]
        index = index + 1
        cv2.imshow("Image",crop)
        cv2.waitKey(0) 
        
      

#Create the training data
train_data = create_train_data()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu', regularizer="L2")
convnet = max_pool_2d(convnet, 2)
convnet = local_response_normalization(convnet)
convnet = conv_2d(convnet, 64, 3, activation='relu', regularizer="L2")
convnet = max_pool_2d(convnet, 2)
convnet = local_response_normalization(convnet)
convnet = fully_connected(convnet, 128, activation='tanh')
convnet = dropout(convnet, 0.8)


convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')



train = train_data[:]

    

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

#test_x = np.array([i for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#test_y = [i[1] for i in test]


model.fit({'input': X}, {'targets': Y}, n_epoch=4, 
    snapshot_step=100, show_metric=True, run_id=MODEL_NAME,batch_size = 82)






def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        try:
            path = os.path.join(TEST_DIR,img)
            img_num = img.split('.')[0]
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            testing_data.append([np.array(img), img_num])
        except:
            pass
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

#test = []
#for x,y in test_data:
    #test.append(x)


# if you need to create the data:
test_data = process_test_data()

import matplotlib.pyplot as plt
# if you already have some saved:
#test_data = np.load('test_data.npy')
fig=plt.figure()
i = 0
setnum = 3
while i < 12:
    num = i
    data = test_data[i+12*setnum]
    # cat: [1,0]
    # dog: [0,1]
    print(num)
    
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0][0]
    print(model.predict([data])[0])
    
    if np.argmax(model_out) == 0:
        str_label='NoAid'
        print("noAid")
    else: 
        str_label='FinancialAid'
        print("aid")
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    i=i+1
plt.show()




