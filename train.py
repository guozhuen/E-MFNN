from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten,concatenate
from keras.utils import to_categorical
import numpy as np
import keras.utils
from sklearn.model_selection import train_test_split




SEED=1
# 设置参数
num_classes = 2
input_shape1 = (10, 11, 11, 5)

batch_size = 1
epochs = 1

#构建脑电训练
# 构建卷积神经网络
cnn_model = Sequential()
cnn_model.add(Conv2D(64,(3,3), padding='same', activation='relu', input_shape=(11, 11, 5)))
cnn_model.add(Conv2D(64, (3,3), activation='relu', padding='same', name='conv1'))
cnn_model.add(Conv2D(128, (3,3), activation='relu', padding='same', name='conv2'))
cnn_model.add(Conv2D(128, (3,3), activation='relu', padding='same', name='conv2_1'))
cnn_model.add(Conv2D(256, (3,3), activation='relu', padding='same', name='conv3'))
cnn_model.add(Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_1'))
cnn_model.add(Conv2D(64, (3,3), activation='relu', padding='same', name='conv4'))
cnn_model.add(Conv2D(64, (3,3), activation='relu', padding='same', name='conv4_1'))
cnn_model.add(Flatten())


# 构建LSTM模型
model1 = Sequential()
model1.add(TimeDistributed(cnn_model, input_shape=input_shape1))
model1.add(LSTM(256, return_sequences=True))
model1.add(LSTM(256, return_sequences=False))
model1.add(Dense(256, activation='relu'))
x1=model1
z1 = Dense(256, activation="relu")(x1.output)
z1 = Dense(256, activation="relu")(z1)
model1 = Model(inputs=[x1.input], outputs=z1)

input_shape2 = (224,224,3)

#构建图片训练，使用的时vggnet的变形
cnn_model2 = Sequential()
cnn_model2.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
cnn_model2.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
cnn_model2.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model2.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn_model2.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn_model2.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model2.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
cnn_model2.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
cnn_model2.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
cnn_model2.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model2.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
cnn_model2.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
cnn_model2.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
cnn_model2.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model2.add(Dense(4096, activation='relu'))
cnn_model2.add(Dense(4096, activation='relu'))
cnn_model2.add(Dense(1000, activation='relu'))
cnn_model2.add(Flatten())
cnn_model2.add(Dense(256, activation='relu'))
cnn_model2.add(Dense(256, activation='relu'))


x2=cnn_model2
z2= Dense(256, activation="relu")(x2.output)

model2 = Model(inputs=[x2.input], outputs=z2)


#对眼动进行训练

cnn_model3 = Sequential()
cnn_model3.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(2, 52,1)))
cnn_model3.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

cnn_model3.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn_model3.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn_model3.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
cnn_model3.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
cnn_model3.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
cnn_model3.add(Conv2D(64, (3, 3), padding='same', activation='relu'))


cnn_model3.add(Flatten())

x3=cnn_model3
z3 = Dense(256, activation="relu")(x3.output)
model3 = Model(inputs=[x3.input], outputs=z3)

#合并
combined = concatenate([model1.output, model2.output, model3.output])

z = Dense(768, activation="relu")(combined)
z = Dense(1, activation="sigmoid")(z)

model = Model(inputs=[model1.input, model2.input, model3.input], outputs=z)
print('method:')
cnn_model.summary()
model.summary()
keras.utils.plot_model(model,show_shapes=True)

x1_train = np.load('eeg_reshaped.npy')
x2_train = np.load('images.npy')
x3_train = np.load('eye_data.npy')
x3_train=x3_train.reshape((5973, 2, 52, 1))
y_train = np.load('labels.npy')

x1_train, x1_val, x2_train, x2_val,x3_train, x3_val,y_train, y_val = train_test_split(x1_train, x2_train,x3_train,y_train, test_size=0.2, random_state=42)

# 将标签转化为分类格式
#y_train = to_categorical(y_train, num_classes=2)

model.compile(optimizer='sgd', loss='binary_crossentropy',loss_weights=[1., 0.2])
#训练模型

model.fit([x1_train, x2_train, x3_train], y_train,epochs=100, batch_size=32,validation_data=([x1_val, x2_val, x3_val], y_val))

