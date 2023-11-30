# Import các thư viện cần thiết
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# Thiết lập đường dẫn cho dữ liệu huấn luyện và kiểm tra
train_dir = 'C:/Users/Abc/Desktop/demo/train'
test_dir = 'C:/Users/Abc/Desktop/demo/test'

# Tạo ra các bộ dữ liệu cho quá trình huấn luyện và kiểm tra
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(224,224),batch_size=20,class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(224,224),batch_size=20,class_mode='binary')

# # Xây dựng mô hình nơ-ron
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# # Compile mô hình
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(learning_rate=1e-3), metrics=['accuracy'])

# # # Huấn luyện mô hình
history = model.fit_generator(train_generator,steps_per_epoch=train_generator.samples//20,epochs=10,validation_data=test_generator,validation_steps=test_generator.samples//20)

#Lưu
model.save('catvsdog.h5')
