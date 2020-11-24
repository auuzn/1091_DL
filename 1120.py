import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import cifar10

(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()
class_names = ['airplain', 'automobile', ' bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def plot_multiimages(images, labels, prediction, idx, num=32):
    plt.gcf().set_size_inches(14, 4)
    if num > 32: num = 32
    for i in range(0, num):
        ax = plt.subplot(4, 8, 1 + i)
        ax.imshow(images[idx], cmap='rainbow')
        title = "l=" + class_names[int(labels[idx])]
        if len(prediction) > 0:
            title = "l={},p={}".format(class_names[int(labels[idx])], class_names[prediction[idx]])
        else:
            title = "l={}".format(class_names[int(labels[idx])])
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1

    plt.tight_layout()
    plt.show()


x_train = x_train_image.reshape(len(x_train_image), 32, 32, 3).astype('float32')
x_test = x_test_image.reshape(len(x_test_image), 32, 32, 3).astype('float32')
x_train_norm = x_train / 255
x_test_norm = x_test / 255
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling2D

model = Sequential()  # Build Linear Model

model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
# model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=x_train_norm[0:5000], y=y_TrainOneHot[0:5000], validation_split=0.2, epochs=5,
                          batch_size=20, verbose=2)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(x_test_norm, y_TestOneHot)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1] * 100.0))
print("\t[Info] Making prediction to x_test_norm")

prediction = model.predict_classes(x_test_norm[0:1000])  # Making prediction and save result to prediction
# classes = np.argmax(prediction, axis=-1)
print()
print("\t[Info] Show 10 prediction result (From 0):")
print("%s\n" % (prediction[0:10]))
plot_multiimages(x_test_image, y_test_label, prediction, idx=0)

print("\t[Info] Error analysis:")
for i in range(len(prediction)):
    if prediction[i] != y_test_label[i]:
        print("\tAt %d'th: %d is with wrong prediction as %d!" % (i, y_test_label[i], prediction[i]))
