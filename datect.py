from scosilt import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

(feature, labels) = load_data()
x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1)
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = tf.keras.models.load_model('mymodel.h5')
model.evaluate(x_test, y_test, verbose=1)

predictions = model.predict(x_test)
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i+1)
    image = (x_test[i] * 255).astype(np.uint8)  # Chuyển đổi kiểu dữ liệu thành uint8
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi BGR sang RGB
    plt.imshow(image_rgb)
    plt.xlabel('Actual: ' + categories[y_test[i]] + '\n' + 'Predicted: ' + categories[np.argmax(predictions[i])])
    plt.xticks([])
plt.show()
