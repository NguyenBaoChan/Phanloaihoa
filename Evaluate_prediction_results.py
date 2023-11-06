#Đánh giá kết quả dự đoán của mô hình
from sklearn.metrics import classification_report
import numpy as np
import cv2

from myclassfier import model, y_test, categories, x_test


predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
print(classification_report(y_test, y_pred, target_names=categories))

