# Trực quan hóa liệu
import matplotlib.pyplot as plt
from myclassfier import y_train, categories, y_test

# Trực quan hóa dữ liệu
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Training Data Distribution')
plt.hist(y_train, bins=len(categories))
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(range(len(categories)), categories, rotation=45)

plt.subplot(1, 2, 2)
plt.title('Test Data Distribution')
plt.hist(y_test, bins=len(categories))
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(range(len(categories)), categories, rotation=45)

plt.tight_layout()
plt.show()


#Đoạn code trên sẽ vẽ hai biểu đồ histogram, một cho phân phối dữ liệu trong tập huấn luyện và một cho phân phối dữ liệu trong tập kiểm tra.
# Mỗi thanh trong biểu đồ đại diện cho một loại hoa, và chiều cao của thanh biểu thị số lượng mẫu trong tập dữ liệu thuộc loại hoa đó.
# Các trục x và y được gắn nhãn để hiển thị loại hoa và số lượng mẫu tương ứng.

#Hàm plt.tight_layout() được sử dụng để tạo khoảng trống tự động giữa các subplot trong biểu đồ.
# Khi bạn gọi plt.tight_layout() sau khi tạo subplot, nó sẽ tính toán và áp dụng tự động các khoảng trống sao cho các subplot không chồng lên nhau.
# Điều này giúp cải thiện tính rõ ràng và hiểu quả của biểu đồ.

#Vì vậy, biểu đồ trong đoạn mã trên có các cột dạng cột dính liền nhau và có khoảng trống giữa các cột để tăng tính trực quan và đọc được của biểu đồ