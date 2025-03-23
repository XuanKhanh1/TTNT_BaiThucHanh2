import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

#Cau 1:
iris = load_iris()
# Gọi thuộc tính data của biến iris
print(" Dữ liệu Iris:")
print(iris.data)

#Cau 2:
iris = load_iris()

# Hiển thị mối quan hệ giữa số và loại hoa
print("Mối quan hệ giữa số và loại hoa:")
for i, species in enumerate(iris.target_names):
    print(f"{i}: {species}")

#Cau 3:
iris = load_iris()
# Lấy dữ liệu chiều dài và chiều rộng của sepal
sepal_length = iris.data[:, 0]  # Chiều dài sepal (cột 0)
sepal_width = iris.data[:, 1]   # Chiều rộng sepal (cột 1)
# Lấy nhãn loại hoa (0, 1, 2)
labels = iris.target
# Tạo biểu đồ phân tán với màu sắc khác nhau cho mỗi loại hoa
for i, species in enumerate(iris.target_names):
    plt.scatter(sepal_length[labels == i], sepal_width[labels == i], label=species)
# Thêm tiêu đề và nhãn trục
plt.xlabel('Chiều dài sepal (cm)')
plt.ylabel('Chiều rộng sepal (cm)')
plt.title('Biểu đồ phân tán chiều dài và chiều rộng sepal của hoa Iris')
plt.legend()  # Hiển thị chú thích
plt.grid(True)
plt.show()

#Cau 4:
iris = load_iris()
X = iris.data  # Dữ liệu đặc trưng
# Áp dụng PCA để giảm chiều xuống còn 3
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
# Đưa dữ liệu vào DataFrame để dễ xem
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Species'] = [iris.target_names[i] for i in iris.target]
# Hiển thị dữ liệu sau khi giảm chiều
print("Dữ liệu sau khi giảm chiều với PCA:")
print(df_pca.head())

#Cau 5:
iris = load_iris()
X = iris.data  # Dữ liệu đặc trưng
y = iris.target  # Nhãn loại hoa
# Chia dữ liệu: 140 mẫu cho training, 10 mẫu cho test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=140, test_size=10, random_state=42)
# Khởi tạo mô hình KNN với k = 3
knn = KNeighborsClassifier(n_neighbors=3)
# Huấn luyện mô hình
knn.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)
# Hiển thị kết quả dự đoán và nhãn thực tế
print("Dự đoán loại hoa:", y_pred)
print("Loại hoa thực tế :", y_test)
print("Độ chính xác: {:.2f}%".format(knn.score(X_test, y_test) * 100))

#Cau 6:
knn_k5 = KNeighborsClassifier(n_neighbors=5)
# Huấn luyện mô hình
knn_k5.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred_k5 = knn_k5.predict(X_test)
# Hiển thị kết quả dự đoán và nhãn thực tế
print("Dự đoán loại hoa với K=5:", y_pred_k5)
print("Loại hoa thực tế:", y_test)
print("Độ chính xác với K=5: {:.2f}%".format(knn_k5.score(X_test, y_test) * 100))

#Cau 7:
comparison = pd.DataFrame({
    'Dự đoán (K=3)': y_pred,
    'Dự đoán (K=5)': y_pred_k5,
    'Thực tế': y_test
})

print("So sánh kết quả dự đoán và thực tế:")
print(comparison)

#Cau 8:
# Chọn hai đặc trưng: chiều dài sepal (cột 0) và chiều rộng sepal (cột 1)
X_2d = X[:, :2]
# Khởi tạo mô hình KNN với K=3 và K=5
knn_k3 = KNeighborsClassifier(n_neighbors=3)
knn_k5 = KNeighborsClassifier(n_neighbors=5)
# Huấn luyện trên tập dữ liệu 2D
knn_k3.fit(X_2d, y)
knn_k5.fit(X_2d, y)
# Tạo lưới điểm để vẽ decision boundary
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
# Dự đoán trên lưới điểm với từng mô hình
Z_k3 = knn_k3.predict(np.c_[xx.ravel(), yy.ravel()])
Z_k5 = knn_k5.predict(np.c_[xx.ravel(), yy.ravel()])
# Định dạng lại dữ liệu để vẽ biểu đồ
Z_k3 = Z_k3.reshape(xx.shape)
Z_k5 = Z_k5.reshape(xx.shape)
# Vẽ Decision Boundary cho K=3
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_k3, alpha=0.4, cmap=plt.cm.rainbow)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolor='k', cmap=plt.cm.rainbow)
plt.xlabel('Chiều dài sepal (cm)')
plt.ylabel('Chiều rộng sepal (cm)')
plt.title('Decision Boundary - KNN với K=3')
# Vẽ Decision Boundary cho K=5
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_k5, alpha=0.4, cmap=plt.cm.rainbow)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolor='k', cmap=plt.cm.rainbow)
plt.xlabel('Chiều dài sepal (cm)')
plt.ylabel('Chiều rộng sepal (cm)')
plt.title('Decision Boundary - KNN với K=5')
plt.tight_layout()
plt.show()

#Cau 9:
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Khởi tạo và huấn luyện mô hình Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Hiển thị kết quả dự đoán và giá trị thực tế
print("Giá trị dự đoán:", np.round(y_pred[:10], 2))
print("Giá trị thực tế :", y_test[:10])
# Tính toán độ chính xác của mô hình
score = model.score(X_test, y_test)
print(f"Độ chính xác của mô hình: {score:.2f}")

#Cau 10:
# Chia dữ liệu thành training set và test set theo yêu cầu
X_train = X[:422]     # Dữ liệu huấn luyện - 422 bệnh nhân đầu tiên
y_train = y[:422]
X_test = X[422:]      # Dữ liệu kiểm tra - 20 bệnh nhân cuối cùng
y_test = y[422:]
# Hiển thị kích thước của tập dữ liệu
print("Kích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm tra:", X_test.shape)

#Cau 11:
model = LinearRegression()
model.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra (test set)
y_pred = model.predict(X_test)
# Hiển thị kết quả dự đoán và giá trị thực tế
print("Giá trị dự đoán:", np.round(y_pred[:10], 2))
print("Giá trị thực tế :", y_test[:10])
# Tính độ chính xác của mô hình trên tập kiểm tra
score = model.score(X_test, y_test)
print(f"Độ chính xác của mô hình (R² score): {score:.2f}")

#Cau 12:
# Hệ số hồi quy của mô hình sau khi huấn luyện
b_coefficients = model.coef_
# Hiển thị 10 hệ số hồi quy tương ứng với 10 đặc trưng
print("Hệ số hồi quy (b coefficients):")
print(np.round(b_coefficients, 4))

#Cau 13:
comparison_df = pd.DataFrame({'Giá trị dự đoán': np.round(y_pred, 2), 'Giá trị thực tế': y_test})
print("So sánh giá trị dự đoán và giá trị thực tế trên test set:")
print(comparison_df.head(10))

#Cau 14:
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

#Cau 15:
# Lấy đặc trưng tuổi (cột thứ 2 của dataset) làm dữ liệu huấn luyện
X_age_train = X_train[:, 2].reshape(-1, 1)  # Cột thứ 2: Age
X_age_test = X_test[:, 2].reshape(-1, 1)
# Khởi tạo và huấn luyện mô hình Linear Regression
model_age = LinearRegression()
model_age.fit(X_age_train, y_train)
# Dự đoán trên test set
y_pred_age = model_age.predict(X_age_test)
# So sánh giá trị dự đoán và giá trị thực tế
print("Giá trị dự đoán (dựa trên age):", np.round(y_pred_age[:10], 2))
print("Giá trị thực tế:", y_test[:10])
# Đánh giá độ chính xác của mô hình
score_age = model_age.score(X_age_test, y_test)
print(f"Độ chính xác của mô hình với age (R² score): {score_age:.2f}")

#Cau 16:
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
# Lấy tên các đặc trưng (feature names)
feature_names = diabetes.feature_names
n_features = X.shape[1]
plt.figure(figsize=(15, 10))
# Huấn luyện mô hình cho từng đặc trưng và vẽ biểu đồ
for i in range(n_features):
    X_feature = X[:, i].reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_feature, y)
    y_pred = model.predict(X_feature)
    plt.subplot(2, 5, i + 1)
    plt.scatter(X_feature, y, color='blue', label='Giá trị thực tế', alpha=0.6)
    plt.plot(X_feature, y_pred, color='red', label='Dự đoán (Linear Regression)')
    plt.xlabel(feature_names[i])
    plt.ylabel('Mục tiêu (Target)')
    plt.legend()
    plt.title(f'Hồi quy tuyến tính - {feature_names[i]}')
plt.tight_layout()
plt.show()

#Cau 17:
breast_cancer = load_breast_cancer() #dataset
print("Các key của dictionary:", breast_cancer.keys())

#Cau 18:
breast_cancer = load_breast_cancer()
# Kiểm tra kích thước của dữ liệu
print("Kích thước của dữ liệu (data.shape):", breast_cancer.data.shape)
# Đếm số lượng mỗi loại u
import numpy as np
unique, counts = np.unique(breast_cancer.target, return_counts=True)
# Hiển thị kết quả
print("Số lượng u ác tính (malignant):", counts[0])
print("Số lượng u lành tính (benign):", counts[1])

#Cau 19:
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_accuracies = []
test_accuracies = []
# Thử nghiệm với số lượng hàng xóm từ 1 đến 10
for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    # Dự đoán và tính độ chính xác
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
# Vẽ biểu đồ trực quan
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), train_accuracies, marker='o', label='Độ chính xác trên tập huấn luyện', color='blue')
plt.plot(range(1, 11), test_accuracies, marker='o', label='Độ chính xác trên tập kiểm tra', color='red')
plt.xlabel("Số lượng hàng xóm (K)")
plt.ylabel("Độ chính xác (Accuracy)")
plt.title("Đánh giá mô hình KNN với các giá trị K từ 1 đến 10")
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)
plt.show()

#Cau 20:
# Tạo dataset make_forge
X, y = mglearn.datasets.make_forge()
# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Khởi tạo mô hình
logreg = LogisticRegression()
linear_svc = LinearSVC(max_iter=10000)
# Huấn luyện mô hình
logreg.fit(X_train, y_train)
linear_svc.fit(X_train, y_train)
# Trực quan hóa phân tách của Logistic Regression
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
mglearn.plots.plot_2d_separator(logreg, X, fill=True, alpha=0.3)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.title("Logistic Regression")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
# Trực quan hóa phân tách của Linear SVC
plt.subplot(1, 2, 2)
mglearn.plots.plot_2d_separator(linear_svc, X, fill=True, alpha=0.3)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.title("Linear SVC")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
# Đánh giá mô hình trên tập kiểm tra
print("Độ chính xác Logistic Regression (test):", logreg.score(X_test, y_test))
print("Độ chính xác Linear SVC (test):", linear_svc.score(X_test, y_test))