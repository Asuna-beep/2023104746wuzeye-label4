
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ====================== 任务1：数据准备 ======================
digits = load_digits()
X = digits.data
y = digits.target

print("===== 数据集基本信息 =====")
print("样本总数：", len(X))
print("图像尺寸：8×8")
print("特征向量形状：", X.shape)
print("类别标签：0~9")
print("类别数量：10")

print("\n===== 展示前10张样本图像 =====")
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"label:{y[i]}")
    plt.axis("off")
plt.suptitle("Sample Images (0-9)")
plt.show()

# ====================== 任务2：数据划分 ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\n===== 数据划分 =====")
print("训练集数量：", len(X_train))
print("测试集数量：", len(X_test))

# ====================== 任务3：特征表示 ======================
print("\n===== 特征表示 =====")
print("每张8×8图像按行展开为64维特征向量")
print("传统机器学习只能处理向量，不能直接处理图像")

sample_img = digits.images[0]  # 取第1张图片
sample_feature = X[0]          # 对应的64维特征

plt.figure(figsize=(10, 4))
# 左边：原始图像
plt.subplot(1, 2, 1)
plt.imshow(sample_img, cmap='gray')
plt.title("原始图像 (8×8)")
plt.axis('off')

# 右边：展平后的64维特征向量
plt.subplot(1, 2, 2)
plt.bar(range(len(sample_feature)), sample_feature)
plt.title("特征表示：64维特征向量")
plt.xlabel("特征维度")
plt.ylabel("像素值")
plt.tight_layout()
plt.show()
# ====================================================

# ====================== 任务4：模型训练（6种） ======================
models = [
    ("KNN", KNeighborsClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("Logistic Regression", LogisticRegression(max_iter=10000)),
    ("SVM", SVC()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier())
]

print("\n===== 各模型测试集准确率 ===================")
acc_list = []
for name, model in models:
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    acc_list.append((name, acc))
    print(f"{name:<20s}: {acc:.4f}")

# ====================== 任务5：结果对比（表格化输出） ======================
print("\n===== 模型准确率汇总表 =====")
print("| 模型                | 测试准确率 |")
print("|---------------------|------------|")
for name, acc in acc_list:
    print(f"| {name:<18s}| {acc:.4f}     |")

# ====================== 任务6：错误样本分析（选SVM为最优模型） ======================
best_model = SVC()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# 混淆矩阵（强制显示0-9标签）
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4,5,6,7,8,9])

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix (Best Model: SVM)")
plt.show()

# 错误样本可视化
wrong_idx = np.where(y_pred != y_test)[0]
print("\n错误分类样本数：", len(wrong_idx))

plt.figure(figsize=(10, 4))
for i, idx in enumerate(wrong_idx[:4]):
    plt.subplot(1, 4, i+1)
    img = X_test[idx].reshape(8, 8)
    plt.imshow(img, cmap="gray")
    plt.title(f"True:{y_test[idx]}\nPred:{y_pred[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.show()