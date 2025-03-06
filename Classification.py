import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import openml
import joblib
import shutil
import pandas as pd
import os
import mlflow
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from streamlit_drawable_canvas import st_canvas
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from mlflow.tracking import MlflowClient

# Load dữ liệu MNIST
def ly_thuyet_Decision_tree():
    st.header("📖 Lý thuyết về Decision Tree") 
    st.header("🌳 Giới thiệu về Decision Tree")
    st.markdown(" ### 1️⃣ Decision Tree là gì?")
    st.write("""
    Decision Tree (Cây quyết định) là một thuật toán học có giám sát được sử dụng trong **phân loại (classification)** và **hồi quy (regression)**.
    Nó hoạt động bằng cách chia dữ liệu thành các nhóm nhỏ hơn dựa trên các điều kiện được thiết lập tại các **nút (nodes)** của cây.
    """) 
    
    st.markdown(" ### 📌 Cấu trúc của Decision Tree") 
    image_url = "https://trituenhantao.io/wp-content/uploads/2019/06/dt.png"
    st.image(image_url, caption="Ví dụ về cách Cây quyết định phân chia dữ liệu", use_column_width=True)

    st.write("""
    - **Nút gốc (Root Node)**: Là điểm bắt đầu của cây, chứa toàn bộ dữ liệu.
    - **Nút quyết định (Decision Nodes)**: Các nút trung gian nơi dữ liệu được chia nhỏ dựa trên một điều kiện.
    - **Nhánh (Branches)**: Các đường nối giữa các nút, thể hiện lựa chọn có thể xảy ra.
    - **Nút lá (Leaf Nodes)**: Điểm cuối của cây, đại diện cho quyết định cuối cùng hoặc nhãn dự đoán.
    """)

    st.markdown(" ### 🔍 Cách hoạt động của Decision Tree")
    st.write("""
    1. **Chọn đặc trưng tốt nhất để chia dữ liệu** bằng các tiêu chí như:
    - Gini Impurity: Đánh giá độ lẫn lộn của tập dữ liệu.
    - Entropy (dùng trong ID3): Xác định mức độ không chắc chắn.
    - Reduction in Variance (dùng cho hồi quy).
    2. **Tạo các nhánh con** từ đặc trưng được chọn.
    3. **Lặp lại quy trình** trên từng nhánh con cho đến khi đạt điều kiện dừng.
    4. **Dự đoán dữ liệu mới** bằng cách đi theo cây từ gốc đến lá.
    """)

    st.markdown("### 🌳 Công Thức Chính của Cây Quyết Định và Cách Áp Dụng")

    st.subheader("📌 1. Entropy – Độ hỗn loạn của dữ liệu")
    st.write("Entropy đo lường mức độ hỗn loạn trong dữ liệu. Nếu một tập dữ liệu càng đồng nhất, entropy càng thấp.")
    st.latex(r"H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i")

    st.write("""
    - Nếu tất cả dữ liệu thuộc cùng một lớp → Entropy = 0 (thuần khiết).
    - Nếu dữ liệu được phân bố đều giữa các lớp → Entropy đạt giá trị cao nhất.
    """)

    st.subheader("📌 2. Information Gain – Mức độ giảm độ hỗn loạn sau khi chia dữ liệu")
    st.latex(r"IG = H(S) - \sum_{j=1}^{k} \frac{|S_j|}{|S|} H(S_j)")

    st.write("""
    - IG càng cao → thuộc tính đó giúp phân loại dữ liệu tốt hơn.
    - IG thấp → thuộc tính đó không có nhiều giá trị trong việc phân tách dữ liệu.
    """)

    st.subheader("📌 3. Gini Impurity – Đo lường mức độ hỗn loạn thay thế Entropy")
    st.latex(r"Gini(S) = 1 - \sum_{i=1}^{c} p_i^2")

    st.write("""
    - Gini = 0 → tập dữ liệu hoàn toàn thuần khiết.
    - Gini càng cao → dữ liệu càng hỗn loạn.
    """)

    st.subheader("💡 Cách Áp Dụng để Xây Dựng Decision Tree")
    st.write("""
    1. Tính Entropy hoặc Gini của tập dữ liệu ban đầu.
    2. Tính Entropy hoặc Gini của từng tập con sau khi chia theo từng thuộc tính.
    3. Tính Information Gain cho từng thuộc tính.
    4. Chọn thuộc tính có Information Gain cao nhất để chia nhánh.
    5. Lặp lại quy trình trên cho đến khi tất cả dữ liệu trong các nhánh đều thuần khiết hoặc đạt điều kiện dừng.
    """)

    st.markdown("""
    **📌 Lưu ý:**  
    - Nếu cây quá sâu → có thể gây overfitting, cần sử dụng cắt tỉa (pruning).  
    - Decision Tree có thể sử dụng với cả phân loại (Classification) và hồi quy (Regression).  
    """)

    st.write("🚀 Cây quyết định là một thuật toán mạnh mẽ và dễ hiểu, nhưng cần điều chỉnh để tránh overfitting và tối ưu hiệu suất!") 
    
    
    
def ly_thuyet_SVM():
    # Tiêu đề chính
    st.title("📖 Lý Thuyết Về Support Vector Machine (SVM)")
    st.image("https://neralnetwork.wordpress.com/wp-content/uploads/2018/01/svm1.png", caption="Hình minh họa SVM")

    st.markdown("""
    Support Vector Machine (SVM) là một thuật toán học máy mạnh mẽ thường được sử dụng cho bài toán **phân loại (classification)** và **hồi quy (regression)**. 
    Nó hoạt động dựa trên nguyên lý tìm **siêu phẳng (hyperplane)** tối ưu để phân tách dữ liệu.
    """)

    # 1. Nguyên lý hoạt động
    st.header("1. Nguyên Lý Hoạt Động của SVM")

    st.subheader("📌 1.1. Tìm Siêu Phẳng Tối Ưu")
    st.markdown("""
    - Một **siêu phẳng (hyperplane)** là một đường (trong không gian 2D) hoặc một mặt phẳng (trong không gian 3D) dùng để phân tách dữ liệu thành các nhóm.
    - SVM tìm **siêu phẳng tối ưu** sao cho khoảng cách từ siêu phẳng đến các điểm dữ liệu gần nhất (**support vectors**) là lớn nhất.
    """)

    st.write("🚀 **Công thức siêu phẳng:**")
    st.latex(r"w \cdot x + b = 0")

    st.markdown("""
    Trong đó:
    - \( w \) là **vector trọng số**,
    - \( x \) là **vector dữ liệu đầu vào**,
    - \( b \) là **bias**.
    """)

    st.subheader("📌 1.2. Khoảng Cách Lề (Margin)")
    st.markdown("""
    - **Soft Margin SVM**: Chấp nhận một số điểm bị phân loại sai nhưng tăng khả năng tổng quát hóa (**giảm overfitting**).
    - **Hard Margin SVM**: Yêu cầu phân tách hoàn hảo, không cho phép lỗi nhưng dễ bị **overfitting**.
    """)

    # 2. Hàm mục tiêu
    st.header("2. Hàm Mục Tiêu trong SVM")
    st.markdown("""
    Mục tiêu của SVM là tìm \( w \) và \( b \) để tối đa hóa khoảng cách lề \( \frac{2}{||w||} \), tương đương với bài toán tối ưu:
    """)

    st.latex(r"\min_{w, b} \frac{1}{2} ||w||^2")

    st.markdown("Sao cho:")

    st.latex(r"y_i (w \cdot x_i + b) \geq 1, \forall i")

    st.markdown("""
    Trong đó:
    - \( y_i \) là **nhãn của dữ liệu** (1 hoặc -1),
    - \( x_i \) là **điểm dữ liệu**.
    """)

    # 3. Kernel Trick
    st.header("3. Kernel Trick – Mở Rộng SVM Cho Dữ Liệu Phi Tuyến")
    st.markdown("""
    Khi dữ liệu không thể phân tách tuyến tính, SVM sử dụng **hàm kernel** để ánh xạ dữ liệu vào không gian chiều cao hơn, nơi có thể phân tách tuyến tính.

    📌 **Một số loại Kernel phổ biến**:
    """)

    st.subheader("1️⃣ Linear Kernel")
    st.latex(r"K(x_i, x_j) = x_i \cdot x_j")
    st.markdown("👉 Sử dụng khi dữ liệu có thể phân tách tuyến tính.")

    st.subheader("2️⃣ Polynomial Kernel")
    st.latex(r"K(x_i, x_j) = (x_i \cdot x_j + c)^d")
    st.markdown("👉 Phù hợp với dữ liệu có ranh giới phi tuyến.")

    st.subheader("3️⃣ Radial Basis Function (RBF) Kernel")
    st.latex(r"K(x_i, x_j) = \exp(- \gamma ||x_i - x_j||^2)")
    st.markdown("👉 Phổ biến nhất vì có thể xử lý **mọi loại dữ liệu**.")

    st.write("🚀 SVM là một thuật toán mạnh mẽ, nhưng cần điều chỉnh đúng tham số để đạt hiệu suất tối ưu!")


def data():
    st.header("MNIST Dataset")
    st.title("Tổng quan về tập dữ liệu MNIST")

    st.header("1. Giới thiệu")
    st.write("Tập dữ liệu MNIST (Modified National Institute of Standards and Technology) là một trong những tập dữ liệu phổ biến nhất trong lĩnh vực Machine Learning và Computer Vision, thường được dùng để huấn luyện và kiểm thử các mô hình phân loại chữ số viết tay.") 

    st.image("https://datasets.activeloop.ai/wp-content/uploads/2019/12/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp", use_container_width=True)

    st.subheader("Nội dung")
    st.write("- 70.000 ảnh grayscale (đen trắng) của các chữ số viết tay từ 0 đến 9.")
    st.write("- Kích thước ảnh: 28x28 pixel.")
    st.write("- Định dạng: Mỗi ảnh được biểu diễn bằng một ma trận 28x28 có giá trị pixel từ 0 (đen) đến 255 (trắng).")
    st.write("- Nhãn: Một số nguyên từ 0 đến 9 tương ứng với chữ số trong ảnh.")

    st.header("2. Nguồn gốc và ý nghĩa")
    st.write("- Được tạo ra từ bộ dữ liệu chữ số viết tay gốc của NIST, do LeCun, Cortes và Burges chuẩn bị.")
    st.write("- Dùng làm benchmark cho các thuật toán nhận diện hình ảnh, đặc biệt là mạng nơ-ron nhân tạo (ANN) và mạng nơ-ron tích chập (CNN).")
    st.write("- Rất hữu ích cho việc kiểm thử mô hình trên dữ liệu hình ảnh thực tế nhưng đơn giản.")

    st.header("3. Phân chia tập dữ liệu")
    st.write("- Tập huấn luyện: 60.000 ảnh.")
    st.write("- Tập kiểm thử: 10.000 ảnh.")
    st.write("- Mỗi tập có phân bố đồng đều về số lượng chữ số từ 0 đến 9.")

    st.header("4. Ứng dụng")
    st.write("- Huấn luyện và đánh giá các thuật toán nhận diện chữ số viết tay.")
    st.write("- Kiểm thử và so sánh hiệu suất của các mô hình học sâu (Deep Learning).")
    st.write("- Làm bài tập thực hành về xử lý ảnh, trích xuất đặc trưng, mô hình phân loại.")
    st.write("- Cung cấp một baseline đơn giản cho các bài toán liên quan đến Computer Vision.")

    st.header("5. Phương pháp tiếp cận phổ biến")
    st.write("- Trích xuất đặc trưng truyền thống: PCA, HOG, SIFT...")
    st.write("- Machine Learning: KNN, SVM, Random Forest, Logistic Regression...")
    st.write("- Deep Learning: MLP, CNN (LeNet-5, AlexNet, ResNet...), RNN")

    st.caption("Ứng dụng hiển thị thông tin về tập dữ liệu MNIST bằng Streamlit 🚀")
    


def up_load_db():
    # Tiêu đề
    st.header("📥 Tải Dữ Liệu")

    # Kiểm tra xem dữ liệu đã tải chưa
    if "data" in st.session_state and st.session_state.data is not None:
        st.warning("🔸 **Dữ liệu đã được tải lên rồi!** Bạn có thể tiếp tục với các bước tiền xử lý và chia dữ liệu.")
    else:
        # Chọn nguồn dữ liệu
        option = st.radio("Chọn nguồn dữ liệu:", ["Tải từ OpenML", "Upload dữ liệu"], key="data_source_radio")

        # Biến để lưu trữ dữ liệu
        if "data" not in st.session_state:
            st.session_state.data = None

        # Nếu chọn tải từ OpenML
        if option == "Tải từ OpenML":
            st.markdown("#### 📂 Tải dữ liệu MNIST từ OpenML")
            if st.button("Tải dữ liệu MNIST", key="download_mnist_button"):
                st.write("🔄 Đang tải dữ liệu MNIST từ OpenML...")
                
                # Tải dữ liệu MNIST từ file .npy
                X = np.load("X.npy")
                y = np.load("y.npy")
                
                st.success("✅ Dữ liệu MNIST đã được tải thành công!")
                st.session_state.data = (X, y)  # Lưu dữ liệu vào session_state

        # Nếu chọn upload dữ liệu từ máy
        else:
            st.markdown("#### 📤 Upload dữ liệu của bạn")

            uploaded_file = st.file_uploader("Chọn một file ảnh", type=["png", "jpg", "jpeg"], key="file_upload")

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

                if image.size != (28, 28):
                    st.error("❌ Ảnh không đúng kích thước 28x28 pixel. Vui lòng tải lại ảnh đúng định dạng.")
                else:
                    st.success("✅ Ảnh hợp lệ!")
                    image = image.convert('L')
                    image_array = np.array(image).reshape(1, 28, 28, 1)
                    st.session_state.data = image_array

    # Kiểm tra nếu dữ liệu đã được tải
    if st.session_state.data is not None:
        st.markdown("#### ✅ Dữ liệu đã sẵn sàng!")
        
        if isinstance(st.session_state.data, tuple):
            X, y = st.session_state.data
            st.markdown("##### 🔄 Tiến hành tiền xử lý dữ liệu MNIST")

            preprocess_option = st.selectbox("Chọn phương pháp tiền xử lý dữ liệu:", 
                                            ["Chuẩn hóa dữ liệu (Normalization)", "Chuẩn hóa dữ liệu (Standardization)", "Xử lý dữ liệu missing", "Không tiền xử lý"], key="preprocess_mnist")

            X_reshaped = X.reshape(X.shape[0], -1)
            
            st.markdown("### Ảnh chưa tiền xử lý")
            fig, axes = plt.subplots(1, 5, figsize=(10, 2))
            for i in range(5):
                axes[i].imshow(X[i].reshape(28, 28), cmap='gray')
                axes[i].set_title(f"Label: {y[i]}")
                axes[i].axis('off')
            st.pyplot(fig)
            
            st.markdown("### Kết quả sau khi tiền xử lý")
            fig, axes = plt.subplots(1, 5, figsize=(10, 2))
            
            if preprocess_option == "Chuẩn hóa dữ liệu (Normalization)":
                X_normalized = MinMaxScaler().fit_transform(X_reshaped)
                for i in range(5):
                    axes[i].imshow(X_normalized[i].reshape(28, 28), cmap='gray')
                    axes[i].set_title(f"Label: {y[i]}")
                    axes[i].axis('off')
                st.success("✅ Đã chuẩn hóa dữ liệu!")
            
            elif preprocess_option == "Chuẩn hóa dữ liệu (Standardization)":
                X_standardized = StandardScaler().fit_transform(X_reshaped)
                for i in range(5):
                    axes[i].imshow(X_standardized[i].reshape(28, 28), cmap='gray')
                    axes[i].set_title(f"Label: {y[i]}")
                    axes[i].axis('off')
                st.success("✅ Đã chuẩn hóa dữ liệu!")
            
            elif preprocess_option == "Xử lý dữ liệu missing":
                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X_reshaped)
                for i in range(5):
                    axes[i].imshow(X_imputed[i].reshape(28, 28), cmap='gray')
                    axes[i].set_title(f"Label: {y[i]}")
                    axes[i].axis('off')
                st.success("✅ Đã xử lý dữ liệu thiếu!")
            else:
                for i in range(5):
                    axes[i].imshow(X[i].reshape(28, 28), cmap='gray')
                    axes[i].set_title(f"Label: {y[i]}")
                    axes[i].axis('off')
                st.success("✅ Không thực hiện tiền xử lý!")
            
            st.pyplot(fig)
    
    else:
        st.warning("🔸 Vui lòng tải dữ liệu trước khi tiếp tục làm việc.")

def chia_du_lieu():
    st.title("📌 Chia dữ liệu Train/Test")

    # Kiểm tra xem dữ liệu đã được tải hay chưa
    if not os.path.exists("X.npy") or not os.path.exists("y.npy"):
        st.error("❌ Dữ liệu chưa được tải! Vui lòng tải dữ liệu trước khi chia.")
        return

    # Đọc dữ liệu từ file
    X = np.load("X.npy")
    y = np.load("y.npy")
    total_samples = X.shape[0]

    # Nếu dữ liệu đã được chia trước đó, hiển thị thông tin và không chia lại
    if "X_train" in st.session_state:
        st.success("✅ **Dữ liệu đã được chia, không cần chạy lại!**")

        # Hiển thị bảng dữ liệu đã chia
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [
                len(st.session_state["X_train"]),
                len(st.session_state["X_val"]),
                len(st.session_state["X_test"])
            ]
        })
        st.table(summary_df)
        return

    # Thanh chọn số lượng ảnh để train
    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, total_samples, 10000)

    # Thanh chọn % dữ liệu Test
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size  # Tính phần còn lại của tập Train

    # Thanh chọn % dữ liệu Validation (trong tập Train)
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)

    st.markdown(f"### 📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    if st.button("✅ Xác nhận & Lưu"):
        # Chọn tập dữ liệu theo số lượng mẫu mong muốn
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)

        # Chia train/test
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_selected, y_selected, 
                                                                      test_size=test_size / 100, 
                                                                      stratify=y_selected, random_state=42)

        # Chia train/val
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, 
                                                          test_size=val_size / (100 - test_size), 
                                                          stratify=y_train_full, random_state=42)

        # Lưu dữ liệu vào session_state để sử dụng sau này
        st.session_state["X_train"] = X_train
        st.session_state["X_val"] = X_val
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_val"] = y_val
        st.session_state["y_test"] = y_test

        # Tạo bảng hiển thị số lượng mẫu của từng tập dữ liệu
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })

        st.success("✅ **Dữ liệu đã được chia thành công!**")
        st.table(summary_df)  # Hiển thị bảng dữ liệu

def train():
    """Huấn luyện mô hình Decision Tree hoặc SVM và lưu trên MLflow."""
    mlflow_input()
    # 📥 Kiểm tra dữ liệu
    if not all(key in st.session_state for key in ["X_train", "y_train", "X_test", "y_test"]):
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train, y_train = st.session_state["X_train"], st.session_state["y_train"]
    X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]

    # 🌟 Chuẩn hóa dữ liệu
    X_train, X_test = X_train.reshape(-1, 28 * 28) / 255.0, X_test.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    # 📌 Lựa chọn mô hình
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])
    
    if model_choice == "Decision Tree":
        criterion = st.selectbox("Criterion (Hàm mất mát: Gini/Entropy) ", ["gini", "entropy"])
        max_depth = st.slider("max_depth (\(d\))", 1, 20, 5, help="Giới hạn độ sâu của cây để tránh overfitting.")
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    else:
        C = st.slider("C (Hệ số điều chuẩn \(C\))", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel (Hàm nhân \(K\))", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel)

    # 📌 Chọn số folds cho KFold Cross-Validation
    k_folds = st.slider("Số folds (\(k\))", 2, 10, 5, help="Số tập chia để đánh giá mô hình.")

    # 🚀 Bắt đầu huấn luyện
    if st.button("Huấn luyện mô hình"):
        if "mlflow_url" not in st.session_state:
            st.session_state["mlflow_url"] = "https://dagshub.com/Snxtruc/HocMayVoiPython.mlflow"

        with mlflow.start_run():
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            # Huấn luyện trên tập Cross-Validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                model.fit(X_train_fold, y_train_fold)
                val_pred = model.predict(X_val_fold)
                val_acc = accuracy_score(y_val_fold, val_pred)
                cv_scores.append(val_acc)
                mlflow.log_metric("cv_accuracy", val_acc, step=fold)

            cv_accuracy_mean = np.mean(cv_scores)
            cv_accuracy_std = np.std(cv_scores)

            # Hiển thị kết quả Cross-Validation
            st.success(f"✅ **Cross-Validation Accuracy:** {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}")

            # Huấn luyện trên tập Test Set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("test_accuracy", test_acc)

            # Hiển thị kết quả Test Set
            st.success(f"✅ **Độ chính xác trên test set:** {test_acc:.4f}")

            # Ghi log lên MLflow
            mlflow.log_param("model", model_choice)
            mlflow.log_param("k_folds", k_folds)
            if model_choice == "Decision Tree":
                mlflow.log_param("criterion", criterion)
                mlflow.log_param("max_depth", max_depth)
            elif model_choice == "SVM":
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)

            mlflow.log_metric("cv_accuracy_mean", cv_accuracy_mean)
            mlflow.log_metric("cv_accuracy_std", cv_accuracy_std)
            mlflow.sklearn.log_model(model, model_choice.lower())

            # 📌 Lưu mô hình vào session_state
            if "models" not in st.session_state:
                st.session_state["models"] = []

            model_name = model_choice.lower().replace(" ", "_")
            if model_choice == "Decision Tree":
                model_name += f"_{criterion}_depth{max_depth}"
            elif model_choice == "SVM":
                model_name += f"_{kernel}"

            # Xử lý trùng lặp tên mô hình
            existing_names = {m["name"] for m in st.session_state["models"]}
            count = 1
            while model_name in existing_names:
                model_name = f"{model_name}_{count}"
                count += 1

            st.session_state["models"].append({"name": model_name, "model": model})
            st.write(f"🔹 Mô hình đã được lưu với tên: **{model_name}**")
            st.write(f"📋 Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

            # Hiển thị danh sách mô hình đã lưu
            model_names = [m["name"] for m in st.session_state["models"]]
            st.write("📋 Danh sách mô hình đã lưu:", ", ".join(model_names))

            st.success("📌 Mô hình đã được lưu trên MLflow!")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")


def mlflow_input():
    #st.title("🚀 MLflow DAGsHub Tracking với Streamlit")
    DAGSHUB_USERNAME = "Snxtruc"  # Thay bằng username của bạn
    DAGSHUB_REPO_NAME = "HocMayPython"
    DAGSHUB_TOKEN = "ca4b78ae4dd9d511c1e0c333e3b709b2cd789a19"  # Thay bằng Access Token của bạn

    # Đặt URI MLflow để trỏ đến DagsHub
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

    # Thiết lập authentication bằng Access Token
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    # Đặt thí nghiệm MLflow
    mlflow.set_experiment("Classifications")   

    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"


def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"⚠️ Không tìm thấy mô hình tại `{path}`")
        st.stop()

# ✅ Xử lý ảnh từ canvas (chuẩn 28x28 cho MNIST)
def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None


def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is None:
        return None
    img = Image.fromarray((canvas_result.image_data[:, :, 0] * 255).astype(np.uint8))
    img = img.convert("L").resize((28, 28))  # Chuyển sang ảnh xám 28x28
    img = np.array(img) / 255.0  # Chuẩn hóa
    return img.reshape(1, -1)


def display_mlflow_experiments():
    try:
        st.title("🔍 Quản lý MLflow Experiments")

        # Kết nối MlflowClient
        client = MlflowClient()

        # Lấy danh sách thí nghiệm
        experiments = mlflow.search_experiments()
        
        if experiments:
            st.write("### 📌 Danh sách Thí nghiệm")
            experiment_data = [
                {"Experiment ID": exp.experiment_id, "Experiment Name": exp.name, "Artifact Location": exp.artifact_location}
                for exp in experiments
            ]
            st.data_editor(pd.DataFrame(experiment_data))
            
            # Chọn thí nghiệm
            selected_exp_id = st.selectbox("🗂 Chọn thí nghiệm", sorted([exp.experiment_id for exp in experiments]))
            
            # Đổi tên thí nghiệm
            new_exp_name = st.text_input("✏️ Nhập tên mới cho thí nghiệm", "")
            if st.button("💾 Đổi tên") and new_exp_name:
                client.rename_experiment(selected_exp_id, new_exp_name)
                st.success("✅ Đổi tên thành công! Vui lòng tải lại trang.")
            
            # Xóa thí nghiệm
            if st.button("🗑️ Xóa thí nghiệm"):
                client.delete_experiment(selected_exp_id)
                st.success("✅ Xóa thí nghiệm thành công! Vui lòng tải lại trang.")
            
            # Lấy danh sách runs trong thí nghiệm đã chọn
            runs = client.search_runs(experiment_ids=[selected_exp_id])
            if runs:
                st.write("### 📌 Danh sách Run")
                
                # Bộ lọc tìm kiếm Run
                search_term = st.text_input("🔍 Tìm kiếm Run", "")
                
                # Bộ lọc theo khoảng thời gian
                start_date = st.date_input("📅 Chọn ngày bắt đầu", pd.to_datetime("2023-01-01"))
                end_date = st.date_input("📅 Chọn ngày kết thúc", pd.to_datetime("today"))
                
                # Bộ lọc theo trạng thái Run
                status_filter = st.multiselect("📌 Lọc theo trạng thái", ["RUNNING", "FINISHED", "FAILED", "KILLED"], default=["RUNNING", "FINISHED"])
                
                # Hiển thị danh sách Runs
                run_data = [
                    {
                        "Run ID": run.info.run_id,
                        "Run Name": run.data.tags.get("mlflow.runName", "Unnamed"),
                        "Start Time": pd.to_datetime(run.info.start_time, unit='ms'),
                        "End Time": pd.to_datetime(run.info.end_time, unit='ms') if run.info.end_time else None,
                        "Duration": (pd.to_datetime(run.info.end_time, unit='ms') - pd.to_datetime(run.info.start_time, unit='ms')).total_seconds() if run.info.end_time else None,
                        "Status": run.info.status,
                        "Source": run.data.tags.get("mlflow.source.name", "Unknown"),
                        "Metrics": run.data.metrics
                    }
                    for run in runs
                ]
                df_runs = pd.DataFrame(run_data).sort_values(by="Start Time", ascending=False)
                
                # Áp dụng bộ lọc
                df_runs = df_runs[(df_runs["Start Time"] >= pd.to_datetime(start_date)) & (df_runs["Start Time"] <= pd.to_datetime(end_date))]
                df_runs = df_runs[df_runs["Status"].isin(status_filter)]
                
                if search_term:
                    df_runs = df_runs[df_runs["Run Name"].str.contains(search_term, case=False, na=False)]
                
                # Bộ lọc theo Metrics cụ thể
                metric_name = st.text_input("📊 Nhập tên Metric để lọc", "accuracy")
                metric_value = st.number_input("📈 Giá trị tối thiểu của Metric", min_value=0.0, step=0.01, format="%.2f")
                
                def filter_by_metric(run):
                    return metric_name in run["Metrics"] and run["Metrics"][metric_name] >= metric_value
                
                df_runs = df_runs[df_runs.apply(filter_by_metric, axis=1)]
                
                st.data_editor(df_runs)
                
                run_options = {run["Run ID"]: f"{run['Run Name']} - {run['Run ID']}" for _, run in df_runs.iterrows()}
                        
                # Chọn Run trong thí nghiệm để đổi tên hoặc xóa
                runs = client.search_runs(experiment_ids=[selected_exp_id])
                if runs:
                    run_options = {run.info.run_id: f"{run.data.tags.get('mlflow.runName', 'Unnamed')} - {run.info.run_id}" for run in runs}
                    selected_run_id = st.selectbox("✏️ Chọn Run để đổi tên", list(run_options.keys()), format_func=lambda x: run_options[x])
                    new_run_name = st.text_input("📛 Nhập tên mới cho Run", "")
                    if st.button("✅ Cập nhật tên Run") and new_run_name:
                        client.set_tag(selected_run_id, "mlflow.runName", new_run_name)
                        st.success("✅ Cập nhật tên Run thành công! Vui lòng tải lại trang.")
                    
                    selected_run_id_delete = st.selectbox("🗑️ Chọn Run để xóa", list(run_options.keys()), format_func=lambda x: run_options[x])
                    if st.button("❌ Xóa Run"):
                        client.delete_run(selected_run_id_delete)
                        st.success("✅ Xóa Run thành công! Vui lòng tải lại trang.")
                    

                # Chọn Run để xem chi tiết
                selected_run_id = st.selectbox("🔍 Chọn Run để xem chi tiết", list(run_options.keys()), format_func=lambda x: run_options[x])
                selected_run = client.get_run(selected_run_id)
                
                st.write("### 📋 Thông tin Run")
                st.write(f"**Run ID:** {selected_run_id}")
                st.write(f"**Run Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")
                st.write(f"**Start Time:** {pd.to_datetime(selected_run.info.start_time, unit='ms')}")
                st.write(f"**End Time:** {pd.to_datetime(selected_run.info.end_time, unit='ms') if selected_run.info.end_time else 'N/A'}")
                st.write(f"**Duration:** {(pd.to_datetime(selected_run.info.end_time, unit='ms') - pd.to_datetime(selected_run.info.start_time, unit='ms')).total_seconds() if selected_run.info.end_time else 'N/A'} seconds")
                st.write(f"**Status:** {selected_run.info.status}")
                st.write(f"**Source:** {selected_run.data.tags.get('mlflow.source.name', 'Unknown')}")
                
                # Hiển thị Metrics
                st.write("### 📊 Metrics")
                metrics = selected_run.data.metrics
                if metrics:
                    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric Name", "Value"])
                    st.data_editor(df_metrics)
                else:
                    st.write("📭 Không có Metrics nào.")
                
                # Hiển thị Artifacts
                artifact_uri = selected_run.info.artifact_uri
                st.write(f"**Artifact Location:** {artifact_uri}")
                
                st.write("### 📂 Danh sách Artifacts")
                artifacts = client.list_artifacts(selected_run_id)
                if artifacts:
                    artifact_paths = [artifact.path for artifact in artifacts]
                    st.write(artifact_paths)
                    for artifact in artifacts:
                        if artifact.path.endswith(".png") or artifact.path.endswith(".jpg"):
                            st.image(f"{artifact_uri}/{artifact.path}", caption=artifact.path)
                        if artifact.path.endswith(".csv") or artifact.path.endswith(".txt"):
                            with open(f"{artifact_uri}/{artifact.path}", "r") as f:
                                st.download_button(label=f"📥 Tải {artifact.path}", data=f.read(), file_name=artifact.path)
                else:
                    st.write("📭 Không có artifacts nào.")
                
                # Truy cập MLflow UI
                st.write("### 🔗 Truy cập MLflow UI")
                st.markdown("[Mở MLflow UI](https://dagshub.com/Snxtruc/HocMayVoiPython.mlflow)")
            else:
                st.warning("⚠️ Không có Run nào trong thí nghiệm này.")
        else:
            st.warning("⚠️ Không có Thí nghiệm nào được tìm thấy.")
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy danh sách thí nghiệm: {e}")

def du_doan():
    st.header("✍️ Dự đoán số")
    
    # 🔹 Chọn phương thức dự đoán
    mode = st.radio("Chọn phương thức dự đoán:", ["Vẽ số", "Upload file test"])
    
    if mode == "Vẽ số":
        # ✍️ Vẽ số
        st.subheader("🖌️ Vẽ số vào khung dưới đây:")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=150,
            width=150,
            drawing_mode="freedraw",
            key="canvas"
        )
    
    elif mode == "Upload file test":
        # 🔹 Upload file test
        st.header("📂 Dự đoán trên tập test")
        uploaded_file = st.file_uploader("Tải tập test (CSV hoặc NPY):", type=["csv", "npy"])
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                test_data = pd.read_csv(uploaded_file).values
            else:
                test_data = np.load(uploaded_file)
            
            st.write(f"📊 Dữ liệu test có {test_data.shape[0]} mẫu.")
    
    # 🔹 Danh sách mô hình có sẵn
    available_models = {
        "SVM Linear": "svm_mnist_linear.joblib",
        "SVM Poly": "svm_mnist_poly.joblib",
        "SVM Sigmoid": "svm_mnist_sigmoid.joblib",
        "SVM RBF": "svm_mnist_rbf.joblib",
    }
    
    # 📌 Chọn mô hình
    model_option = st.selectbox("🔍 Chọn mô hình:", list(available_models.keys()))
    
    # Tải mô hình
    model = joblib.load(available_models[model_option])
    st.success(f"✅ Mô hình {model_option} đã được tải thành công!")
    
    if mode == "Vẽ số":
        if st.button("Dự đoán số"):
            if canvas_result.image_data is not None:
                img = preprocess_canvas_image(canvas_result)
                st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)
                prediction = model.predict(img)
                probabilities = model.decision_function(img) if hasattr(model, 'decision_function') else model.predict_proba(img)
                confidence = np.max(probabilities) if probabilities is not None else "Không xác định"
                st.subheader(f"🔢 Kết quả dự đoán: {prediction[0]} (Độ tin cậy: {confidence:.2f})")
            else:
                st.error("⚠️ Vui lòng vẽ một số trước khi dự đoán!")
    
    elif mode == "Upload file test" and uploaded_file is not None:
        if st.button("Dự đoán trên tập test"):
            predictions = model.predict(test_data)
            probabilities = model.decision_function(test_data) if hasattr(model, 'decision_function') else model.predict_proba(test_data)
            confidences = np.max(probabilities, axis=1) if probabilities is not None else ["Không xác định"] * len(predictions)
            
            st.write("🔢 Kết quả dự đoán:")
            for i in range(min(10, len(predictions))):
                st.write(f"Mẫu {i + 1}: {predictions[i]} (Độ tin cậy: {confidences[i]:.2f})")
            
            fig, axes = plt.subplots(1, min(5, len(test_data)), figsize=(10, 2))
            for i, ax in enumerate(axes):
                ax.imshow(test_data[i].reshape(28, 28), cmap='gray')
                ax.set_title(f"{predictions[i]} ({confidences[i]:.2f})")
                ax.axis("off")
            st.pyplot(fig)




def Classification():
    # Thiết lập CSS để hỗ trợ hiển thị tabs với hiệu ứng hover và thanh cuộn
    st.markdown(
        """
        <style>
        .stTabs [role="tablist"] {
            overflow-x: auto;
            white-space: nowrap;
            display: flex;
            scrollbar-width: thin;
            scrollbar-color: #888 #f0f0f0;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar {
            height: 6px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-thumb {
            background-color: #888;
            border-radius: 3px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-track {
            background: #f0f0f0;
        }
        .stTabs [role="tab"]:hover {
            background-color: #f0f0f0;
            transition: background-color 0.3s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Tiêu đề ứng dụng
    st.title("🖥️ MNIST Classification App")

    # Tạo các tab trong giao diện Streamlit
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📖 Lý thuyết Decision Tree", 
        "📖 Lý thuyết SVM", 
        "🚀 Review database", 
        "📥 Tải dữ liệu", 
        "⚙️ Huấn luyện", 
        "Tracking mlflow",
        "🔮 Dự đoán"
    ])

    # Nội dung của từng tab
    with tab1:
        ly_thuyet_Decision_tree()

    with tab2:
        ly_thuyet_SVM()
    
    with tab3:
        data()

    with tab4:
        up_load_db()
    
    with tab5:      
        chia_du_lieu()
        train()
    
    with tab6:
        display_mlflow_experiments()

    with tab7:
        du_doan()  # Gọi hàm dự đoán để xử lý khi vào tab Dự đoán

def run(): 
    Classification()

if __name__ == "__main__":
    run()