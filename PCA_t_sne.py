import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import openml
import os
import mlflow
import plotly.express as px
import shutil
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 



def ly_thuyet_PCA(): 


    st.title("Matrix Factorization")

    st.markdown(
        """
        **Matrix Factorization** là phương pháp phân rã ma trận để trích xuất đặc trưng và giảm chiều dữ liệu.
        Các phương pháp phổ biến gồm:
        - **Principal Component Analysis (PCA)**: Giảm chiều bằng cách tìm trục chính.
        - **Singular Value Decomposition (SVD)**: Phân rã ma trận thành ba ma trận con.
        - **Non-Negative Matrix Factorization (NMF)**: Xấp xỉ ma trận với các giá trị không âm.
        """
    )

    # Tiêu đề phụ
    st.header("🔢 Khái niệm PCA")
    st.write("📉 PCA (Principal Component Analysis – Phân tích thành phần chính) là một kỹ thuật giảm chiều dữ liệu bằng cách tìm các hướng (thành phần chính) có phương sai lớn nhất trong dữ liệu.")

    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://machinelearningcoban.com/assets/27_pca/pca_var0.png" width="300">
            <p><em>Matrix Factorization</em></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.header("📌 Ý tưởng của PCA")

    st.markdown(
        """
        ### 1️⃣ Loại bỏ thành phần có phương sai nhỏ  
        - PCA tìm các hướng có **phương sai lớn nhất** để giữ lại.  
        - Các hướng có phương sai nhỏ bị loại bỏ vì chúng không đóng góp nhiều vào sự thay đổi của dữ liệu.  
        
        <div style="text-align: center;">
            <img src="https://machinelearningcoban.com/assets/27_pca/pca_diagvar.png" width="50%">
        </div>

        ### 2️⃣ Xoay dữ liệu theo trục chính  
        - PCA tìm một hệ trục tọa độ mới sao cho dữ liệu được trải dài theo các trục có phương sai lớn.  
        - Điều này giúp giảm chiều dữ liệu mà vẫn giữ lại nhiều thông tin quan trọng.  

        <div style="text-align: center;">
            <img src="https://setosa.io/ev/principal-component-analysis/fb-thumb.png" width="50%">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("📌 Công thức PCA")
    st.write("📊 PCA sử dụng giá trị kỳ vọng, phương sai, ma trận hiệp phương sai và phân rã giá trị kỳ dị (SVD - Singular Value Decomposition) để tìm các thành phần chính.")

    st.subheader("🧮 Bước 1: Chuẩn hóa dữ liệu")
    st.latex(r"""
    X = \begin{bmatrix}
    x_{11} & x_{12} & \dots & x_{1d} \\
    x_{21} & x_{22} & \dots & x_{2d} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{n1} & x_{n2} & \dots & x_{nd}
    \end{bmatrix}
    """)
    st.write("📏 Trừ đi giá trị trung bình của từng đặc trưng để đưa dữ liệu về trung tâm gốc tọa độ:")
    st.latex(r"""
    \bar{x}_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}
    """)
    st.latex(r"""
    X' = X - \bar{X}
    """)

    st.subheader("📐 Bước 2: Tính ma trận hiệp phương sai")
    st.latex(r"""
    C = \frac{1}{n-1} X'^T X'
    """)
    st.write("🔗 C là ma trận d × d, biểu diễn mối quan hệ tuyến tính giữa các đặc trưng.")

    st.subheader("🧩 Bước 3: Tính toán vector riêng và giá trị riêng")
    st.latex(r"""
    C v = \lambda v
    """)
    st.write("📌 Trong đó:")
    st.write("- 🔹 v là vector riêng (eigenvector) (hướng chính của dữ liệu).")
    st.write("- 🔸 λ là giá trị riêng (eigenvalue) (lượng phương sai giữ lại trên hướng đó).")
    st.write("📌 Chọn k vector riêng tương ứng với k giá trị riêng lớn nhất.")

    st.subheader("🔀 Bước 4: Chuyển đổi dữ liệu sang không gian mới")
    st.latex(r"""
    Z = X' W
    """)
    st.write("📂 Ma trận các thành phần chính W chứa các vector riêng tương ứng với k giá trị riêng lớn nhất.")
    st.write("📉 Ma trận Z là dữ liệu mới sau khi giảm chiều.") 

    # Thêm phần ưu điểm và nhược điểm của PCA
    st.header("✅ Ưu điểm & ❌ Nhược điểm của PCA")

    st.subheader("✅ Ưu điểm:")
    st.write("- 📊 Giảm chiều dữ liệu, giúp tăng tốc độ huấn luyện mô hình.")
    st.write("- 🎯 Loại bỏ nhiễu trong dữ liệu, giúp mô hình chính xác hơn.")
    st.write("- 🔄 Giúp trực quan hóa dữ liệu tốt hơn bằng cách giảm xuống 2D hoặc 3D.")

    st.subheader("❌ Nhược điểm:")
    st.write("- 🔍 Mất một phần thông tin khi giảm chiều, có thể ảnh hưởng đến hiệu suất mô hình.")
    st.write("- 🏷️ PCA không bảo toàn tính diễn giải của dữ liệu, do các thành phần chính không tương ứng với đặc trưng ban đầu.")
    st.write("- 🧮 Giả định rằng dữ liệu có quan hệ tuyến tính, không phù hợp với dữ liệu phi tuyến.")  
    

    st.markdown("## 📉 Minh họa thu gọn chiều bằng PCA")
    # Tham số điều chỉnh với tooltip
    num_samples = st.slider("Số điểm dữ liệu 🟢", 100, 1000, 300, step=50, help="Số lượng điểm dữ liệu được tạo ra để thực hiện phân tích PCA. Giá trị càng lớn, dữ liệu càng phong phú nhưng cũng có thể làm tăng thời gian xử lý.")
    num_features = st.slider("Số chiều ban đầu 🔵", 3, 10, 3, help="Số lượng đặc trưng (features) ban đầu của dữ liệu. PCA sẽ giúp giảm số chiều này trong khi vẫn giữ lại tối đa thông tin quan trọng.")
    num_clusters = st.slider("Số cụm 🔴", 2, 5, 3, help="Số lượng nhóm (clusters) trong dữ liệu. Dữ liệu sẽ được tạo thành các cụm trước khi áp dụng PCA.")

    # Giới hạn số thành phần PCA hợp lệ
    max_components = max(2, num_features)
    n_components = st.slider("Số thành phần PCA 🟣", 2, max_components, min(2, max_components), help="Số thành phần chính sau khi giảm chiều dữ liệu bằng PCA. Giá trị nhỏ hơn số chiều ban đầu nhưng vẫn giữ lại phần lớn thông tin quan trọng.")

    # Thêm nút Reset và Thực hiện PCA với icon
    if st.button("🔄 Reset"):
        st.rerun()

    if st.button("📊 Thực hiện PCA"):
        # Tạo dữ liệu ngẫu nhiên
        X, y = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, random_state=42)

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Áp dụng PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Vẽ biểu đồ
        fig = plt.figure(figsize=(12, 6))
        if num_features == 3:
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=y, cmap='viridis', alpha=0.6)
            ax.set_title('Dữ liệu ban đầu (3D)')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
        else:
            ax = fig.add_subplot(121)
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6)
            ax.set_title(f'Dữ liệu ban đầu ({num_features}D, chỉ hiển thị 2 trục)')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            plt.colorbar(scatter, ax=ax, label='Cluster Label')

        ax2 = fig.add_subplot(122)
        if n_components > 1:
            scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
            ax2.set_xlabel('Principal Component 1')
            ax2.set_ylabel('Principal Component 2')
        else:
            ax2.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c=y, cmap='viridis', alpha=0.6)
            ax2.set_xlabel('Principal Component 1')
            ax2.set_yticks([])
        ax2.set_title(f'Dữ liệu sau PCA ({n_components}D)')
        plt.colorbar(scatter, ax=ax2, label='Cluster Label')

        st.pyplot(fig)


def ly_thuyet_tSne():
    # Tiêu đề chính
    st.title("🔢 Tổng quan về t-SNE")

    st.write("""
    **t-SNE (t-Distributed Stochastic Neighbor Embedding)** là một thuật toán **giảm chiều dữ liệu**, 
    giúp **trực quan hóa dữ liệu cao chiều** trong **không gian thấp chiều** bằng cách **bảo toàn cấu trúc cục bộ** giữa các điểm dữ liệu.
    """)

    # Tóm tắt ý tưởng
    st.header("🔽 Ý tưởng chính")

    st.markdown("""
    - **Mục tiêu chính**: Giảm chiều dữ liệu từ không gian cao chiều xuống 2D hoặc 3D mà vẫn **bảo toàn cấu trúc cục bộ**.
    - **Cách hoạt động**:
        1. **Chuyển đổi khoảng cách thành xác suất**:
            - Trong **không gian cao chiều**, t-SNE sử dụng **phân phối Gaussian** để đo độ tương đồng giữa các điểm dữ liệu.
            - Trong **không gian thấp chiều**, t-SNE sử dụng **phân phối t-Student** để giảm tác động của **outliers**.
        2. **Tối ưu hóa bằng KL-Divergence**:
            - Điều chỉnh vị trí các điểm trong không gian nhúng sao cho **phân phối xác suất** giống nhất với không gian gốc.
            - Sử dụng **gradient descent** để cập nhật tọa độ các điểm.
    """)

    # Hiển thị công thức toán học
    st.header("📊 Công thức Toán học của t-SNE")

    st.markdown("**1️⃣ Xác suất trong không gian cao chiều:**")
    st.latex(r"""
    p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma^2)}
    """)
    st.write("""
    - \( \sigma \) là độ lệch chuẩn điều chỉnh mức độ "mở rộng" của Gaussian.
    - \( p_{j|i} \) là xác suất có điều kiện, nghĩa là mức độ điểm \( x_j \) là hàng xóm của \( x_i \).
    """)

    st.markdown("**2️⃣ Xác suất trong không gian thấp chiều:**")
    st.latex(r"""
    q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
    """)
    st.write("""
    - \( y_i \) là điểm dữ liệu sau khi chiếu xuống không gian thấp chiều.
    - Phân phối t-Student có đuôi dài hơn, giúp ngăn việc outliers ảnh hưởng quá mạnh đến vị trí các điểm.
    """)

    st.markdown("**3️⃣ Tối ưu hóa bằng KL-Divergence:**")
    st.latex(r"""
    KL(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
    """)
    st.write("""
    - Mục tiêu của t-SNE là giảm thiểu KL-Divergence bằng cách sử dụng **gradient descent** để tối ưu hóa vị trí các điểm.
    """)

    st.success("✅ t-SNE giúp trực quan hóa dữ liệu phức tạp một cách hiệu quả!") 

    # Tiêu đề ứng dụng
    st.title("📉 Minh họa thu gọn chiều bằng t-SNE")

    # **Bước 1: Chọn tham số cho dữ liệu**
    num_samples = st.slider("Số điểm dữ liệu", 100, 1000, 300, step=50, key="num_samples", help="Số lượng điểm dữ liệu sẽ được tạo ra cho việc giảm chiều. Tăng số lượng điểm có thể làm t-SNE chạy lâu hơn.")
    num_features = 3  # Hiển thị 3D ban đầu
    num_clusters = st.slider("Số cụm", 2, 5, 3, key="num_clusters", help="Số lượng cụm dữ liệu trong không gian ban đầu. Mỗi cụm đại diện cho một nhóm dữ liệu khác nhau.")
    perplexity = st.slider("Perplexity", 5, 50, 30, key="perplexity", help="Tham số quyết định cách phân bố điểm trong không gian t-SNE. Giá trị thấp có thể làm mất cấu trúc dữ liệu, trong khi giá trị cao có thể làm mờ đi các cụm.")

    # **Bước 2: Thêm nút Reset để làm mới giao diện**
    if st.button("🔄 Reset", key="reset_button", help="Nhấn để làm mới toàn bộ giao diện và thiết lập lại các tham số về giá trị mặc định."):
        st.rerun()

    # **Bước 3: Nhấn nút để thực hiện thuật toán**
    if st.button("📊 Thực hiện", key="process_button", help="Nhấn để tạo dữ liệu ngẫu nhiên và áp dụng t-SNE để giảm chiều xuống 2D, giúp trực quan hóa dữ liệu dễ dàng hơn."):
        st.write("### 🔹 Tạo dữ liệu giả lập")
        # Tạo dữ liệu ngẫu nhiên với số cụm và số chiều đã chọn
        X, y = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, random_state=42)
        st.write(f"✅ Đã tạo dữ liệu với {num_samples} điểm, {num_features} chiều và {num_clusters} cụm.")
        
        # **Hiển thị dữ liệu ban đầu (3D)**
        st.write("### 🔹 Dữ liệu ban đầu (3D)")
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.set_title("Dữ liệu ban đầu (3D)")
        
        # **Giảm chiều bằng t-SNE (2D)**
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        ax2 = fig.add_subplot(122)
        scatter = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
        ax2.set_xlabel("t-SNE Component 1")
        ax2.set_ylabel("t-SNE Component 2")
        ax2.set_title("Dữ liệu sau t-SNE (2D)")
        plt.colorbar(scatter, label='Cluster Label')
        
        st.pyplot(fig)


def data(): 
    st.title("📚 Tập Dữ Liệu MNIST")
    
    st.markdown("""
    Tập dữ liệu **MNIST (Modified National Institute of Standards and Technology)** là một trong những bộ dữ liệu nổi bật và phổ biến nhất trong lĩnh vực học máy và nhận dạng hình ảnh. Đây là tập dữ liệu bao gồm các hình ảnh của các chữ số viết tay từ 0 đến 9, được thu thập để thử nghiệm các thuật toán phân loại và nhận dạng mẫu.
    
    ![Mnist-dataset](https://datasets.activeloop.ai/wp-content/uploads/2019/12/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp)
                               

    ## 1. Tổng Quan về MNIST:
    MNIST gồm hai phần chính:
    
    - **Dữ liệu huấn luyện (Training Set)**: Gồm 60.000 hình ảnh.
    - **Dữ liệu kiểm tra (Test Set)**: Gồm 10.000 hình ảnh.
    
    Mỗi hình ảnh trong bộ dữ liệu có kích thước là 28x28 pixel và biểu diễn một trong 10 chữ số (0 đến 9). Dữ liệu đã được chuẩn hóa, với các hình ảnh được căn chỉnh và có nền trắng, giúp việc xử lý trở nên đơn giản hơn.
    
    ## 2. Mục Tiêu Sử Dụng Tập Dữ Liệu MNIST:
    MNIST chủ yếu được sử dụng để huấn luyện và kiểm tra các thuật toán phân loại. Các mục tiêu chính khi làm việc với MNIST bao gồm:
    
    - **Phân loại chữ số viết tay**: Dự đoán chữ số tương ứng với mỗi hình ảnh.
    - **Kiểm thử mô hình học máy**: Được sử dụng để kiểm tra hiệu quả của các mô hình học máy, từ các thuật toán cổ điển như K-Nearest Neighbors (KNN), Support Vector Machines (SVM) đến các mô hình học sâu như mạng nơ-ron tích chập (CNN).
    - **Tiền xử lý và học máy cơ bản**: Đây là một bộ dữ liệu tuyệt vời để hiểu rõ các quy trình tiền xử lý dữ liệu và cách thức hoạt động của các mô hình phân loại.
    
    ## 3. Cấu Trúc Dữ Liệu MNIST:
    Mỗi hình ảnh trong bộ dữ liệu MNIST có kích thước 28x28 pixel, tức là mỗi hình ảnh sẽ có 784 giá trị số nguyên, tương ứng với độ sáng của từng pixel. Tất cả các giá trị này sẽ được sử dụng để huấn luyện mô hình. Dữ liệu này có thể được sử dụng cho các tác vụ như:
    
    - **Phân loại hình ảnh**: Các mô hình học máy có thể học cách phân loại các hình ảnh thành các nhóm chữ số từ 0 đến 9.
    - **Tiền xử lý hình ảnh**: Việc chuẩn hóa dữ liệu và áp dụng các kỹ thuật tiền xử lý giúp cải thiện hiệu quả của mô hình.
    
    ## 4. Ứng Dụng Của Tập Dữ Liệu MNIST:
    - **Nhận dạng chữ viết tay**: Đây là ứng dụng phổ biến nhất của MNIST.
    - **Học sâu và phân loại hình ảnh**: Các mô hình học sâu, đặc biệt là mạng nơ-ron tích chập, được huấn luyện với bộ dữ liệu này để phân loại chữ số.
    """)


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
                
                # Hiển thị 5 dòng dữ liệu đầu tiên, chuyển đổi mỗi ảnh thành vector 1 chiều
                st.write("📊 **Dữ liệu mẫu:**")
                X_reshaped = X[:5].reshape(5, -1)
                st.write(pd.DataFrame(X_reshaped))

                st.success("✅ Dữ liệu MNIST đã được tải thành công!")
                st.session_state.data = (X, y)

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
                    image_array = np.array(image).reshape(1, -1)
                    st.session_state.data = image_array

    # Kiểm tra nếu dữ liệu đã được tải
    if st.session_state.data is not None:
        st.markdown("#### ✅ Dữ liệu đã sẵn sàng!")
        
        if isinstance(st.session_state.data, tuple):
            X, y = st.session_state.data
            st.markdown("##### 🔄 Tiến hành tiền xử lý dữ liệu MNIST")

            preprocess_option = st.selectbox("Chọn phương pháp tiền xử lý dữ liệu:", 
                                            ["Chuẩn hóa dữ liệu (Standardization)", "Giảm chiều (PCA)", "Giảm chiều (t-SNE)", "Không tiền xử lý"], key="preprocess_mnist")
            
            X_reshaped = X.reshape(X.shape[0], -1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_reshaped)
            
            if preprocess_option == "Chuẩn hóa dữ liệu (Standardization)":
                st.write("📊 **Dữ liệu sau khi chuẩn hóa**:")
                st.write(pd.DataFrame(X_scaled).head())
                st.session_state.processed_data = X_scaled

            elif preprocess_option == "Giảm chiều (PCA)":
                n_components = st.slider("Chọn số chiều PCA:", min_value=10, max_value=100, value=50, step=10)
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                st.write(f"📊 **Dữ liệu sau khi giảm chiều với PCA ({n_components} chiều):**")
                st.write(pd.DataFrame(X_pca).head())
                st.session_state.processed_data = X_pca

            elif preprocess_option == "Giảm chiều (t-SNE)":
                n_components = st.radio("Chọn số chiều t-SNE:", [2, 3], key="tsne_components")
                perplexity = st.slider("Chọn perplexity:", min_value=5, max_value=50, value=30, step=5)
                tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
                X_tsne = tsne.fit_transform(X_scaled)
                st.write(f"📊 **Dữ liệu sau khi giảm chiều với t-SNE ({n_components} chiều):**")
                st.write(pd.DataFrame(X_tsne).head())
                st.session_state.processed_data = X_tsne

            else:
                st.write("📊 **Dữ liệu không có tiền xử lý**.")
                st.session_state.processed_data = X_reshaped

        elif isinstance(st.session_state.data, np.ndarray):
            st.markdown("#### 👁️ Tiến hành tiền xử lý ảnh")
            preprocess_option_image = st.selectbox("Chọn phương pháp tiền xử lý ảnh:",
                                                   ["Chuẩn hóa ảnh", "Không tiền xử lý"], key="preprocess_image")

            if preprocess_option_image == "Chuẩn hóa ảnh":
                image_scaled = st.session_state.data / 255.0
                st.write("📊 **Ảnh sau khi chuẩn hóa**:")
                st.image(image_scaled.reshape(28, 28), caption="Ảnh sau khi chuẩn hóa", use_column_width=True)
                st.session_state.processed_data = image_scaled
            else:
                st.write("📊 **Ảnh không có tiền xử lý**.")
                st.session_state.processed_data = st.session_state.data
    else:
        st.warning("🔸 Vui lòng tải dữ liệu trước khi tiếp tục làm việc.")

    st.markdown("""
    🔹 **Lưu ý:**
    - Ứng dụng chỉ sử dụng dữ liệu ảnh dạng **28x28 pixel (grayscale)**.
    - Dữ liệu phải có cột **'label'** chứa nhãn (số từ 0 đến 9) khi tải từ OpenML.
    - Nếu dữ liệu của bạn không đúng định dạng, vui lòng sử dụng dữ liệu MNIST từ OpenML.
    """)


def chia_du_lieu():
    st.title("📌 Chia dữ liệu Train/Test")

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
        st.table(summary_df)


def train_model():
    st.title("📉 Giảm chiều dữ liệu MNIST với PCA & t-SNE")

    # Khởi tạo session state nếu chưa có
    if "run_name" not in st.session_state:
        st.session_state["run_name"] = "default_run"
    if "mlflow_url" not in st.session_state:
        st.session_state["mlflow_url"] = ""

    # Load dữ liệu
    Xmt = np.load("X.npy")
    ymt = np.load("y.npy")
    X = Xmt.reshape(Xmt.shape[0], -1) 
    y = ymt.reshape(-1) 

    # Tùy chọn thuật toán
    method = st.radio("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"], help="Phương pháp giảm chiều dữ liệu: PCA (Phân tích thành phần chính) hoặc t-SNE (Nhúng tạp chí Stochastic).")
    n_components = st.slider("Số chiều giảm xuống", 2, 10, 2, help="Số chiều đầu ra của dữ liệu sau khi giảm chiều (\(n\)-components).")

    # Nếu chọn t-SNE, thêm tùy chọn Perplexity
    perplexity = 30
    if method == "t-SNE":
        perplexity = st.slider("Chọn Perplexity", 5, 50, 30, step=5, help="Perplexity là tham số cân bằng giữa cục bộ và toàn cục trong t-SNE. Giá trị lớn hơn sẽ ưu tiên cấu trúc toàn cục hơn.")

    # Thanh trượt chọn số lượng mẫu sử dụng từ MNIST
    num_samples = st.slider("Chọn số lượng mẫu MNIST sử dụng:", min_value=1000, max_value=60000, value=5000, step=1000, help="Số lượng mẫu dữ liệu từ tập MNIST sẽ được sử dụng để huấn luyện.")

    # Giới hạn số mẫu để tăng tốc
    X_subset, y_subset = X[:num_samples], y[:num_samples]

    # Hàm giả định để thiết lập MLflow
    def mlflow_input():
        pass

    mlflow_input()

    if st.button("🚀 Chạy giảm chiều"):
        with st.spinner("Đang xử lý..."):
            mlflow.start_run(run_name=st.session_state["run_name"])
            mlflow.log_param("method", method)
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("num_samples", num_samples)
            mlflow.log_param("original_dim", X.shape[1])

            if method == "t-SNE":
                mlflow.log_param("perplexity", perplexity)
                reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            else:
                reducer = PCA(n_components=n_components)

            start_time = time.time()
            X_reduced = reducer.fit_transform(X_subset)
            elapsed_time = time.time() - start_time
            mlflow.log_metric("elapsed_time", elapsed_time)

            if method == "PCA":
                explained_variance = np.sum(reducer.explained_variance_ratio_)
                mlflow.log_metric("explained_variance_ratio", explained_variance)
            elif method == "t-SNE" and hasattr(reducer, "kl_divergence_"):
                mlflow.log_metric("KL_divergence", reducer.kl_divergence_)

            # Hiển thị kết quả nếu n_components <= 3
            if n_components == 2:
                fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=y_subset.astype(str),
                                 title=f"{method} giảm chiều xuống {n_components}D",
                                 labels={'x': "Thành phần 1", 'y': "Thành phần 2"})
                st.plotly_chart(fig)
            elif n_components == 3:
                fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                     color=y_subset.astype(str),
                                     title=f"{method} giảm chiều xuống {n_components}D",
                                     labels={'x': "Thành phần 1", 'y': "Thành phần 2", 'z': "Thành phần 3"})
                st.plotly_chart(fig)
            else:
                st.warning(f"Số chiều = {n_components} lớn hơn 3, không thể hiển thị trực quan!")

            # Lưu kết quả vào MLflow
            os.makedirs("logs", exist_ok=True)
            np.save(f"logs/{method}_X_reduced.npy", X_reduced)
            mlflow.log_artifact(f"logs/{method}_X_reduced.npy")

            mlflow.end_run()
            st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")

            if st.session_state["mlflow_url"]:
                st.markdown(f"### 🔗 [Truy cập MLflow]({st.session_state['mlflow_url']})")
            else:
                st.warning("⚠️ Chưa có đường link MLflow!")

            st.success("Hoàn thành!")




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
    mlflow.set_experiment("PCA & t-SNE")   

    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"


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


def PCA_T_sne():
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


    st.title("🖊️ MNIST Classification App")

    # Ensure the tab names are properly separated
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📘 Lý thuyết PCA", 
    "📘 Lý thuyết T-sne", 
    "📘 Review database", 
    "📥 Tải dữ liệu", 
    "🔀 Chia dữ liệu",
    "Thông tin thu gọn chiều"
    ])

    with tab1: 
        ly_thuyet_PCA() 

    with tab2:
        ly_thuyet_tSne()

    with tab3: 
        data()    
    
    with tab4: 
        up_load_db()
    
    with tab5: 
        chia_du_lieu()
        train_model()
    with tab6: 
        display_mlflow_experiments()    

def run(): 
    PCA_T_sne()        

if __name__ == "__main__":
    run()
