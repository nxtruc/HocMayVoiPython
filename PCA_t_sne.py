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
import datetime
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
    st.header("📖 Lý thuyết về PCA")
    st.markdown(" ### 1️⃣ PCA là gì?")
    st.write("PCA (Principal Component Analysis) là một kỹ thuật phân tích dữ liệu dùng để giảm số chiều (dimension reduction) trong dữ liệu, giúp tìm ra các yếu tố quan trọng nhất (các thành phần chính) trong một tập dữ liệu có nhiều chiều. Mục tiêu của PCA là giảm số lượng các biến đầu vào trong khi vẫn giữ lại phần lớn thông tin trong dữ liệu.")
    image_url = "https://machinelearningcoban.com/assets/27_pca/pca_var0.png"
    article_url = "https://machinelearningcoban.com/2017/06/15/pca/"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="200">
            </a>
            <p style="font-size: 14px; color: gray;"></p>
        </div>
        """,
        unsafe_allow_html=True
    ) 
    st.markdown(" ### 2️⃣ ý tưởng") 
    st.write("""1. **Loại bỏ thành phần với phương sai nhỏ**: Trong PCA, các thành phần có phương sai nhỏ biểu thị chiều dữ liệu mà sự thay đổi không đáng kể. Do đó, ta có thể loại bỏ các thành phần này để giảm chiều dữ liệu mà không mất thông tin quan trọng.""")
    image_url = "https://machinelearningcoban.com/assets/27_pca/pca_diagvar.png"
    article_url = "https://machinelearningcoban.com/2017/06/15/pca/"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;"></p>
        </div>
        """,
        unsafe_allow_html=True
    ) 
    st.write("""2. **Xoay dữ liệu để tăng phương sai**: PCA "xoay" dữ liệu sao cho các thành phần chính có phương sai lớn nhất. Điều này giúp chọn ra các chiều dữ liệu quan trọng nhất, giảm nhiễu và tối ưu hóa không gian dữ liệu.""")
    image_path = "rotateTheImagePCA.jpeg"
    # Hiển thị ảnh sử dụng Streamlit
    st.image(image_path)

    # URL bài viết
    article_url = "https://setosa.io/ev/principal-component-analysis/"

    # HTML để hiển thị ảnh và liên kết bài viết
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_path}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;">nguồn ảnh</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(" ### 3️⃣ Thuật toán PCA")
    st.markdown(" ##### 1.Tìm điểm trung tâm (Mean Vector)")
    st.markdown("""
    Trước tiên, tính giá trị trung bình của từng đặc trưng (feature) trong tập dữ liệu.
    Vector trung bình này giúp xác định "trung tâm" của dữ liệu. Công thức tính trung bình:
    """)
    st.latex(r"""
    \mu = \frac{1}{n} \sum_{i=1}^{n} x_i
    """)
    st.markdown("""
    Trong đó:
    - \(n\) là số lượng mẫu dữ liệu.
    - \(x_i\) là từng điểm dữ liệu.
    """)
    image_path = "img1.png"
    # Hiển thị ảnh sử dụng Streamlit
    st.image(image_path)

    # URL bài viết
    article_url = "https://machinelearningcoban.com/2017/06/15/pca/"

    # HTML để hiển thị ảnh và liên kết bài viết
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_path}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;">nguồn ảnh</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(" ##### 2.Dịch chuyển dữ liệu về gốc tọa độ")
    st.markdown("""
    Để đảm bảo phân tích chính xác hơn, ta dịch chuyển dữ liệu sao cho trung tâm của nó nằm tại gốc tọa độ bằng cách trừ đi vector trung bình:
    """)
    st.latex(r"""
    X_{norm} = X - \mu
    """)
    st.markdown("Khi đó, dữ liệu sẽ có giá trị trung bình bằng 0.")
    mage_path = "img2.png"
    # Hiển thị ảnh sử dụng Streamlit
    st.image(image_path)

    # URL bài viết
    article_url = "https://machinelearningcoban.com/2017/06/15/pca/"

    # HTML để hiển thị ảnh và liên kết bài viết
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_path}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;">nguồn ảnh</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(" ##### 3.Tính ma trận hiệp phương sai (Covariance Matrix)")
    st.markdown("""
    Ma trận hiệp phương sai giúp đo lường mức độ biến thiên giữa các đặc trưng:
    """)
    st.latex(r"""
    C = \frac{1}{n} X_{norm}^T X_{norm}
    """)
    st.markdown("""
    Ý nghĩa:
    - Nếu phần tử \( C_{ij} \) có giá trị lớn $\rightarrow$ Hai đặc trưng \(i\) và \(j\) có mối tương quan mạnh.
    - Nếu \( C_{ij} \) gần 0 $\rightarrow$ Hai đặc trưng không liên quan nhiều.
    """)

    st.markdown(" ##### 4.Tìm các hướng quan trọng nhất (Eigenvalues & Eigenvectors)")
    st.markdown("""
    Tính trị riêng (eigenvalues) và vector riêng (eigenvectors) từ ma trận hiệp phương sai:
    """)
    st.latex(r"""
    C v = \lambda v
    """)
    st.markdown("""
    Trong đó:
    - \(v\) là vector riêng (eigenvector) - đại diện cho các hướng chính của dữ liệu.
    - \(\lambda\) là trị riêng (eigenvalue) - thể hiện độ quan trọng của từng hướng.
    Vector riêng có trị riêng lớn hơn sẽ mang nhiều thông tin quan trọng hơn.
    """)

    st.markdown(" ##### 5.Chọn số chiều mới và tạo không gian con")
    st.markdown("""
    Chọn \(K\) vector riêng tương ứng với \(K\) trị riêng lớn nhất để tạo thành ma trận \(U_K\):
    """)
    st.latex(r"""
    U_K = [v_1, v_2, \dots, v_K]
    """)
    st.markdown("Các vector này tạo thành không gian trực giao và giúp biểu diễn dữ liệu tối ưu trong không gian mới.")
    mage_path = "img4.png"
    # Hiển thị ảnh sử dụng Streamlit
    st.image(image_path)

    # URL bài viết
    article_url = "https://machinelearningcoban.com/2017/06/15/pca/"

    # HTML để hiển thị ảnh và liên kết bài viết
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_path}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;">nguồn ảnh</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(" ##### 6.Chiếu dữ liệu vào không gian mới")
    st.markdown("""
    Biểu diễn dữ liệu trong hệ trục mới bằng cách nhân dữ liệu chuẩn hóa với ma trận \(U_K\):
    """)
    st.latex(r"""
    X_{new} = X_{norm} U_K
    """)
    st.markdown("Dữ liệu mới \(X_{new}\) có số chiều ít hơn nhưng vẫn giữ lại thông tin quan trọng.")

    mage_path = "img5.png"
    # Hiển thị ảnh sử dụng Streamlit
    st.image(image_path)

    # URL bài viết
    article_url = "https://machinelearningcoban.com/2017/06/15/pca/"

    # HTML để hiển thị ảnh và liên kết bài viết
    st.markdown(
    f"""
    <div style="text-align: center;">
        <a href="{article_url}" target="_blank">
            <img src="{image_path}" width="300">
        </a>
        <p style="font-size: 14px; color: gray;">Nguồn ảnh</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown(" ##### 7.Dữ liệu mới")
    st.markdown("""
    Dữ liệu mới \(X_{new}\) là tọa độ của các điểm trong không gian mới, với các thành phần chính làm trục mới.
    """)
    st.markdown("## 4️⃣ Minh họa thu gọn chiều bằng PCA")

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


def train_model():
    st.title("📉 Giảm chiều dữ liệu MNIST với PCA & t-SNE")
    
    mlflow_input()

    # Khởi tạo session state nếu chưa có
    if "run_name" not in st.session_state:
        st.session_state["run_name"] = "default_run"
    if "mlflow_url" not in st.session_state:
        st.session_state["mlflow_url"] = ""

    # Nhập tên thí nghiệm
    st.session_state["run_name"] = st.text_input("🔖 Đặt tên thí nghiệm:", value=st.session_state["run_name"])

    # Load dữ liệu
    Xmt = np.load("X.npy")
    ymt = np.load("y.npy")
    X = Xmt.reshape(Xmt.shape[0], -1) 
    y = ymt.reshape(-1) 

    # Tùy chọn thuật toán
    method = st.radio("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"])
    n_components = st.slider("Chọn số chiều giảm xuống", 2, 50, 2)

    # Chọn cách trực quan hóa
    visualization_dim = st.radio("Chọn cách trực quan hóa", ["2D", "3D"])
    
    # Nếu chọn t-SNE, thêm tùy chọn Perplexity
    perplexity = 30
    if method == "t-SNE":
        perplexity = st.slider("Chọn Perplexity", 5, 50, 30, step=5)

    # Thanh trượt chọn số lượng mẫu sử dụng từ MNIST
    num_samples = st.slider("Chọn số lượng mẫu MNIST sử dụng:", 1000, 60000, 5000, step=1000)

    # Giới hạn số mẫu để tăng tốc
    X_subset, y_subset = X[:num_samples], y[:num_samples]

    if st.button("🚀 Chạy giảm chiều"):
        with st.spinner("Đang xử lý..."):
            progress_bar = st.progress(0)  # Khởi tạo thanh tiến trình
            status_text = st.empty()  # Ô hiển thị phần trăm tiến trình

            mlflow.start_run(run_name=st.session_state["run_name"])
            mlflow.log_param("experiment_name", st.session_state["run_name"])
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

            # Huấn luyện mô hình và cập nhật tiến trình
            for i in range(1, 101):
                time.sleep(0.02)  # Mô phỏng thời gian xử lý
                progress_bar.progress(i)  # Cập nhật tiến trình
                status_text.text(f"🔄 Tiến độ: {i}%")  # Hiển thị phần trăm

                if i == 50:  # Bắt đầu huấn luyện khi tiến trình đạt 50%
                    X_reduced = reducer.fit_transform(X_subset)

            elapsed_time = time.time() - start_time
            mlflow.log_metric("elapsed_time", elapsed_time)

            if method == "PCA":
                explained_variance = np.sum(reducer.explained_variance_ratio_)
                mlflow.log_metric("explained_variance_ratio", explained_variance)
            elif method == "t-SNE" and hasattr(reducer, "kl_divergence_"):
                mlflow.log_metric("KL_divergence", reducer.kl_divergence_)

            # Hiển thị kết quả
            if visualization_dim == "2D" and n_components >= 2:
                fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=y_subset.astype(str),
                                 title=f"{method} giảm chiều xuống {n_components}D")
                st.plotly_chart(fig)
            elif visualization_dim == "3D" and n_components >= 3:
                fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                     color=y_subset.astype(str),
                                     title=f"{method} giảm chiều xuống {n_components}D")
                st.plotly_chart(fig)
            else:
                st.warning(f"Không thể hiển thị trực quan với {visualization_dim} khi số chiều = {n_components}!")

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

            progress_bar.empty()  # Xóa thanh tiến trình sau khi hoàn tất
            status_text.empty()  # Xóa hiển thị phần trăm tiến trình



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



def format_time_relative(timestamp_ms):
    """Chuyển timestamp milliseconds thành thời gian dễ đọc."""
    if timestamp_ms is None:
        return "N/A"
    dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def display_mlflow_experiments():
    """Hiển thị danh sách Runs trong MLflow với thanh trạng thái tiến trình."""
    st.title("📊 MLflow Experiment Viewer")

    # Lấy danh sách thí nghiệm
    experiment_name = "PCA & t-SNE"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # --- 🏃‍♂️ Lấy danh sách Runs với thanh trạng thái ---
    st.write("### 🔄 Đang tải danh sách Runs...")
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    total_runs = len(runs)
    run_info = []
    
    progress_bar = st.progress(0)  # Thanh tiến trình

    for i, (_, run) in enumerate(runs.iterrows()):
        run_id = run["run_id"]
        run_data = mlflow.get_run(run_id)
        run_tags = run_data.data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")  # Tên Run
        created_time = format_time_relative(run_data.info.start_time)
        duration = (run_data.info.end_time - run_data.info.start_time) / 1000 if run_data.info.end_time else "Đang chạy"
        source = run_tags.get("mlflow.source.name", "Unknown")

        run_info.append({
            "Run Name": run_name,
            "Run ID": run_id,
            "Created": created_time,
            "Duration (s)": duration if isinstance(duration, str) else f"{duration:.1f}s",
            "Source": source
        })

        # Cập nhật thanh tiến trình
        progress_bar.progress(int((i + 1) / total_runs * 100))

    progress_bar.empty()  # Xóa thanh tiến trình khi hoàn thành

    # Sắp xếp và hiển thị bảng danh sách Runs
    run_info_df = pd.DataFrame(run_info).sort_values(by="Created", ascending=False)
    st.write("### 🏃‍♂️ Danh sách Runs:")
    st.dataframe(run_info_df, use_container_width=True)

    # Chọn Run từ dropdown
    run_names = run_info_df["Run Name"].tolist()
    selected_run_name = st.selectbox("🔍 Chọn một Run để xem chi tiết:", run_names)

    # Lấy Run ID tương ứng
    selected_run_id = run_info_df.loc[run_info_df["Run Name"] == selected_run_name, "Run ID"].values[0]
    selected_run = mlflow.get_run(selected_run_id)

    # --- 📝 ĐỔI TÊN RUN ---
    st.write("### ✏️ Đổi tên Run")
    new_run_name = st.text_input("Nhập tên mới:", selected_run_name)
    if st.button("💾 Lưu tên mới"):
        try:
            mlflow.set_tag(selected_run_id, "mlflow.runName", new_run_name)
            st.success(f"✅ Đã đổi tên thành **{new_run_name}**. Hãy tải lại trang để thấy thay đổi!")
        except Exception as e:
            st.error(f"❌ Lỗi khi đổi tên: {e}")

    # --- 🗑️ XÓA RUN ---
    st.write("### ❌ Xóa Run")
    if st.button("🗑️ Xóa Run này"):
        try:
            mlflow.delete_run(selected_run_id)
            st.success(f"✅ Đã xóa run **{selected_run_name}**! Hãy tải lại trang để cập nhật danh sách.")
        except Exception as e:
            st.error(f"❌ Lỗi khi xóa run: {e}")

    # --- HIỂN THỊ CHI TIẾT RUN ---
    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")

        start_time_ms = selected_run.info.start_time
        start_time = datetime.datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
        st.write(f"**Thời gian chạy:** {start_time}")

        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        # Hiển thị model artifact (nếu có)
        model_artifact_path = f"{st.session_state['mlflow_url']}/{selected_experiment.experiment_id}/{selected_run_id}/artifacts/model"
        st.write("### 📂 Model Artifact:")
        st.write(f"📥 [Tải mô hình]({model_artifact_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")

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


    st.title("🖊️ MNIST PCA & t-SNE App")

    # Ensure the tab names are properly separated
    tab1, tab2, tab3, tab4, tab5= st.tabs([
    "📘 Lý thuyết PCA", 
    "📘 Lý thuyết T-sne", 
    "📘 Review database",  
    "🔀 Giảm chiều",
    " 🚀 Thông tin thu gọn chiều"
    ])

    with tab1: 
        ly_thuyet_PCA() 

    with tab2:
        ly_thuyet_tSne()

    with tab3: 
        data()    

    with tab4:
        train_model()
    with tab5: 
        display_mlflow_experiments()    

def run(): 
    PCA_T_sne()        

if __name__ == "__main__":
    run()
