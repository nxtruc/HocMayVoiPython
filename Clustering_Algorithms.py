import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import openml
import os
import mlflow
import time
import shutil
import humanize
from datetime import datetime
from scipy.stats import mode
from scipy import stats
from mlflow.tracking import MlflowClient
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# Tải dữ liệu MNIST từ OpenML


import streamlit as st

def ly_thuyet_kmeans():
    st.header("📖 Lý thuyết về K-Means")
    st.markdown(" ### 1️⃣ K-Means là gì?")
    st.write("K-means là một thuật toán **học không giám sát** dùng để phân cụm dữ liệu thành k cụm dựa trên khoảng cách Euclid.")
    st.markdown(" ##### 🎯 Mục tiêu của thuật toán K-Means")

    st.write("""
    Thuật toán **K-Means** có mục tiêu chính là **tìm các cụm tối ưu** trong tập dữ liệu bằng cách **tối thiểu hóa tổng bình phương khoảng cách** từ các điểm dữ liệu đến tâm cụm của chúng.
    """)

    st.markdown(" ##### Hàm mục tiêu (Objective Function)")
    st.write("K-Means cố gắng tối thiểu hóa tổng phương sai trong cụm, được biểu diễn bằng công thức:")

    st.latex(r"""
    J = \sum_{k=1}^{K} \sum_{x_i \in C_k} || x_i - \mu_k ||^2
    """)

    st.write("""
    Trong đó:
    - \\( K \\): Số lượng cụm.
    - \\( C_k \\): Tập hợp các điểm dữ liệu thuộc cụm thứ \\( k \\).
    - \\( x_i \\): Điểm dữ liệu trong cụm \\( C_k \\).
    - \\( \mu_k \\): Tâm cụm của \\( C_k \\).
    - \\( || x_i - \mu_k ||^2 \\): Khoảng cách Euclidean bình phương giữa điểm \\( x_i \\) và tâm cụm \\( \mu_k \\).
    """)

    st.markdown(" ### 2️⃣ ý tưởng") 
    st.markdown(
    """
    - Chia tập dữ liệu thành 𝐾K cụm (clusters), với mỗi cụm có một tâm cụm (centroid).
    - Dữ liệu được gán vào cụm có tâm cụm gần nó nhất.
    - Cập nhật tâm cụm bằng cách tính trung bình các điểm thuộc cụm.
    - Lặp lại cho đến khi không có sự thay đổi đáng kể trong cụm.
    """    
    )
    image_url = "https://miro.medium.com/v2/resize:fit:720/format:webp/1*fz-rjYPPRlGEMdTI-RLbDg.png" 
    article_url = "https://meghachhabria10.medium.com/k-means-clustering-algorithm-its-use-cases-13dfc0020b23"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="500">
            </a>
            <p style="font-size: 14px; color: gray;"></p>
        </div>
        """,
        unsafe_allow_html=True
    ) 
    st.markdown(" ### 3️⃣ Thuật toán K-Means") 
    st.markdown(
    """
    - 1.Chọn số cụm 𝐾 (được xác định trước).
    - 2.Khởi tạo 𝐾 tâm cụm (chọn ngẫu nhiên hoặc theo K-Means++ để tốt hơn).
    - 3.Gán dữ liệu vào cụm: Mỗi điểm dữ liệu được gán vào cụm có tâm cụm gần nhất
    """
    )
    st.latex(r"""d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}""")
    st.markdown(
    """
    - 4.Cập nhật tâm cụm: Tính lại tâm cụm bằng cách lấy trung bình các điểm trong mỗi cụm.
    """
    )
    st.latex(r"""\mu_k = \frac{1}{N_k} \sum_{i=1}^{N_k} x_i""")
    st.markdown(
    """
    - 5.Lặp lại các bước 3 & 4 cho đến khi các tâm cụm không thay đổi nhiều nữa hoặc đạt đến số lần lặp tối đa.
    """
    )
    image_url = "https://machinelearningcoban.com/assets/kmeans/kmeans11.gif"
    article_url = "https://machinelearningcoban.com/2017/01/01/kmeans/"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="500">
            </a>
            <p style="font-size: 14px; color: gray;">Minh họa thuật toán K-Means</p>
        </div>
        """,
        unsafe_allow_html=True
    ) 
    st.markdown(" ###  4️⃣ Đánh giá thuật toán K-Means")
    st.markdown(" ##### 📌 Elbow Method")
    st.write("""
    - Tính tổng khoảng cách nội cụm WCSS (Within-Cluster Sum of Squares) cho các giá trị k khác nhau.
    - Điểm "khuỷu tay" (elbow point) là giá trị k tối ưu, tại đó việc tăng thêm cụm không làm giảm đáng kể WCSS.
    """)
    st.latex(r"""
    WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
    """)
    image_url = "https://miro.medium.com/v2/resize:fit:720/format:webp/0*aY163H0kOrBO46S-.png"
    article_url = "https://medium.com/@zalarushirajsinh07/the-elbow-method-finding-the-optimal-number-of-clusters-d297f5aeb189"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="500">
            </a>
            <p style="font-size: 14px; color: gray;"></p>
        </div>
        """,
        unsafe_allow_html=True
    ) 

    st.markdown(" ##### 📌 Silhouette Score")
    st.write("""
    - So sánh mức độ gần gũi giữa các điểm trong cụm với các điểm ở cụm khác.
    """)
    st.latex(r"""
    s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
    """)
    st.write("""
    - \\( a(i) \\): Khoảng cách trung bình từ điểm i đến các điểm trong cùng cụm.
    - \\( b(i) \\): Khoảng cách trung bình từ điểm i đến các điểm trong cụm gần nhất.
    """)
    image_url = "https://miro.medium.com/v2/resize:fit:720/format:webp/1*pw0s5xkiyVxD4f2XqrPmuQ.png"
    article_url = "https://medium.com/@ayse_nur_safak/the-silhouette-score-finding-the-optimal-number-of-clusters-using-k-means-clustering-9af3be119848"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="400">
            </a>
            <p style="font-size: 14px; color: gray;"></p>
        </div>
        """,
        unsafe_allow_html=True
    ) 
    st.markdown(" ##### 📌 Gap Statistic")
    st.write("""
    - So sánh hiệu quả phân cụm trên dữ liệu thực với dữ liệu ngẫu nhiên (không có cấu trúc).
    """)
    st.latex(r"""
    Gap(k) = \mathbb{E}[\log(W_k^{random})] - \log(W_k^{data})
    """)
    st.write("""
    - \\( W_k^{random} \\): WCSS trên random data.
    - \\( W_k^{data} \\): WCSS trên actual data.
    """)
    image_url = "https://media.geeksforgeeks.org/wp-content/uploads/20241227193645905364/Gap-Statistics-vs-Number-of-Clusters-.png"
    article_url = "https://www.geeksforgeeks.org/gap-statistics-for-optimal-number-of-cluster/"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="400">
            </a>
            <p style="font-size: 14px; color: gray;"></p>
        </div>
        """,
        unsafe_allow_html=True
    ) 

def ly_thuyet_dbscans():
   st.header("📖 Lý thuyết về DBSCAN")
   st.markdown(" ### 1️⃣ DBSCAN là gì?")
   st.markdown(
   """
   - DBSCAN là một thuật toán phân cụm dựa trên mật độ, được thiết kế để tìm các cụm dữ liệu có hình dạng bất kỳ và phát hiện các điểm nhiễu (noise).
   - Không yêu cầu biết trước số cụm.
   """
   )
   st.markdown(" ##### 🎯 Mục tiêu của thuật toán DBSCAN ")
   st.write("1. **Phát hiện cụm có hình dạng bất kỳ:** Không giống như K-Means (yêu cầu cụm có dạng hình cầu), DBSCAN có thể tìm ra các cụm có hình dạng bất kỳ, kể cả dạng phi tuyến tính.")
   st.write("2. **Không cần chỉ định số cụm trước:** Không giống K-Means, DBSCAN tự động tìm ra số lượng cụm dựa trên mật độ điểm dữ liệu mà không cần tham số 𝑘")
   st.write("3. **Xác định điểm nhiễu (outliers):** Các điểm không thuộc cụm nào được xác định là nhiễu, giúp làm sạch dữ liệu.")
   st.markdown(" ##### 🎯 Giải thích thuật toán DBSCAN: Epsilon, MinPts và Phân loại điểm ")
   st.write(" 1. Epsilon (ε)  Bán kính để xác định khu vực lân cận của một điểm.")
   st.write(" 2. MinPts Số lượng điểm tối thiểu cần thiết để một khu vực được coi là đủ mật độ.")
   st.markdown("###### 3. Loại điểm trong DBSCAN:")
   st.write("- **Core Point**: Điểm có ít nhất MinPts điểm khác nằm trong khoảng \\( \epsilon \\).")
   st.write("- **Border Point**: Điểm không phải là Core Point nhưng nằm trong vùng lân cận của một Core Point.")
   st.write("- **Noise**: Điểm không thuộc Core Point hoặc Border Point.")
   image_url = "https://media.datacamp.com/cms/google/ad_4nxczbbrn-drkfpsiiqf1zayyt5xnqiwgpz0qocpnt6au5mintqlk4r1mxlognyzyxxmewlx35vcn53cbwm6iun4hh5i-aokth6fyqhovv1dlill6myhah4hzizcpb-bmv-g8vbiwawgudq8_gkuhqb8yiwja.jpeg"
   article_url = "https://www.datacamp.com/tutorial/dbscan-clustering-algorithm"
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
   st.markdown(" ### 3️⃣ Thuật toán DBSCAN")
   st.write("1. Chọn một điểm chưa được thăm")
   st.write("2. Kiểm tra xem có ít nhất MinPts điểm trong vùng \( \\varepsilon \) của nó hay không:")
   st.write("- ✅ **Nếu có**: Điểm đó là **Core Point**, và một cụm mới bắt đầu.")
   st.write("- ❌ **Nếu không**: Điểm đó là **Noise** (nhiễu), nhưng sau này có thể trở thành **Border Point** nếu thuộc vùng lân cận của một **Core Point**.")
   st.write("3. Nếu điểm là Core Point, mở rộng cụm bằng cách tìm tất cả các điểm lân cận thỏa mãn điều kiện.")
   st.write("4. Lặp lại cho đến khi không còn điểm nào có thể được thêm vào cụm.")
   st.write("5. Chuyển sang điểm chưa được thăm tiếp theo và lặp lại quá trình.")
   image_url = "https://media.geeksforgeeks.org/wp-content/uploads/20190418023034/781ff66c-b380-4a78-af25-80507ed6ff26.jpeg"
   article_url = "https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/"
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
    st.header("📥 Tải Dữ Liệu")
    
    if "data" in st.session_state and st.session_state.data is not None:
        st.warning("🔸 **Dữ liệu đã được tải lên rồi!** Bạn có thể tiếp tục với các bước tiền xử lý và chia dữ liệu.")
    else:
        option = st.radio("Chọn nguồn dữ liệu:", ["Tải từ OpenML", "Upload dữ liệu"], key="data_source_radio")
        
        if "data" not in st.session_state:
            st.session_state.data = None
        
        if option == "Tải từ OpenML":
            st.markdown("#### 📂 Tải dữ liệu MNIST từ OpenML")
            if st.button("Tải dữ liệu MNIST", key="download_mnist_button"):
                with st.status("🔄 Đang tải dữ liệu MNIST từ OpenML...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    for percent_complete in range(0, 101, 20):
                        time.sleep(0.5)
                        progress_bar.progress(percent_complete)
                        status.update(label=f"🔄 Đang tải... ({percent_complete}%)")
                    
                    X = np.load("X.npy")
                    y = np.load("y.npy")
                    
                    status.update(label="✅ Tải dữ liệu thành công!", state="complete")
                    
                    st.session_state.data = (X, y)
        
        else:
            st.markdown("#### 📤 Upload dữ liệu của bạn")
            uploaded_file = st.file_uploader("Chọn một file ảnh", type=["png", "jpg", "jpeg"], key="file_upload")
            
            if uploaded_file is not None:
                with st.status("🔄 Đang xử lý ảnh...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    for percent_complete in range(0, 101, 25):
                        time.sleep(0.3)
                        progress_bar.progress(percent_complete)
                        status.update(label=f"🔄 Đang xử lý... ({percent_complete}%)")
                    
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
                    
                    if image.size != (28, 28):
                        status.update(label="❌ Ảnh không đúng kích thước 28x28 pixel.", state="error")
                    else:
                        status.update(label="✅ Ảnh hợp lệ!", state="complete")
                        image = image.convert('L')
                        image_array = np.array(image).reshape(1, -1)
                        st.session_state.data = image_array
    
    if st.session_state.data is not None:
        st.markdown("#### ✅ Dữ liệu đã sẵn sàng!")
        
        if isinstance(st.session_state.data, tuple):
            X, y = st.session_state.data
            st.markdown("##### 🔄 Tiến hành tiền xử lý dữ liệu MNIST")
            preprocess_option = st.selectbox("Chọn phương pháp tiền xử lý dữ liệu:", 
                                            ["Chuẩn hóa dữ liệu (Standardization)", "Giảm chiều (PCA)", "Không tiền xử lý"], 
                                            key="preprocess_mnist")
            if preprocess_option == "Chuẩn hóa dữ liệu (Standardization)":
                X_reshaped = X.reshape(X.shape[0], -1)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_reshaped)
                st.write("📊 **Dữ liệu sau khi chuẩn hóa**:")
                st.write(pd.DataFrame(X_scaled).head())
            elif preprocess_option == "Giảm chiều (PCA)":
                pca = PCA(n_components=50)
                X_pca = pca.fit_transform(X.reshape(X.shape[0], -1))
                st.write("📊 **Dữ liệu sau khi giảm chiều (PCA)**:")
                st.write(pd.DataFrame(X_pca).head())
            else:
                st.write("📊 **Dữ liệu không có tiền xử lý**.")
        
        elif isinstance(st.session_state.data, np.ndarray):
            st.markdown("#### 👁️ Tiến hành tiền xử lý ảnh")
            preprocess_option_image = st.selectbox("Chọn phương pháp tiền xử lý ảnh:",
                                                   ["Chuẩn hóa ảnh", "Không tiền xử lý"], 
                                                   key="preprocess_image")
            if preprocess_option_image == "Chuẩn hóa ảnh":
                image_scaled = st.session_state.data / 255.0
                st.write("📊 **Ảnh sau khi chuẩn hóa**:")
                st.image(image_scaled.reshape(28, 28), caption="Ảnh sau khi chuẩn hóa", use_column_width=True)
            else:
                st.write("📊 **Ảnh không có tiền xử lý**.")
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

    # Đọc dữ liệu
    Xmt = np.load("X.npy")
    ymt = np.load("y.npy")
    X = Xmt.reshape(Xmt.shape[0], -1)  # Giữ nguyên định dạng dữ liệu
    y = ymt.reshape(-1)  

    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:", min_value=1000, max_value=total_samples, value=10000)

    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("Chọn tỷ lệ test:", min_value=0.1, max_value=0.5, value=0.2)

    if st.button("✅ Xác nhận & Lưu"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Cập nhật tiến trình
        progress_stages = [(10, "🔄 Đang chọn số lượng ảnh..."),
                           (50, "🔄 Đang chia dữ liệu Train/Test..."),
                           (80, "🔄 Đang lưu dữ liệu vào session..."),
                           (100, "✅ Hoàn tất!")]

        for progress, message in progress_stages:
            progress_bar.progress(progress)
            status_text.text(f"{message} ({progress}%)")
            time.sleep(0.5)  # Tạo độ trễ để hiển thị tiến trình rõ ràng hơn

        X_selected, y_selected = X[:num_samples], y[:num_samples]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=test_size, random_state=42)

        # Lưu vào session_state để sử dụng sau
        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.success(f"🔹 Dữ liệu đã được chia: Train ({len(X_train)}), Test ({len(X_test)})")

    if "X_train" in st.session_state:
        st.write("📌 Dữ liệu train/test đã sẵn sàng để sử dụng!")

###Thiet lap dagshub
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
    mlflow.set_experiment("Clustering Algorithms")   

    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"

def train():
    st.header("⚙️ Chọn mô hình & Huấn luyện")

    if "X_train" not in st.session_state:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi train!")
        return

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]
    X_train_norm = (X_train / 255.0).reshape(X_train.shape[0], -1)

    model_choice = st.selectbox("Chọn mô hình:", ["K-Means", "DBSCAN"])
    
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run").strip()

    if model_choice == "K-Means":
        st.markdown("🔹 **K-Means**")
        n_clusters = st.slider("🔢 Chọn số cụm (K):", 2, 20, 10)
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_norm)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    elif model_choice == "DBSCAN":
        st.markdown("🛠️ **DBSCAN**")
        eps = st.slider("📏 Bán kính lân cận (eps):", 0.1, 10.0, 0.5)
        min_samples = st.slider("👥 Số điểm tối thiểu trong cụm:", 2, 20, 5)
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_norm)
        model = DBSCAN(eps=eps, min_samples=min_samples)

    mlflow_input()
    if st.button("🚀 Huấn luyện mô hình"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        with mlflow.start_run(run_name=run_name):
            total_delay = 10  # Tổng thời gian delay thêm
            steps = 10  # Chia thành 10 bước
            step_delay = total_delay / steps

            for percent_complete in range(0, 101, 10):  
                time.sleep(step_delay)  
                progress_bar.progress(percent_complete)
                status_text.text(f"🔄 Huấn luyện: {percent_complete}%")

            model.fit(X_train_pca)
            progress_bar.progress(100)
            status_text.text("✅ Huấn luyện hoàn tất!")
            st.success(f"✅ Huấn luyện thành công! (Run: `{run_name}`)")

            labels = model.labels_

            if model_choice == "K-Means":
                label_mapping = {}
                for i in range(n_clusters):
                    mask = labels == i
                    if np.sum(mask) > 0:
                        most_common_label = stats.mode(y_train[mask], keepdims=True).mode[0]
                        label_mapping[i] = most_common_label

                predicted_labels = np.array([label_mapping[label] for label in labels])
                accuracy_train = np.mean(predicted_labels == y_train)
                st.write(f"🎯 **Độ chính xác trên tập train:** `{accuracy_train * 100:.2f}%`")

                mlflow.log_param("model", "K-Means")
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_metric("accuracy_train", accuracy_train)
                mlflow.sklearn.log_model(model, "kmeans_model")

            elif model_choice == "DBSCAN":
                unique_clusters = set(labels) - {-1}
                n_clusters_found = len(unique_clusters)
                noise_ratio = np.sum(labels == -1) / len(labels)
                st.write(f"🔍 **Số cụm tìm thấy:** `{n_clusters_found}`")
                st.write(f"🚨 **Tỉ lệ nhiễu:** `{noise_ratio * 100:.2f}%`")

                mlflow.log_param("model", "DBSCAN")
                mlflow.log_param("eps", eps)
                mlflow.log_param("min_samples", min_samples)
                mlflow.log_metric("n_clusters_found", n_clusters_found)
                mlflow.log_metric("noise_ratio", noise_ratio)
                mlflow.sklearn.log_model(model, "dbscan_model")

            if "models" not in st.session_state:
                st.session_state["models"] = []

            st.session_state["models"].append({"name": run_name, "model": model})
            st.write(f"🔹 **Mô hình đã được lưu với tên:** `{run_name}`")
            st.write(f"📋 **Danh sách các mô hình:** {[m['name'] for m in st.session_state['models']]}")
            mlflow.end_run()
            st.success("✅ Đã log dữ liệu!")
            st.markdown(f"### 🔗 [Truy cập MLflow]({st.session_state['mlflow_url']})")

def du_doan():
    st.header("Demo Dự đoán Cụm")

    # Kiểm tra xem mô hình phân cụm và nhãn đã có chưa
    if 'cluster_model' in st.session_state and 'cluster_labels' in st.session_state:
        # Tải lên ảnh hoặc file CSV
        uploaded_image = st.file_uploader("Upload ảnh chữ số (28x28, grayscale) hoặc file CSV", type=["png", "jpg", "csv"])
        true_label = st.text_input("Nhập nhãn thật (nếu có):")
        
        if uploaded_image is not None:
            if uploaded_image.name.endswith('.csv'):
                # Đọc file CSV và tiền xử lý
                df = pd.read_csv(uploaded_image)
                # Giả sử dữ liệu CSV có cột tên 'features' chứa dữ liệu đặc trưng ảnh 28x28
                # Nếu file CSV có cấu trúc khác, bạn cần điều chỉnh phần này cho phù hợp
                img_array = df['features'].values.flatten() / 255.0  # Tiền xử lý nếu cần
            else:
                # Đọc ảnh và tiền xử lý
                img = Image.open(uploaded_image).convert('L').resize((28, 28))
                img_array = np.array(img).flatten() / 255.0  # Tiền xử lý ảnh để đưa về dạng (1, 28*28)

            if st.button("Dự đoán cụm"):
                model = st.session_state['cluster_model']
                if isinstance(model, KMeans):
                    # Dự đoán cụm với KMeans
                    predicted_cluster = model.predict([img_array])[0]
                elif isinstance(model, DBSCAN):
                    # DBSCAN không có phương thức predict() nên cần tính toán khoảng cách
                    # Tính toán khoảng cách với các điểm dữ liệu đã được phân cụm
                    distances = np.linalg.norm(model.components_ - img_array, axis=1)
                    predicted_cluster = model.labels_[np.argmin(distances)]  # Dự đoán cụm với DBSCAN

                # Lấy nhãn phân cụm từ session_state
                cluster_labels = st.session_state['cluster_labels']
                st.write(f"**Dự đoán cụm:** {predicted_cluster} - Nhãn phân cụm: {cluster_labels[predicted_cluster]}")

                # Ánh xạ cụm thành chữ số nếu có
                if 'cluster_mapping' in st.session_state:
                    mapped_digit = st.session_state['cluster_mapping'].get(predicted_cluster, "N/A")
                    st.write(f"**Mã hóa thành chữ số:** {mapped_digit}")
                    
                    if true_label:
                        if str(mapped_digit) == str(true_label):
                            st.success("Dự đoán chính xác!")
                        else:
                            st.error("Dự đoán chưa chính xác!")

                # Hiển thị ảnh hoặc dữ liệu từ file CSV
                if uploaded_image.name.endswith('.csv'):
                    st.write("Dữ liệu từ file CSV đã được sử dụng cho dự đoán.")
                else:
                    st.image(img, caption="Ảnh đã upload", use_container_width=True)
    else:
        st.info("Vui lòng thực hiện phân cụm và huấn luyện mô hình trước.")

def format_time_relative(timestamp_ms):
    """Chuyển timestamp milliseconds thành thời gian dễ đọc."""
    if timestamp_ms is None:
        return "N/A"
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def display_mlflow_experiments():
    """Hiển thị danh sách Runs trong MLflow."""
    st.title("📊 MLflow Experiment Viewer")

    mlflow_input()

    experiment_name = "Clustering Algorithms"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # Lấy danh sách Runs
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    # Xử lý dữ liệu runs để hiển thị
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_data = mlflow.get_run(run_id)
        run_tags = run_data.data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")  # Lấy tên từ tags nếu có
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

    # Sắp xếp run theo thời gian chạy (mới nhất trước)
    run_info_df = pd.DataFrame(run_info)
    run_info_df = run_info_df.sort_values(by="Created", ascending=False)

    # Hiển thị danh sách Runs trong bảng
    st.write("### 🏃‍♂️ Danh sách Runs:")
    st.dataframe(run_info_df, use_container_width=True)

    # Chọn Run từ dropdown
    run_names = run_info_df["Run Name"].tolist()
    selected_run_name = st.selectbox("🔍 Chọn một Run để xem chi tiết:", run_names)

    # Lấy Run ID tương ứng
    selected_run_id = run_info_df.loc[run_info_df["Run Name"] == selected_run_name, "Run ID"].values[0]

    # Lấy thông tin Run
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
        if start_time_ms:
            start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"

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


def ClusteringAlgorithms():
  
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

    st.title("🖊️ MNIST Clusterings App")

    # Ensure the tab names are properly separated
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📘 Lý thuyết K-MEANS", 
    "📘 Lý thuyết DBSCANS", 
    "📘 Data", 
    "📥 Tải dữ liệu", 
    "🔀 Chia dữ liệu", 
    "🤖 Phân cụm", 
    "🔍 Thông tin phân cụm"
    ])


    with tab1:
        ly_thuyet_kmeans()

    with tab2:
        ly_thuyet_dbscans()
    
    with tab3:
        data()
        
    with tab4:
       up_load_db()
        
    with tab5:
        chia_du_lieu()
    
    with tab6: 
        train()

    with tab7: 
        display_mlflow_experiments() 

def run():
    ClusteringAlgorithms()

if __name__ == "__main__":
    run()
