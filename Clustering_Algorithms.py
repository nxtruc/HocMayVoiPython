import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import openml
import os
import mlflow
import shutil
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
    st.title("📊 Lý thuyết về K-means")

    # Chèn đường link ảnh trước phần Mục tiêu của K-means
    st.markdown("""
    

    **K-means** là thuật toán phân cụm phổ biến trong học máy không giám sát (unsupervised learning). Thuật toán này nhóm các điểm dữ liệu thành các cụm sao cho các điểm trong cùng một cụm có sự tương đồng cao, và các cụm có sự khác biệt lớn. Một trong những điểm nổi bật của K-means là người dùng không cần phải biết trước số lượng cụm cần phân chia.
    ![K-means Algorithm](https://machinelearningcoban.com/assets/kmeans/figure_2.png)           
                
    ### 1. Mục tiêu của K-means
    K-means nhằm phân chia tập dữ liệu thành **K cụm**, sao cho:
    - Các điểm trong cùng một cụm có độ tương đồng cao (dựa trên khoảng cách giữa các điểm).
    - Khoảng cách giữa các cụm là lớn nhất, nghĩa là các cụm phải phân tách rõ ràng.

    ### 2. Nguyên lý hoạt động
    Quy trình của K-means bao gồm các bước chính sau:
    1. **Khởi tạo số cụm (K)**: Người dùng phải chỉ định trước số lượng cụm K.
    2. **Khởi tạo các tâm cụm (centroids)**: Sau khi chọn K, thuật toán khởi tạo K centroid (tâm cụm) bằng cách chọn ngẫu nhiên hoặc sử dụng phương pháp K-means++ để cải thiện việc khởi tạo.
    3. **Gán điểm dữ liệu vào các cụm**: Mỗi điểm dữ liệu được gán vào cụm có centroid gần nhất, thường sử dụng khoảng cách Euclidean.
    4. **Cập nhật centroid**: Sau khi các điểm được gán vào cụm, centroid của mỗi cụm được tính lại là trung bình của các điểm trong cụm.
    5. **Lặp lại**: Quá trình gán điểm vào cụm và cập nhật centroid tiếp tục cho đến khi các centroid không thay đổi.

    ### 3. Thuật toán K-means
    1. Chọn K và khởi tạo các centroid.
    2. Gán mỗi điểm dữ liệu vào cụm có centroid gần nhất.
    3. Tính toán lại centroid của các cụm.
    4. Lặp lại các bước trên cho đến khi không có sự thay đổi.

    ### 4. Đánh giá chất lượng phân cụm
    Một trong những cách đánh giá phổ biến là sử dụng **Inertia** (hoặc SSE - Sum of Squared Errors), tính bằng tổng bình phương khoảng cách giữa các điểm và centroid tương ứng. Inertia càng nhỏ, các cụm càng chặt chẽ.

    ### 5. Các cải tiến của K-means
    - **K-means++**: Phương pháp này cải thiện việc khởi tạo centroid để giảm thiểu rủi ro dính vào tối ưu địa phương.
    - **Elbow Method**: Phương pháp này giúp xác định K tối ưu bằng cách vẽ đồ thị Inertia theo K và tìm điểm "elbow", nơi độ giảm của Inertia bắt đầu chậm lại.

    ### 6. Ứng dụng của K-means
    K-means có thể ứng dụng trong nhiều lĩnh vực:
    - Phân loại khách hàng trong marketing (segmentation).
    - Phân tích hình ảnh và nhận dạng mẫu.
    - Phân cụm tài liệu trong xử lý ngôn ngữ tự nhiên (NLP).
    - Phân tích dữ liệu gene trong sinh học.

    ### 7. Ví dụ về K-means trong Python
    Dưới đây là ví dụ sử dụng K-means với thư viện scikit-learn:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    # Tạo dữ liệu giả
    X, _ = make_blobs(n_samples=1000, centers=4, random_state=42)

    # Khởi tạo và huấn luyện mô hình K-means với K=4
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)

    # Dự đoán nhãn các điểm dữ liệu
    y_kmeans = kmeans.predict(X)

    # Vẽ đồ thị phân cụm
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    # Vẽ các centroid
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
    plt.title("K-means Clustering")
    plt.show()
    ```

    **Mô tả**:
    - Tạo dữ liệu giả với 4 cụm.
    - Áp dụng thuật toán K-means với K=4.
    - Vẽ đồ thị phân cụm với các centroid màu đỏ.
    """)



def ly_thuyet_dbscans():
    # Tiêu đề ứng dụng
    st.title("Thuật Toán DBSCAN - Density-Based Spatial Clustering with Noise")

    # Mô tả lý thuyết
    st.header("1. Mục tiêu của DBSCAN")
    st.write("""
        DBSCAN (Density-Based Spatial Clustering of Applications with Noise) là một thuật toán phân cụm không giám sát, 
        được sử dụng để phân chia dữ liệu thành các cụm dựa trên mật độ điểm dữ liệu và phát hiện các điểm ngoại lai.
        Thuật toán này không yêu cầu bạn phải chỉ định số cụm K trước khi chạy và có khả năng phát hiện các điểm ngoại lai.
    """)
    st.image("https://miro.medium.com/v2/resize:fit:875/0*PAjsvIpK5dmNXfM0.png", caption = "Phan cum dbscan", use_column_width=True)

    st.header("2. Cách thức hoạt động của DBSCAN")
    st.write("""
        Thuật toán DBSCAN hoạt động theo ba bước cơ bản:
        - **Điểm lõi (Core point)**: Một điểm được coi là điểm lõi nếu có ít nhất MinPts điểm trong bán kính Epsilon.
        - **Điểm biên (Border point)**: Là điểm không phải điểm lõi nhưng nằm trong bán kính Epsilon của một điểm lõi.
        - **Điểm ngoại lai (Noise point)**: Là điểm không phải điểm lõi và không nằm trong bán kính Epsilon của bất kỳ điểm lõi nào.
        
        DBSCAN hoạt động qua các bước sau:
        1. Khởi tạo: Chọn một điểm chưa được phân cụm.
        2. Xây dựng cụm: Nếu điểm này là điểm lõi, mở rộng cụm từ điểm lõi đó, bao gồm các điểm biên và các điểm lõi.
        3. Điểm ngoại lai: Các điểm không thuộc bất kỳ cụm nào sẽ là điểm ngoại lai.
    """)

    # Thực hiện phân cụm DBSCAN
    st.header("3. Phân cụm với DBSCAN")
    st.write("""
        Dưới đây là ví dụ về phân cụm dữ liệu bằng thuật toán DBSCAN.
        Thuật toán sẽ phân chia dữ liệu thành các cụm dựa trên mật độ và phát hiện các điểm ngoại lai.
    """)

    # Tạo dữ liệu giả
    X, _ = make_blobs(n_samples=1000, centers=4, random_state=42)

    # Khởi tạo và huấn luyện mô hình DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    y_dbscan = dbscan.fit_predict(X)

    # Vẽ đồ thị phân cụm
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis')
    centroids = dbscan.components_

    # Thêm tiêu đề và hiển thị đồ thị
    plt.title("Phân Cụm DBSCAN")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    st.pyplot(fig)

    # Hiển thị giải thích về các điểm ngoại lai
    st.header("4. Điểm Ngoại Lai")
    st.write("""
        Các điểm ngoại lai (noise points) sẽ được gán nhãn là -1 và không thuộc vào bất kỳ cụm nào. 
        Trong đồ thị trên, các điểm ngoại lai sẽ có màu sắc khác biệt.
    """)

    st.header("5. Ưu điểm và Nhược điểm của DBSCAN")
    st.write("""
        **Ưu điểm**:
        - Không cần biết trước số cụm.
        - Phát hiện điểm ngoại lai rất hiệu quả.
        - Phân cụm các dạng hình phức tạp.
        
        **Nhược điểm**:
        - Cần chọn tham số **Epsilon** và **MinPts** phù hợp.
        - Khó xử lý với dữ liệu có mật độ không đồng đều.
        - Không hiệu quả với dữ liệu quá lớn.
    """)

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
                X_reshaped = X[:5].reshape(5, -1)  # Chuyển đổi 5 ảnh đầu tiên thành các vector 1 chiều
                st.write(pd.DataFrame(X_reshaped))  # Hiển thị dưới dạng DataFrame

                st.success("✅ Dữ liệu MNIST đã được tải thành công!")
                st.session_state.data = (X, y)  # Lưu dữ liệu vào session_state

        # Nếu chọn upload dữ liệu từ máy
        else:
            st.markdown("#### 📤 Upload dữ liệu của bạn")

            uploaded_file = st.file_uploader("Chọn một file ảnh", type=["png", "jpg", "jpeg"], key="file_upload")

            if uploaded_file is not None:
                # Mở file hình ảnh
                image = Image.open(uploaded_file)

                # Hiển thị ảnh
                st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

                # Kiểm tra kích thước ảnh
                if image.size != (28, 28):
                    st.error("❌ Ảnh không đúng kích thước 28x28 pixel. Vui lòng tải lại ảnh đúng định dạng.")
                else:
                    st.success("✅ Ảnh hợp lệ!")
                    # Chuyển đổi hình ảnh thành dạng mảng cho mô hình
                    image = image.convert('L')  # Chuyển thành ảnh grayscale
                    image_array = np.array(image).reshape(1, -1)  # Reshape để tương thích với mô hình
                    st.session_state.data = image_array  # Lưu ảnh vào session_state

    # Kiểm tra nếu dữ liệu đã được tải
    if st.session_state.data is not None:
        st.markdown("#### ✅ Dữ liệu đã sẵn sàng!")
        
        # Nếu là dữ liệu MNIST từ OpenML
        if isinstance(st.session_state.data, tuple):
            X, y = st.session_state.data
            # Tiền xử lý dữ liệu MNIST
            st.markdown("##### 🔄 Tiến hành tiền xử lý dữ liệu MNIST")

            # Chọn loại tiền xử lý
            preprocess_option = st.selectbox("Chọn phương pháp tiền xử lý dữ liệu:", 
                                            ["Chuẩn hóa dữ liệu (Standardization)", "Giảm chiều (PCA)", "Không tiền xử lý"], key="preprocess_mnist")

            if preprocess_option == "Chuẩn hóa dữ liệu (Standardization)":
                # Chuyển đổi X thành mảng 2D
                X_reshaped = X.reshape(X.shape[0], -1)  # Reshape thành (n_samples, n_features)
                # Chuẩn hóa dữ liệu
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_reshaped)
                st.write("📊 **Dữ liệu sau khi chuẩn hóa**:")
                st.write(pd.DataFrame(X_scaled).head())

            elif preprocess_option == "Giảm chiều (PCA)":
                # Giảm chiều dữ liệu với PCA
                pca = PCA(n_components=50)  # Giảm xuống 50 chiều
                X_pca = pca.fit_transform(X.reshape(X.shape[0], -1))  # Reshape trước khi PCA
                st.write("📊 **Dữ liệu sau khi giảm chiều (PCA)**:")
                st.write(pd.DataFrame(X_pca).head())

            else:
                st.write("📊 **Dữ liệu không có tiền xử lý**.")

        # Nếu là ảnh tải lên từ máy
        elif isinstance(st.session_state.data, np.ndarray):  # Nếu là ảnh người dùng tải lên
            st.markdown("#### 👁️ Tiến hành tiền xử lý ảnh")

            # Chọn loại tiền xử lý cho ảnh
            preprocess_option_image = st.selectbox("Chọn phương pháp tiền xử lý ảnh:",
                                                   ["Chuẩn hóa ảnh", "Không tiền xử lý"], key="preprocess_image")

            if preprocess_option_image == "Chuẩn hóa ảnh":
                # Chuẩn hóa ảnh
                image_scaled = st.session_state.data / 255.0  # Chuyển đổi giá trị pixel về phạm vi [0, 1]
                st.write("📊 **Ảnh sau khi chuẩn hóa**:")
                st.image(image_scaled.reshape(28, 28), caption="Ảnh sau khi chuẩn hóa", use_column_width=True)

            else:
                st.write("📊 **Ảnh không có tiền xử lý**.")

    else:
        st.warning("🔸 Vui lòng tải dữ liệu trước khi tiếp tục làm việc.")

    # Hiển thị lưu ý
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
    X_train_norm = (X_train / 255.0).reshape(X_train.shape[0], -1)  # Chuẩn hóa và làm phẳng

    model_choice = st.selectbox("Chọn mô hình:", ["K-Means", "DBSCAN"])

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
        with mlflow.start_run():
            model.fit(X_train_pca)
            st.success("✅ Huấn luyện thành công!")

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

                # Kiểm tra và tính độ chính xác trên tập validation và test nếu có
                if "X_val" in st.session_state and "y_val" in st.session_state:
                    X_val = st.session_state["X_val"]
                    y_val = st.session_state["y_val"]
                    X_val_norm = (X_val / 255.0).reshape(X_val.shape[0], -1)
                    X_val_pca = pca.transform(X_val_norm)
                    val_labels = model.predict(X_val_pca)
                    predicted_val_labels = np.array([label_mapping.get(label, -1) for label in val_labels])
                    accuracy_val = np.mean(predicted_val_labels == y_val)
                    st.write(f"🎯 **Độ chính xác trên tập validation:** `{accuracy_val * 100:.2f}%`")

                if "X_test" in st.session_state and "y_test" in st.session_state:
                    X_test = st.session_state["X_test"]
                    y_test = st.session_state["y_test"]
                    X_test_norm = (X_test / 255.0).reshape(X_test.shape[0], -1)
                    X_test_pca = pca.transform(X_test_norm)
                    test_labels = model.predict(X_test_pca)
                    predicted_test_labels = np.array([label_mapping.get(label, -1) for label in test_labels])
                    accuracy_test = np.mean(predicted_test_labels == y_test)
                    st.write(f"🎯 **Độ chính xác trên tập test:** `{accuracy_test * 100:.2f}%`")

                # Log vào MLflow
                mlflow.log_param("model", "K-Means")
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_metric("accuracy_train", accuracy_train)
                if "accuracy_val" in locals():
                    mlflow.log_metric("accuracy_val", accuracy_val)
                if "accuracy_test" in locals():
                    mlflow.log_metric("accuracy_test", accuracy_test)
                mlflow.sklearn.log_model(model, "kmeans_model")

            elif model_choice == "DBSCAN":
                unique_clusters = set(labels) - {-1}
                n_clusters_found = len(unique_clusters)
                noise_ratio = np.sum(labels == -1) / len(labels)
                st.write(f"🔍 **Số cụm tìm thấy:** `{n_clusters_found}`")
                st.write(f"🚨 **Tỉ lệ nhiễu:** `{noise_ratio * 100:.2f}%`")

                # Log vào MLflow
                mlflow.log_param("model", "DBSCAN")
                mlflow.log_param("eps", eps)
                mlflow.log_param("min_samples", min_samples)
                mlflow.log_metric("n_clusters_found", n_clusters_found)
                mlflow.log_metric("noise_ratio", noise_ratio)
                mlflow.sklearn.log_model(model, "dbscan_model")

            if "models" not in st.session_state:
                st.session_state["models"] = []

            model_name = model_choice.lower().replace(" ", "_")
            count = 1
            new_model_name = model_name
            while any(m["name"] == new_model_name for m in st.session_state["models"]):
                new_model_name = f"{model_name}_{count}"
                count += 1

            st.session_state["models"].append({"name": new_model_name, "model": model})
            st.write(f"🔹 **Mô hình đã được lưu với tên:** `{new_model_name}`")
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
    "📘 Review database", 
    "📥 Tải dữ liệu", 
    "🔀 Chia dữ liệu", 
    "🤖 Phân cụm", 
    "🔍 Thông tin phân cụm",
    "🧠 Dự đoán"
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
    
    with tab8: 
        du_doan()

def run():
    ClusteringAlgorithms()

if __name__ == "__main__":
    run()