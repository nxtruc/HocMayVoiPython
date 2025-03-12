import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import openml
import joblib
import shutil
import pandas as pd
import time
import os
import mlflow
import humanize
from datetime import datetime
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


def ly_thuyet_Decision_tree():
    st.header("📖 Lý thuyết về Decision Tree") 
    st.markdown(" ### 1️⃣ Decision Tree là gì?")
    st.write("""
    Decision Tree (Cây quyết định) là một thuật toán học có giám sát được sử dụng trong **phân loại (classification)** và **hồi quy (regression)**.
    Nó hoạt động bằng cách chia dữ liệu thành các nhóm nhỏ hơn dựa trên các điều kiện được thiết lập tại các **nút (nodes)** của cây.
    """) 
    
    image_url = "https://machinelearningcoban.com/assets/34_id3/dt_ex1.png"
    article_url = "https://machinelearningcoban.com/2018/01/14/id3/"

    # Hiển thị ảnh có thể nhấp vào, căn giữa và thêm caption
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;">Ví dụ về việc ra quyết định dựa trên các câu hỏi.</p>
        </div>
        """,
        unsafe_allow_html=True
    ) 

    st.markdown(" ### 2️⃣ ý tưởng") 

    st.markdown(
    """
    ##### 2.1 Vấn đề cần giải quyết:  
    - Khi xây dựng cây quyết định, ta cần xác định thứ tự thuộc tính được sử dụng để chia dữ liệu.  
    - Với bài toán có nhiều thuộc tính và mỗi thuộc tính có nhiều giá trị, việc tìm giải pháp tối ưu là không khả thi.  
    - Thay vì tìm nghiệm tối ưu toàn cục, ta sử dụng một phương pháp **tham lam (greedy)**:  
      → Chọn thuộc tính **tốt nhất** tại mỗi bước dựa trên một tiêu chí nào đó.
    """
    )   
    image_url = "https://www.mdpi.com/entropy/entropy-27-00035/article_deploy/html/images/entropy-27-00035-g001-550.jpg"
    article_url = "http://mdpi.com/1099-4300/27/1/35"

    # Hiển thị ảnh có thể nhấp vào, căn giữa và thêm caption
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;"><i>Set of decision trees 𝑆={{𝑇𝑟𝑒𝑒1, 𝑇𝑟𝑒𝑒2}}</i></p>
        </div>
        """,
        unsafe_allow_html=True
    )   
    st.markdown(
    """
    ##### 2.2 Quá trình chia nhỏ dữ liệu:
    - Với mỗi thuộc tính được chọn, dữ liệu được chia thành các **child node** theo giá trị của thuộc tính đó.
    - Sau đó, tiếp tục lặp lại quá trình này cho từng **child node**.
    """
    )
    image_url = "https://cdn.analyticsvidhya.com/wp-content/uploads/2024/09/ns1.webp"
    article_url = "https://www.analyticsvidhya.com/blog/2020/06/4-ways-split-decision-tree/"

    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;"><i>Ví dụ quá trình chia nhỏ dữ liệu</i></p>
        </div>
        """,
        unsafe_allow_html=True
    )   
    st.markdown(
    """
    ##### 2.4 Hàm số Entropy: 
    - Entropy là một khái niệm trong lý thuyết thông tin, được sử dụng để đo **độ hỗn loạn (impurity)** hoặc **độ không chắc chắn** của một tập dữ liệu. 
    - Trong cây quyết định (Decision Tree), entropy giúp đánh giá chất lượng của một phép chia dữ liệu.
    """
    )
    st.latex(r"H(p) = - \sum_{i=1}^{n} p_i \log(p_i)")
    st.markdown(
    """
    Trong đó:
    - log có thể là logarit tự nhiên hoặc log cơ số 2.
    - Quy ước: \\( 0 \log 0 = 0 \\).
    """
    )

    st.markdown(
    """
    ##### 🔍 Ý nghĩa của Entropy trong phân phối xác suất:
    """)

    st.markdown(
        """
        - Nếu **phân phối tinh khiết** (chỉ có một giá trị có xác suất 1, còn lại là 0):  
        → **Entropy = 0**, tức **không có sự không chắc chắn**.
        - Nếu **phân phối vẩn đục nhất** (các giá trị có xác suất bằng nhau, ví dụ p1 = p2 = 0.5)  
        → **Entropy đạt giá trị cao nhất**, tức **độ không chắc chắn lớn nhất**.
        """
    )
    image_url = "https://machinelearningcoban.com/assets/34_id3/entropy.png"
    article_url = "https://machinelearningcoban.com/2018/01/14/id3/"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;"><i>Ví dụ Đồ thị của hàm entropy với 
            n
            =
            2
            </i></p>
        </div>
        """,
        unsafe_allow_html=True
    )   

    st.markdown(" ### 3️⃣ Thuật toán ID3")
    st.markdown("##### Tính toán Entropy tại một Node")
    st.markdown(
        """
        Với tập dữ liệu **S** gồm **N** điểm dữ liệu thuộc **C** lớp, entropy tại node được tính bằng:
        """
    )
    st.latex(r"H(S) = - \sum_{c=1}^{C} \frac{N_c}{N} \log \left(\frac{N_c}{N} \right)")
    st.markdown("Trong đó, \\( N_c \\) là số điểm thuộc class **c**.")

    st.markdown("##### Entropy sau khi phân chia theo thuộc tính **x**")
    st.markdown(
        """
        Khi chọn thuộc tính **x**, tập **S** được chia thành **K** child node \\( S_1, S_2, ..., S_K \\) 
        với kích thước tương ứng \\( m_1, m_2, ..., m_K \\). Entropy tổng có trọng số sau khi phân chia:
        """
    )
    st.latex(r"H(x,S) = \sum_{k=1}^{K} \frac{m_k}{N} H(S_k)")
    st.markdown("Việc lấy trọng số là cần thiết vì mỗi node có số lượng điểm dữ liệu khác nhau.")

    st.markdown("##### Information Gain – Tiêu chí chọn thuộc tính")
    st.markdown("Để xác định thuộc tính nào giúp giảm entropy tốt nhất, ta tính **Information Gain**:")
    st.latex(r"G(x,S) = H(S) - H(x,S)")

    st.markdown("ID3 chọn thuộc tính \\( x^* \\) sao cho **Information Gain** lớn nhất:")
    st.latex(r"x^* = \arg\max_{x} G(x,S) = \arg\min_{x} H(x,S)")
    st.markdown("Nghĩa là ta chọn thuộc tính giúp entropy giảm nhiều nhất sau khi phân chia.")

    st.markdown("##### Khi nào dừng phân chia?")
    st.markdown(
        """
        ID3 dừng phân chia khi:
        - ✅ Tất cả dữ liệu trong node thuộc cùng một class.
        - ✅ Không còn thuộc tính nào để chia tiếp.
        - ✅ Số lượng điểm dữ liệu trong node quá nhỏ.
        """
    )

def ly_thuyet_SVM():
    st.header("📖 Lý thuyết về SVM")
    st.markdown(" ### 1️⃣ SVM là gì?")
    st.write("""
    - Support Vector Machine (SVM) là một thuật toán học có giám sát dùng cho **phân loại** và hồi quy.    
    - Mục tiêu của SVM là tìm ra **siêu phẳng** (hyperplane) tối ưu để phân tách dữ liệu thuộc các lớp khác nhau với một **khoảng cách lề** (margin) lớn nhất.
        """
    )

    image_url = "https://neralnetwork.wordpress.com/wp-content/uploads/2018/01/svm1.png"
    article_url = "https://neralnetwork.wordpress.com/2018/05/11/thuat-toan-support-vector-machine-svm/"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;"><i>minh họa về SVM
            </i></p>
        </div>
        """,
        unsafe_allow_html=True
    )   
    st.markdown(" ### 2️⃣ Ý tưởng của SVM") 
    st.markdown(" ##### 2.1 Tìm siêu phẳng phân tách tối ưu")
    st.write(
        "Một siêu phẳng (hyperplane) trong không gian đặc trưng có dạng:\n"
        "$w \cdot x + b = 0$\n"
        "Trong đó:\n"
        "- $w$ là vector pháp tuyến của siêu phẳng.\n"
        "- $x$ là điểm dữ liệu.\n"
        "- $b$ là hệ số điều chỉnh độ dịch chuyển của siêu phẳng.\n"
        "\n"
    )
    image_url = "https://www.researchgate.net/publication/244858164/figure/fig3/AS:670028080898057@1536758551648/An-example-of-the-optimal-separating-hyperplane-of-support-vector-machine-SVM-with-the.png"
    article_url = "https://www.researchgate.net/figure/An-example-of-the-optimal-separating-hyperplane-of-support-vector-machine-SVM-with-the_fig3_244858164"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;"><i>minh họa quá trình tìm siêu phẳng phân tách tối ưu
            </i></p>
        </div>
        """,
        unsafe_allow_html=True
    )   
    st.write("Mục tiêu của SVM là tìm siêu phẳng có khoảng cách lớn nhất tới các điểm gần nhất thuộc hai lớp khác nhau (các support vectors).\n"
    "Khoảng cách này được gọi là lề (margin).")

    st.markdown(" ##### 2.2 Tối đa hóa lề (Maximum Margin)")
    st.write(
        "Lề (margin) là khoảng cách giữa siêu phẳng và các điểm dữ liệu gần nhất thuộc hai lớp.\n"
        "SVM cố gắng tối đa hóa lề này để đảm bảo mô hình có khả năng tổng quát hóa tốt nhất."
    )

    st.latex(r"""
    D = \frac{|w^T x_0 + b|}{||w||_2}
    """)

    st.markdown("##### Trong đó:")
    st.markdown("- $w^T x_0$ là tích vô hướng giữa vector pháp tuyến của hyperplane và điểm $x_0$.")
    st.markdown("- $||w||_2$ là độ dài (norm) của vector pháp tuyến $w$, được tính bằng công thức:")

    st.latex(r"""
    ||w||_2 = \sqrt{w_1^2 + w_2^2 + \dots + w_n^2}
    """)

    st.markdown("- Dấu $| \cdot |$ biểu thị giá trị tuyệt đối, giúp đảm bảo khoảng cách luôn là giá trị không âm.")

    image_url = "https://www.researchgate.net/publication/226587707/figure/fig3/AS:669184333725696@1536557386160/Margin-maximization-principle-the-basic-idea-of-Support-Vector-Machine.ppm"
    article_url = "https://www.researchgate.net/figure/Margin-maximization-principle-the-basic-idea-of-Support-Vector-Machine_fig3_226587707"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="300">
            </a>
            <p style="font-size: 14px; color: gray;"><i>minh họa tìm khoảng cách từ điểm đến siêu phẳng
            </i></p>
        </div>
        """,
        unsafe_allow_html=True
    )   

    st.markdown(" ##### 2.3 Khi dữ liệu không tách được tuyến tính")
    st.write(
        "Trong trường hợp dữ liệu không thể phân tách bằng một đường thẳng (tức là không tuyến tính), \n"
        "SVM sử dụng hàm kernel (kernel trick) để ánh xạ dữ liệu sang không gian bậc cao hơn, nơi chúng có thể phân tách tuyến tính."
    )

    st.markdown(" ##### Các kernel phổ biến:")
    st.markdown("- **Linear Kernel**: Sử dụng khi dữ liệu có thể phân tách tuyến tính.")
    st.markdown("- **Polynomial Kernel**: Ánh xạ dữ liệu sang không gian bậc cao hơn.")
    st.markdown("- **RBF (Radial Basis Function) Kernel**: Tốt cho dữ liệu phi tuyến tính.")
    st.markdown("- **Sigmoid Kernel**: Mô phỏng như mạng neural.")

    st.markdown(" ##### 2.4 Vị trí tương đối với một siêu phẳng ")
    st.markdown(
    """
    **Nếu** $w^T x + b > 0$ **:**
    - Điểm $x$ nằm ở **phía dương** của siêu phẳng.
    - Trong hình, các điểm thuộc lớp dương (dấu "+") nằm ở vùng này.
    
    **Nếu** $w^T x + b < 0$ **:**
    - Điểm $x$ nằm ở **phía âm** của siêu phẳng.
    - Trong hình, các điểm thuộc lớp âm (dấu "-") nằm ở vùng này.
    
    **Nếu** $w^T x + b = 0$ **:**
    - Điểm $x$ nằm **trên siêu phẳng phân tách**.
    - Trong SVM, siêu phẳng này là đường quyết định, phân chia dữ liệu thành hai lớp khác nhau.
    
    Hình bên dưới minh họa cách siêu phẳng phân chia dữ liệu.
    """
    )
    image_url = "https://machinelearningcoban.com/assets/19_svm/svm2.png"
    article_url = "https://machinelearningcoban.com/2017/04/09/smv/"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <a href="{article_url}" target="_blank">
                <img src="{image_url}" width="500">
            </a>
            <p style="font-size: 14px; color: gray;"><i>
            </i></p>
        </div>
        """,
        unsafe_allow_html=True
    )  

def data():
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

                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                for percent_complete in range(100):
                    time.sleep(0.05 + (27 / 100))  # Thêm 27 giây vào tiến trình tải
                    progress_bar.progress(percent_complete + 1)
                    progress_text.text(f"⏳ Đang tải... {percent_complete + 1}%")
                
                # Tải dữ liệu MNIST từ file .npy
                X = np.load("X.npy")
                y = np.load("y.npy")

                st.success("✅ Dữ liệu MNIST đã được tải thành công!")
                st.session_state.data = (X, y)  # Lưu dữ liệu vào session_state
                progress_bar.empty()
                progress_text.empty()

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
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            for percent_complete in range(100):
                time.sleep(0.02 + (27 / 100))  # Thêm 27 giây vào tiến trình tiền xử lý
                progress_bar.progress(percent_complete + 1)
                progress_text.text(f"⏳ Đang xử lý... {percent_complete + 1}%")
            
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
            
            progress_bar.empty()
            progress_text.empty()
            st.pyplot(fig)
    
    else:
        st.warning("🔸 Vui lòng tải dữ liệu trước khi tiếp tục làm việc.")


def chia_du_lieu():
    st.title("📌 Chia dữ liệu Train/Test")

    # Đọc dữ liệu
    X = np.load("X.npy")
    y = np.load("y.npy")
    total_samples = X.shape[0]

    
    # Nếu chưa có cờ "data_split_done", đặt mặc định là False
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, total_samples, 10000)
    
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    if st.button("✅ Xác nhận & Lưu") and not st.session_state.data_split_done:
        st.session_state.data_split_done = True  # Đánh dấu đã chia dữ liệu
        
        # Chia dữ liệu theo tỷ lệ đã chọn
        X_selected, _, y_selected, _ = train_test_split(
            X, y, train_size=num_samples, stratify=y, random_state=42
        )

        # Chia train/test
        stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        # Chia train/val
        stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size / (100 - test_size),
            stratify=stratify_option, random_state=42
        )

        # Lưu dữ liệu vào session_state
        st.session_state.total_samples= num_samples
        st.session_state.X_train = X_train
        st.session_state.X_val = X_val
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_val = y_val
        st.session_state.y_test = y_test
        st.session_state.test_size = X_test.shape[0]
        st.session_state.val_size = X_val.shape[0]
        st.session_state.train_size = X_train.shape[0]

        # Hiển thị thông tin chia dữ liệu
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")



def train():
    """Huấn luyện mô hình Decision Tree hoặc SVM và lưu trên MLflow với thanh tiến trình hiển thị %."""
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

    # 📌 Đặt tên thí nghiệm
    experiment_name = st.text_input("📌 Đặt tên thí nghiệm:", "default_experiment", 
                                    help="Tên của thí nghiệm để dễ dàng quản lý trên MLflow.")

    # 📌 Lựa chọn mô hình
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])
    
    if model_choice == "Decision Tree":
        criterion = st.selectbox("Criterion (Hàm mất mát: Gini/Entropy) ", ["gini", "entropy"])
        max_depth = st.slider("max_depth", 1, 20, 5, help="Giới hạn độ sâu của cây để tránh overfitting.")
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    else:
        C = st.slider("C (Hệ số điều chuẩn)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel (Hàm nhân)", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel)

    # 📌 Chọn số folds cho KFold Cross-Validation
    k_folds = st.slider("Số folds", 2, 10, 5, help="Số tập chia để đánh giá mô hình.")

    # 🚀 Bắt đầu huấn luyện
    if st.button("Huấn luyện mô hình"):
        with st.spinner("🔄 Đang huấn luyện mô hình..."):
            progress_bar = st.progress(0)
            percent_text = st.empty()  # Chỗ hiển thị %

            with mlflow.start_run(run_name=experiment_name):
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                cv_scores = []

                # Vòng lặp Cross-Validation
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                    model.fit(X_train_fold, y_train_fold)
                    val_pred = model.predict(X_val_fold)
                    val_acc = accuracy_score(y_val_fold, val_pred)
                    cv_scores.append(val_acc)
                    mlflow.log_metric("cv_accuracy", val_acc, step=fold)

                    # Cập nhật thanh trạng thái (bỏ qua hiển thị từng fold)
                    percent_done = int(((fold + 1) / k_folds) * 70)
                    progress_bar.progress(percent_done)
                    percent_text.write(f"**Tiến độ: {percent_done}%**")

                    time.sleep(1)  

                # Kết quả CV
                cv_accuracy_mean = np.mean(cv_scores)
                cv_accuracy_std = np.std(cv_scores)
                st.success(f"✅ **Cross-Validation Accuracy:** {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}")

                # Huấn luyện trên toàn bộ tập train
                model.fit(X_train, y_train)

                # Cập nhật tiến trình (85%)
                progress_bar.progress(85)
                percent_text.write("**Tiến độ: 85%**")

                # Dự đoán trên test set
                y_pred = model.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                mlflow.log_metric("test_accuracy", test_acc)
                st.success(f"✅ **Độ chính xác trên test set:** {test_acc:.4f}")

                # Delay thêm 20s trước khi hoàn thành
                for i in range(1, 21):
                    progress_percent = 85 + (i // 2)
                    progress_bar.progress(progress_percent)
                    percent_text.write(f"**Tiến độ: {progress_percent}%**")
                    time.sleep(1)

                # Hoàn thành tiến trình
                progress_bar.progress(100)
                percent_text.write("✅ **Tiến độ: 100% - Hoàn thành!**")

                # Log tham số vào MLflow
                mlflow.log_param("experiment_name", experiment_name)
                mlflow.log_param("model", model_choice)
                mlflow.log_param("k_folds", k_folds)
                if model_choice == "Decision Tree":
                    mlflow.log_param("criterion", criterion)
                    mlflow.log_param("max_depth", max_depth)
                else:
                    mlflow.log_param("C", C)
                    mlflow.log_param("kernel", kernel)

                mlflow.log_metric("cv_accuracy_mean", cv_accuracy_mean)
                mlflow.log_metric("cv_accuracy_std", cv_accuracy_std)
                mlflow.sklearn.log_model(model, model_choice.lower())

                st.success(f"✅ Đã log dữ liệu cho **{experiment_name}**!")
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


def format_time_relative(timestamp_ms):
    """Chuyển timestamp sang dạng 'X minutes ago'."""
    if timestamp_ms:
        created_at_dt = datetime.fromtimestamp(timestamp_ms / 1000)
        return humanize.naturaltime(datetime.now() - created_at_dt)
    return "N/A"

def display_mlflow_experiments():
    """Hiển thị danh sách Runs trong MLflow."""
    st.title("📊 MLflow Experiment Viewer")

    # Kết nối MLflow (Tự động gọi mlflow_input)
    mlflow_input()

    experiment_name = "Classifications"
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
        duration = run_data.info.end_time - run_data.info.start_time if run_data.info.end_time else "Đang chạy"
        source = run_tags.get("mlflow.source.name", "Unknown")

        run_info.append({
            "Run Name": run_name,
            "Run ID": run_id,
            "Created": created_time,
            "Duration": duration,
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
        
        start_time_ms = selected_run.info.start_time  # Thời gian lưu dưới dạng milliseconds
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
