import streamlit as st
import graphviz

def draw_usecase_diagram():
    st.title("Biểu đồ Use Case - Quản lý cửa hàng Mỳ cay SASIN")
    
    diagram = graphviz.Digraph()
    
    # Tác nhân
    diagram.node("QL", "Người Quản Lý", shape="ellipse")
    diagram.node("TN", "Nhân Viên Thu Ngân", shape="ellipse")
    diagram.node("PV", "Nhân Viên Phục Vụ", shape="ellipse")
    diagram.node("BEP", "Nhân Viên Bếp", shape="ellipse")
    diagram.node("CHU", "Chủ Cửa Hàng", shape="ellipse")
    
    # Use Cases chính
    diagram.node("UC1", "Quản lý danh mục", shape="box")
    diagram.node("UC2", "Quản lý kho", shape="box")
    diagram.node("UC3", "Quản lý bán hàng", shape="box")
    diagram.node("UC4", "Quản lý nhân sự & lương", shape="box")
    diagram.node("UC5", "Thống kê, báo cáo", shape="box")
    
    # Use Cases chi tiết
    diagram.node("UC1.1", "Quản lý nhân viên", shape="box")
    diagram.node("UC1.2", "Quản lý chức vụ", shape="box")
    diagram.node("UC1.3", "Quản lý nhà cung cấp", shape="box")
    
    diagram.node("UC2.1", "Nhập kho", shape="box")
    diagram.node("UC2.2", "Xuất kho", shape="box")
    diagram.node("UC2.3", "Báo cáo xuất nhập kho", shape="box")
    
    diagram.node("UC3.1", "Quản lý bàn", shape="box")
    diagram.node("UC3.2", "Quản lý thực đơn", shape="box")
    diagram.node("UC3.3", "Quản lý order", shape="box")
    diagram.node("UC3.4", "Quản lý hóa đơn", shape="box")
    diagram.node("UC3.5", "Thanh toán hóa đơn", shape="box")
    
    diagram.node("UC4.1", "Chấm công nhân viên", shape="box")
    diagram.node("UC4.2", "Tính lương", shape="box")
    
    diagram.node("UC5.1", "Báo cáo bán hàng", shape="box")
    diagram.node("UC5.2", "Báo cáo xuất nhập kho", shape="box")
    diagram.node("UC5.3", "Báo cáo lương", shape="box")
    
    # Mối quan hệ giữa tác nhân và Use Cases
    diagram.edge("QL", "UC1")
    diagram.edge("QL", "UC2")
    diagram.edge("QL", "UC4")
    diagram.edge("QL", "UC5")
    
    diagram.edge("TN", "UC3")
    diagram.edge("PV", "UC3")
    diagram.edge("BEP", "UC3")
    
    diagram.edge("CHU", "UC5")
    diagram.edge("CHU", "UC2")
    
    # Quan hệ include
    diagram.edge("UC1", "UC1.1", label="<<include>> (Quản lý nhân viên là một phần của danh mục)")
    diagram.edge("UC1", "UC1.2", label="<<include>> (Quản lý chức vụ là một phần của danh mục)")
    diagram.edge("UC1", "UC1.3", label="<<include>> (Quản lý nhà cung cấp là một phần của danh mục)")
    
    diagram.edge("UC2", "UC2.1", label="<<include>> (Nhập kho là một phần của quản lý kho)")
    diagram.edge("UC2", "UC2.2", label="<<include>> (Xuất kho là một phần của quản lý kho)")
    diagram.edge("UC2", "UC2.3", label="<<include>> (Báo cáo xuất nhập kho là một phần của quản lý kho)")
    
    diagram.edge("UC3", "UC3.3", label="<<include>> (Quản lý order là một phần của bán hàng)")
    diagram.edge("UC3.3", "UC3.5", label="<<extend>> (Thanh toán hóa đơn có thể phát sinh sau order)")
    
    diagram.edge("UC4.1", "UC4.2", label="<<extend>> (Tính lương dựa trên chấm công)")
    
    diagram.edge("UC5", "UC5.1", label="<<include>> (Báo cáo bán hàng là một phần của thống kê)")
    diagram.edge("UC5", "UC5.2", label="<<include>> (Báo cáo xuất nhập kho là một phần của thống kê)")
    diagram.edge("UC5", "UC5.3", label="<<include>> (Báo cáo lương là một phần của thống kê)")
    
    # Hiển thị biểu đồ
    st.graphviz_chart(diagram)

if __name__ == "__main__":
    draw_usecase_diagram()