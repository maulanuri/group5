import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

# ===================== CONFIG & THEME =====================

st.set_page_config(
    page_title="🧮 Matrix Transformations in Image Processing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- VIDEO BACKGROUND (HTML langsung) ----------
def set_video_background(video_path: str):
    """Set an mp4 video as full-screen background using HTML/CSS."""
    if not os.path.exists(video_path):
        st.warning(f"Video background tidak ditemukan: {video_path}")
        return

    with open(video_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    video_data_url = f"data:video/mp4;base64,{b64}"

    st.markdown(
        f"""
        <style>
        .video-bg {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1;
            object-fit: cover;
        }}
        .stApp {{
            background: transparent !important;
        }}
        </style>
        <video class="video-bg" autoplay muted loop playsinline>
            <source src="{video_data_url}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )

set_video_background("assets/background.mp4")

# ----- Initialize Session State -----
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "light"
if "language" not in st.session_state:
    st.session_state["language"] = "id"
if "original_img" not in st.session_state:
    st.session_state.original_img = None
if "geo_transform" not in st.session_state:
    st.session_state["geo_transform"] = None
if "image_filter" not in st.session_state:
    st.session_state["image_filter"] = None
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "tools"

# ===================== TRANSLATIONS =====================

translations = {
    "id": {
        "title": "🧮 Transformasi Matriks dalam Pengolahan Gambar",
        "subtitle": "📌 Demo interaktif transformasi matriks 2D dan filtering Gambar",
        "app_goal": "🎯 **Tujuan Aplikasi**",
        "app_goal_content": "Aplikasi ini bertujuan untuk memberikan pemahaman visual tentang bagaimana transformasi matriks dan operasi filtering bekerja dalam pengolahan gambar digital melalui demo interaktif.",
        "matrix_concept": "🧊 **Konsep Transformasi Matriks 2D**",
        "matrix_concept_content": "Gambar 2D direpresentasikan sebagai kumpulan titik (x, y). Transformasi linear seperti translasi, scaling, rotasi, shearing, dan refleksi dapat diwujudkan melalui perkalian matriks.",
        "filter_concept": "📐 **Konsep Filtering Gambar**",
        "filter_concept_content": "Filtering menggunakan kernel konvolusi yang digeser pada gambar untuk menghasilkan efek seperti blur, sharpen, atau deteksi tepi melalui operasi matematika pixel-per-pixel.",
        "interactive_concept": "🕹️ **Mengapa Interaktif?**",
        "interactive_concept_content": "Interaktivitas memungkinkan pengguna langsung melihat efek parameter perubahan, membuat konsep matematika lebih intuitif dan mudah dipahami.",
        "quick_concepts": "📖 **Konsep Singkat**",
        "quick_concepts_text": "- 🔁 Transformasi 2D: Ubah posisi pixel\n- 🧮 Konvolusi: Kernel geser untuk filtering\n- 🎯 Interaktif: Lihat perubahan langsung",
        "upload_explanation_title": "📸 **Panduan Unggah Gambar**",
        "upload_explanation_text": "1. Klik tombol 'Unggah gambar' di bawah\n2. Pilih file gambar (PNG/JPG/JPEG)\n3. Tunggu hingga gambar ditampilkan\n4. Gunakan alat processing sesuai kebutuhan",
        "upload_title": "🖼️ **Unggah Gambar**",
        "upload_label": "📤 Klik untuk mengunggah gambar (PNG/JPG/JPEG)",
        "upload_success": "✅ Gambar berhasil diunggah!",
        "upload_preview": "🖼️ Pratinjau Gambar Asli",
        "upload_info": "⬆️ Silakan unggah gambar terlebih dahulu untuk menggunakan alat di bawah ini.",
        "tools_title": "🛠️ **Alat Pengolahan Gambar**",
        "tools_subtitle": "🎛️ Pilih transformasi atau filter untuk memulai",
        "geo_title": "🔁 **Transformasi Geometri**",
        "geo_desc": "Ubah posisi, ukuran, dan orientasi pixel menggunakan operasi matriks linear.",
        "btn_translation": "↔️ Translasi",
        "btn_scaling": "📏 Scaling",
        "btn_rotation": "🔄 Rotasi",
        "btn_shearing": "📐 Shearing",
        "btn_reflection": "🪞 Refleksi",
        "geo_info": "🔔 Silakan unggah gambar terlebih dahulu untuk mencoba transformasi.",
        "trans_settings": "↔️ **Pengaturan Translasi**",
        "trans_dx": "dx (pergeseran horizontal)",
        "trans_dy": "dy (pergeseran vertikal)",
        "btn_apply": "Terapkan",
        "trans_result": "**Hasil Translasi**",
        "scale_settings": "📏 **Pengaturan Scaling**",
        "scale_x": "Skala X",
        "scale_y": "Skala Y",
        "scale_result": "**Hasil Scaling**",
        "rot_settings": "🔄 **Pengaturan Rotasi**",
        "rot_angle": "Sudut rotasi (derajat)",
        "rot_result": "**Hasil Rotasi**",
        "shear_settings": "📐 **Pengaturan Shearing**",
        "shear_x": "Faktor shear X",
        "shear_y": "Faktor shear Y",
        "shear_result": "**Hasil Shearing**",
        "refl_settings": "🪞 **Pengaturan Refleksi**",
        "refl_axis": "Sumbu refleksi",
        "refl_result": "**Hasil Refleksi**",
        "hist_title": "📊 **Histogram Gambar**",
        "hist_desc": "Analisis distribusi intensitas pixel untuk optimasi brightness dan kontras.",
        "btn_histogram": "Tampilkan Histogram 📈",
        "hist_warning": "Silakan unggah gambar terlebih dahulu untuk menampilkan histogram.",
        "filter_title": "🧮 **Filtering Gambar**",
        "filter_desc": "Modifikasi nilai pixel melalui konvolusi untuk berbagai efek visual.",
        "btn_blur": "🔲 Blur",
        "btn_sharpen": "✨ Sharpen",
        "btn_background": "🎯 Background",
        "btn_grayscale": "⚫ Grayscale",
        "btn_edge": "🔍 Edge",
        "btn_brightness": "☀️ Brightness",
        "filter_info": "🔔 Silakan unggah gambar terlebih dahulu untuk menggunakan filter.",
        "blur_settings": "🔲 **Pengaturan Filter Blur**",
        "blur_kernel": "Ukuran kernel",
        "blur_result": "**Hasil Blur**",
        "sharpen_settings": "✨ **Pengaturan Filter Sharpen**",
        "sharpen_desc": "Tingkatkan detail dan tepi pada gambar.",
        "sharpen_result": "**Hasil Sharpen**",
        "bg_settings": "🎯 **Pengaturan Penghapusan Background**",
        "bg_method": "Metode",
        "bg_result": "**Hasil Penghapusan Background**",
        "gray_settings": "⚫ **Pengaturan Konversi Grayscale**",
        "gray_desc": "Konversi gambar ke grayscale (hitam putih).",
        "gray_result": "**Hasil Grayscale**",
        "edge_settings": "🔍 **Pengaturan Deteksi Tepi**",
        "edge_method": "Metode edge",
        "edge_result": "**Gambar Edge**",
        "bright_settings": "☀️ **Pengaturan Brightness & Kontras**",
        "bright_brightness": "Brightness",
        "bright_contrast": "Kontras",
        "bright_result": "**Hasil Brightness/Kontras**",
        "team_title": "👥 **Anggota Tim**",
        "team_subtitle": "Kelompok 5 – Anggota dan Peran",
        "team_sid": "NIM",
        "team_role": "Peran",
        "team_contribution": "Kontribusi",
        "team_group": "Kelompok",
        "axis_x": "Sumbu-X",
        "axis_y": "Sumbu-Y",
        "axis_diag": "Diagonal",
        "dark_mode": "Mode Gelap",
        "light_mode": "Mode Terang",
        "menu_tools": "🛠️ Alat Processing",
        "menu_team": "👥 Tim Kami",
    },
    "en": {
        "title": "🧮 Matrix Transformations in Image Processing",
        "subtitle": "📌 Interactive demo of 2D matrix transformations and image filtering",
        "app_goal": "🎯 **Application Goal**",
        "app_goal_content": "This application aims to provide visual understanding of how matrix transformations and filtering operations work in digital image processing through interactive demonstrations.",
        "matrix_concept": "🧊 **2D Matrix Transformation Concept**",
        "matrix_concept_content": "2D images are represented as sets of points (x, y). Linear transformations like translation, scaling, rotation, shearing, and reflection can be achieved through matrix multiplication.",
        "filter_concept": "📐 **Image Filtering Concept**",
        "filter_concept_content": "Filtering uses convolution kernels that slide over the image to produce effects like blur, sharpen, or edge detection through pixel-by-pixel mathematical operations.",
        "interactive_concept": "🕹️ **Why Interactive?**",
        "interactive_concept_content": "Interactivity allows users to directly see the effects of parameter changes, making mathematical concepts more intuitive and easier to understand.",
        "quick_concepts": "📖 **Quick Concepts**",
        "quick_concepts_text": "- 🔁 2D Transform: Change pixel positions\n- 🧮 Convolution: Sliding kernel for filtering\n- 🎯 Interactive: See changes in real-time",
        "upload_explanation_title": "📸 **Image Upload Guide**",
        "upload_explanation_text": "1. Click the 'Upload image' button below\n2. Select an image file (PNG/JPG/JPEG)\n3. Wait for the image to display\n4. Use processing tools as needed",
        "upload_title": "🖼️ **Upload Image**",
        "upload_label": "📤 Click to upload image (PNG/JPG/JPEG)",
        "upload_success": "✅ Image uploaded successfully!",
        "upload_preview": "🖼️ Original Image Preview",
        "upload_info": "⬆️ Please upload an image first to use the tools below.",
        "tools_title": "🛠️ **Image Processing Tools**",
        "tools_subtitle": "🎛️ Select transformation or filter to begin",
        "geo_title": "🔁 **Geometric Transformations**",
        "geo_desc": "Change position, size, and orientation of pixels using linear matrix operations.",
        "btn_translation": "↔️ Translation",
        "btn_scaling": "📏 Scaling",
        "btn_rotation": "🔄 Rotation",
        "btn_shearing": "📐 Shearing",
        "btn_reflection": "🪞 Reflection",
        "geo_info": "🔔 Please upload an image first to try transformations.",
        "trans_settings": "↔️ **Translation Settings**",
        "trans_dx": "dx (horizontal shift)",
        "trans_dy": "dy (vertical shift)",
        "btn_apply": "Apply",
        "trans_result": "**Translation Result**",
        "scale_settings": "📏 **Scaling Settings**",
        "scale_x": "Scale X",
        "scale_y": "Scale Y",
        "scale_result": "**Scaling Result**",
        "rot_settings": "🔄 **Rotation Settings**",
        "rot_angle": "Rotation angle (degrees)",
        "rot_result": "**Rotation Result**",
        "shear_settings": "📐 **Shearing Settings**",
        "shear_x": "Shear factor X",
        "shear_y": "Shear factor Y",
        "shear_result": "**Shearing Result**",
        "refl_settings": "🪞 **Reflection Settings**",
        "refl_axis": "Reflection axis",
        "refl_result": "**Reflection Result**",
        "hist_title": "📊 **Image Histogram**",
        "hist_desc": "Analyze pixel intensity distribution for brightness and contrast optimization.",
        "btn_histogram": "Show Histogram 📈",
        "hist_warning": "Please upload an image first to display the histogram.",
        "filter_title": "🧮 **Image Filtering**",
        "filter_desc": "Modify pixel values through convolution for various visual effects.",
        "btn_blur": "🔲 Blur",
        "btn_sharpen": "✨ Sharpen",
        "btn_background": "🎯 Background",
        "btn_grayscale": "⚫ Grayscale",
        "btn_edge": "🔍 Edge",
        "btn_brightness": "☀️ Brightness",
        "filter_info": "🔔 Please upload an image first to use the filters.",
        "blur_settings": "🔲 **Blur Filter Settings**",
        "blur_kernel": "Kernel size",
        "blur_result": "**Blur Result**",
        "sharpen_settings": "✨ **Sharpen Filter Settings**",
        "sharpen_desc": "Enhance details and edges in the image.",
        "sharpen_result": "**Sharpen Result**",
        "bg_settings": "🎯 **Background Removal Settings**",
        "bg_method": "Method",
        "bg_result": "**Background Removal Result**",
        "gray_settings": "⚫ **Grayscale Conversion Settings**",
        "gray_desc": "Convert the image to grayscale (black and white).",
        "gray_result": "**Grayscale Result**",
        "edge_settings": "🔍 **Edge Detection Settings**",
        "edge_method": "Edge method",
        "edge_result": "**Edge Image**",
        "bright_settings": "☀️ **Brightness & Contrast Settings**",
        "bright_brightness": "Brightness",
        "bright_contrast": "Contrast",
        "bright_result": "**Brightness/Contrast Result**",
        "team_title": "👥 **Team Members**",
        "team_subtitle": "Group 5 – Members and Roles",
        "team_sid": "SID",
        "team_role": "Role",
        "team_contribution": "Contribution",
        "team_group": "Group",
        "axis_x": "X-axis",
        "axis_y": "Y-axis",
        "axis_diag": "Diagonal",
        "dark_mode": "Dark Mode",
        "light_mode": "Light Mode",
        "menu_tools": "🛠️ Processing Tools",
        "menu_team": "👥 Our Team",
    }
}

# ===================== CUSTOM STYLING =====================

st.markdown("""
<style>
/* Main container styling */
.main-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    border: 2px solid rgba(76, 175, 80, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

/* Box styling */
.styled-box {
    background: linear-gradient(135deg, rgba(30, 30, 40, 0.95), rgba(40, 40, 60, 0.95));
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    border: 2px solid #66BB6A;
    box-shadow: 0 6px 20px rgba(102, 187, 106, 0.2);
    transition: all 0.3s ease;
    color: #ffffff;
}

.styled-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 25px rgba(102, 187, 106, 0.3);
}

.styled-box h3 {
    color: #81C784;
    border-bottom: 2px solid #66BB6A;
    padding-bottom: 10px;
    margin-bottom: 15px;
}

/* Light mode styling */
[data-theme="light"] .styled-box {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(240, 248, 255, 0.95));
    border-color: #4CAF50;
    color: #000000;
}

[data-theme="light"] .styled-box h3 {
    color: #2E7D32;
    border-bottom-color: #4CAF50;
}

/* Concept boxes */
.concept-box {
    background: linear-gradient(135deg, #1E3A1E, #2D5A2D);
    border-radius: 18px;
    padding: 20px;
    height: 100%;
    border: 2px solid #66BB6A;
    box-shadow: 0 4px 15px rgba(102, 187, 106, 0.15);
    color: #ffffff;
}

.concept-box h4 {
    color: #81C784;
    margin-bottom: 10px;
}

/* Light mode concept boxes */
[data-theme="light"] .concept-box {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border-color: #4CAF50;
    color: #000000;
}

[data-theme="light"] .concept-box h4 {
    color: #1B5E20;
}

/* Button styling */
.stButton > button {
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    border: 2px solid transparent !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3) !important;
}

/* Menu toggle styling */
.menu-toggle {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    justify-content: center;
}

.menu-btn {
    flex: 1;
    max-width: 200px;
}

.menu-btn.active {
    background: linear-gradient(135deg, #66BB6A, #4CAF50) !important;
    color: white !important;
    border-color: #2E7D32 !important;
}

/* Upload area styling */
.upload-area {
    background: linear-gradient(135deg, rgba(30, 58, 30, 0.9), rgba(45, 90, 45, 0.9));
    border-radius: 20px;
    padding: 30px;
    border: 3px dashed #66BB6A;
    text-align: center;
    color: #ffffff;
}

[data-theme="light"] .upload-area {
    background: linear-gradient(135deg, rgba(232, 245, 233, 0.9), rgba(200, 230, 201, 0.9));
    border-color: #4CAF50;
    color: #000000;
}

/* Team member cards */
.team-card {
    background: linear-gradient(135deg, #2E3A2E, #3D5A3D);
    border-radius: 18px;
    padding: 25px;
    border: 2px solid #66BB6A;
    box-shadow: 0 4px 15px rgba(102, 187, 106, 0.15);
    height: 100%;
    color: #ffffff;
    text-align: center;
}

[data-theme="light"] .team-card {
    background: linear-gradient(135deg, #F1F8E9, #DCEDC8);
    border-color: #4CAF50;
    color: #000000;
}

/* Updated team photo container - centered and larger */
.team-photo-container {
    width: 180px;
    height: 180px;
    border-radius: 20px;
    overflow: hidden;
    margin: 0 auto 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #66BB6A;
    border: 4px solid #4CAF50;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

[data-theme="light"] .team-photo-container {
    background: #4CAF50;
    border-color: #2E7D32;
}

.team-photo-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
}

/* Team info styling */
.team-info {
    text-align: center;
    margin-bottom: 15px;
}

.team-name {
    font-size: 1.4em;
    font-weight: bold;
    color: #81C784;
    margin-bottom: 10px;
}

[data-theme="light"] .team-name {
    color: #2E7D32;
}

.team-detail {
    font-size: 1em;
    margin-bottom: 8px;
    line-height: 1.5;
}

.team-detail strong {
    color: #FFD54F;
}

[data-theme="light"] .team-detail strong {
    color: #FF8F00;
}

/* Result display styling */
.result-box {
    background: linear-gradient(135deg, #2D3A2D, #1E2A1E);
    border-radius: 16px;
    padding: 20px;
    border: 2px solid #66BB6A;
    margin-top: 15px;
    margin-bottom: 20px;
    color: #ffffff;
}

[data-theme="light"] .result-box {
    background: linear-gradient(135deg, #FFFFFF, #F5F5F5);
    border-color: #4CAF50;
    color: #000000;
}

/* Slider styling */
.stSlider > div > div > div {
    background: #66BB6A !important;
}

[data-theme="light"] .stSlider > div > div > div {
    background: #4CAF50 !important;
}

/* Header styling */
.header-box {
    background: linear-gradient(135deg, rgba(30, 30, 40, 0.95), rgba(40, 40, 60, 0.95));
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 25px;
    border: 2px solid #66BB6A;
    box-shadow: 0 6px 20px rgba(102, 187, 106, 0.2);
    color: #ffffff;
}

[data-theme="light"] .header-box {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(240, 248, 255, 0.95));
    border-color: #4CAF50;
    color: #000000;
}

/* Quick concepts styling */
.quick-concepts-box {
    background: linear-gradient(135deg, #3A2E1E, #5A3D1E);
    border-radius: 18px;
    padding: 20px;
    border: 2px solid #FFB74D;
    box-shadow: 0 4px 15px rgba(255, 183, 77, 0.15);
    color: #ffffff;
}

[data-theme="light"] .quick-concepts-box {
    background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
    border-color: #FF9800;
    color: #000000;
}

/* Tools box styling */
.tools-box {
    background: linear-gradient(135deg, #1E2A3A, #2D3A5A);
    border-radius: 18px;
    padding: 25px;
    border: 2px solid #64B5F6;
    box-shadow: 0 4px 15px rgba(100, 181, 246, 0.15);
    color: #ffffff;
}

[data-theme="light"] .tools-box {
    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
    border-color: #2196F3;
    color: #000000;
}

/* Filter box styling */
.filter-box {
    background: linear-gradient(135deg, #2E1E3A, #3D1E5A);
    border-radius: 18px;
    padding: 25px;
    border: 2px solid #BA68C8;
    box-shadow: 0 4px 15px rgba(186, 104, 200, 0.15);
    color: #ffffff;
}

[data-theme="light"] .filter-box {
    background: linear-gradient(135deg, #F3E5F5, #E1BEE7);
    border-color: #9C27B0;
    color: #000000;
}

/* Histogram box styling */
.histogram-box {
    background: linear-gradient(135deg, #1E3A3A, #2D5A5A);
    border-radius: 18px;
    padding: 25px;
    border: 2px solid #4DD0E1;
    box-shadow: 0 4px 15px rgba(77, 208, 225, 0.15);
    color: #ffffff;
}

[data-theme="light"] .histogram-box {
    background: linear-gradient(135deg, #E0F7FA, #B2EBF2);
    border-color: #00BCD4;
    color: #000000;
}

/* Page toggle box styling */
.page-toggle-box {
    background: linear-gradient(135deg, rgba(40, 40, 50, 0.9), rgba(30, 30, 40, 0.9));
    border-radius: 18px;
    padding: 20px;
    margin: 20px 0;
    border: 2px solid #66BB6A;
    text-align: center;
    color: #ffffff;
}

[data-theme="light"] .page-toggle-box {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(245, 245, 245, 0.9));
    border-color: #4CAF50;
    color: #000000;
}

/* Explanation box styling */
.explanation-box {
    background: linear-gradient(135deg, #3A3A1E, #5A5A2D);
    border-radius: 18px;
    padding: 20px;
    border: 2px solid #FFD54F;
    box-shadow: 0 4px 15px rgba(255, 213, 79, 0.15);
    margin-bottom: 20px;
    color: #ffffff;
}

[data-theme="light"] .explanation-box {
    background: linear-gradient(135deg, #FFFDE7, #FFF9C4);
    border-color: #FFC107;
    color: #000000;
}

.explanation-box h4 {
    color: #FFD54F;
    margin-bottom: 10px;
}

[data-theme="light"] .explanation-box h4 {
    color: #FF8F00;
}

/* Title box styling */
.title-box {
    background: linear-gradient(135deg, #3A1E2A, #5A1E3D);
    border-radius: 18px;
    padding: 20px;
    border: 2px solid #EC407A;
    box-shadow: 0 4px 15px rgba(236, 64, 122, 0.15);
    margin-bottom: 20px;
    color: #ffffff;
}

[data-theme="light"] .title-box {
    background: linear-gradient(135deg, #FCE4EC, #F8BBD0);
    border-color: #E91E63;
    color: #000000;
}

.title-box h3 {
    color: #F48FB1;
    border-bottom: 2px solid #EC407A;
    padding-bottom: 10px;
    margin-bottom: 15px;
}

[data-theme="light"] .title-box h3 {
    color: #C2185B;
    border-bottom-color: #E91E63;
}

/* Method description styling */
.method-desc {
    background: linear-gradient(135deg, #2E3A2E, #3D5A3D);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #66BB6A;
    color: #ffffff;
}

[data-theme="light"] .method-desc {
    background: linear-gradient(135deg, #F1F8E9, #DCEDC8);
    border-left-color: #4CAF50;
    color: #000000;
}

/* Color tag styling */
.color-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: bold;
    margin-right: 8px;
    margin-bottom: 5px;
}

/* Settings box styling */
.settings-box {
    background: linear-gradient(135deg, #3A2E1E, #5A3D1E);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #FFB74D;
    color: #ffffff;
}

[data-theme="light"] .settings-box {
    background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
    border-color: #FF9800;
    color: #000000;
}

/* Image preview box */
.image-preview-box {
    background: linear-gradient(135deg, #2D3A2D, #1E2A1E);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #66BB6A;
    text-align: center;
    color: #ffffff;
}

[data-theme="light"] .image-preview-box {
    background: linear-gradient(135deg, #FFFFFF, #F5F5F5);
    border-color: #4CAF50;
    color: #000000;
}

/* Download buttons box */
.download-box {
    background: linear-gradient(135deg, #1E3A1E, #2D5A2D);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #66BB6A;
    color: #ffffff;
}

[data-theme="light"] .download-box {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border-color: #4CAF50;
    color: #000000;
}

/* Text content box */
.text-box {
    background: linear-gradient(135deg, #2A2A2A, #1E1E1E);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #616161;
    color: #ffffff;
}

[data-theme="light"] .text-box {
    background: linear-gradient(135deg, #F5F5F5, #EEEEEE);
    border-color: #BDBDBD;
    color: #000000;
}

/* Warning box */
.warning-box {
    background: linear-gradient(135deg, #3A1E1E, #5A1E1E);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #EF5350;
    color: #ffffff;
}

[data-theme="light"] .warning-box {
    background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
    border-color: #F44336;
    color: #000000;
}

/* Success box */
.success-box {
    background: linear-gradient(135deg, #1E3A1E, #2D5A2D);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #66BB6A;
    color: #ffffff;
}

[data-theme="light"] .success-box {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border-color: #4CAF50;
    color: #000000;
}

/* Info box */
.info-box {
    background: linear-gradient(135deg, #1E2A3A, #2D3A5A);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #64B5F6;
    color: #ffffff;
}

[data-theme="light"] .info-box {
    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
    border-color: #2196F3;
    color: #000000;
}

/* Apply consistent text color to all elements inside boxes */
.styled-box *,
.concept-box *,
.result-box *,
.upload-area *,
.team-card *,
.header-box *,
.quick-concepts-box *,
.tools-box *,
.filter-box *,
.histogram-box *,
.page-toggle-box *,
.explanation-box *,
.title-box *,
.method-desc *,
.settings-box *,
.image-preview-box *,
.download-box *,
.text-box *,
.warning-box *,
.success-box *,
.info-box * {
    color: inherit !important;
}

/* Ensure form elements have proper contrast */
.stSelectbox,
.stSlider,
.stNumberInput,
.stTextInput {
    color: #000000 !important;
}

[data-theme="dark"] .stSelectbox,
[data-theme="dark"] .stSlider,
[data-theme="dark"] .stNumberInput,
[data-theme="dark"] .stTextInput {
    color: #ffffff !important;
}

/* Team layout styling */
.team-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 25px;
    margin-top: 20px;
}

/* Responsive design for team cards */
@media (max-width: 768px) {
    .team-grid {
        grid-template-columns: 1fr;
    }
    
    .team-photo-container {
        width: 160px;
        height: 160px;
    }
}
</style>
""", unsafe_allow_html=True)

# ===================== HELPER FUNCTIONS =====================

def load_image(file):
    """Load image from uploaded file and convert to RGB numpy array."""
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)
    return img_np

def to_opencv(img_rgb):
    """Convert RGB numpy array to BGR for OpenCV."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def to_streamlit(img_bgr):
    """Convert BGR numpy array to RGB for Streamlit display."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_affine_transform(img_rgb, M, output_size=None):
    """Apply affine transformation to image."""
    img_bgr = to_opencv(img_rgb)
    h, w = img_bgr.shape[:2]
    if output_size is None:
        output_size = (w, h)

    if M.shape == (3, 3):
        M_affine = M[0:2, :]
    else:
        M_affine = M

    transformed = cv2.warpAffine(
        img_bgr, M_affine, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return to_streamlit(transformed)

def manual_convolution_gray(img_gray, kernel):
    """Apply manual convolution on grayscale image."""
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    img_float = img_gray.astype(np.float32)
    padded = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    h, w = img_gray.shape
    output = np.zeros((h, w), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)
    
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def manual_convolution_rgb(img_rgb, kernel):
    """Apply convolution to each RGB channel separately."""
    if img_rgb is None:
        return None
        
    channels = []
    for c in range(3):  # RGB channels
        channel = img_rgb[:, :, c]
        if channel.ndim > 2:
            channel = channel.squeeze()
        convolved = manual_convolution_gray(channel.astype(np.float32), kernel)
        channels.append(convolved)
    output = np.stack(channels, axis=-1).astype(np.uint8)
    return output

def rgb_to_gray(img_rgb):
    """Convert RGB to grayscale and return as 2D array."""
    if img_rgb is None:
        return None
        
    if img_rgb.ndim == 2:
        return img_rgb
    
    if img_rgb.dtype != np.float32:
        img_rgb = img_rgb.astype(np.float32)
    
    gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114])
    return np.clip(gray, 0, 255).astype(np.uint8)

def adjust_brightness_contrast(img_rgb, brightness=0, contrast=0):
    """Adjust brightness and contrast with input validation."""
    if img_rgb is None:
        return None
    
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)
    
    img_bgr = to_opencv(img_rgb)
    beta = brightness
    alpha = 1 + (contrast / 100.0)
    adjusted = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return to_streamlit(adjusted)

def image_to_bytes(img_array, fmt="PNG"):
    """Convert numpy image array to bytes for download."""
    if img_array is None:
        raise ValueError("image_to_bytes received None image")
    
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    if img_array.ndim == 2:
        pil_img = Image.fromarray(img_array, mode='L')
    elif img_array.ndim == 3:
        if img_array.shape[2] == 3:
            pil_img = Image.fromarray(img_array, mode='RGB')
        elif img_array.shape[2] == 4:
            pil_img = Image.fromarray(img_array, mode='RGBA')
        else:
            raise ValueError(f"Unexpected image shape: {img_array.shape}")
    else:
        raise ValueError(f"Unexpected image dimensions: {img_array.ndim}")
    
    buf = BytesIO()
    
    if fmt.upper() == "JPEG":
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            pil_img = pil_img.convert('RGB')
        elif img_array.ndim == 2:
            pil_img = pil_img.convert('RGB')
    
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def compute_histogram(img_rgb):
    """Compute and return histogram figure."""
    if img_rgb is None:
        return None
        
    img_bgr = to_opencv(img_rgb)
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, col in enumerate(color):
        hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
    ax.set_title("Color Histogram")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

def advanced_background_removal(img_rgb, method="hsv", bg_color=(255, 255, 255)):
    """Advanced background removal with multiple methods."""
    if img_rgb is None:
        return None
        
    img_bgr = to_opencv(img_rgb)
    
    # Create mask for green background (assuming green screen)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Define green color range
    lower_green1 = np.array([35, 50, 50])
    upper_green1 = np.array([85, 255, 255])
    
    # Alternative green range for better detection
    lower_green2 = np.array([25, 40, 40])
    upper_green2 = np.array([95, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Improve mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Invert mask to get foreground
    mask_inv = cv2.bitwise_not(mask)
    
    # Extract foreground
    fg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_inv)
    
    if method == "hsv":
        # White background
        bg = np.full_like(img_bgr, bg_color, dtype=np.uint8)
        bg = cv2.bitwise_and(bg, bg, mask=mask)
        result = cv2.add(fg, bg)
        
    elif method == "blur_bg":
        # Create blurred background
        blurred_bg = cv2.GaussianBlur(img_bgr, (51, 51), 0)
        bg = cv2.bitwise_and(blurred_bg, blurred_bg, mask=mask)
        result = cv2.add(fg, bg)
        
    elif method == "transparent":
        # Create RGBA image with transparency
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = mask_inv
        result = rgba
        
    elif method == "grayscale_bg":
        # Grayscale background
        gray_bg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_bg = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)
        bg = cv2.bitwise_and(gray_bg, gray_bg, mask=mask)
        result = cv2.add(fg, bg)
        
    elif method == "edge_detection_bg":
        # Edge detection background
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        bg = cv2.bitwise_and(edges_color, edges_color, mask=mask)
        result = cv2.add(fg, bg)
        
    else:
        # Solid color background
        if method == "red":
            bg_color_bgr = (0, 0, 255)
        elif method == "blue":
            bg_color_bgr = (255, 0, 0)
        elif method == "yellow":
            bg_color_bgr = (0, 255, 255)
        elif method == "green":
            bg_color_bgr = (0, 255, 0)
        elif method == "purple":
            bg_color_bgr = (255, 0, 255)
        elif method == "orange":
            bg_color_bgr = (0, 165, 255)
        elif method == "pink":
            bg_color_bgr = (203, 192, 255)
        elif method == "brown":
            bg_color_bgr = (42, 42, 165)
        elif method == "black":
            bg_color_bgr = (0, 0, 0)
        else:
            bg_color_bgr = (255, 255, 255)
        
        bg = np.full_like(img_bgr, bg_color_bgr, dtype=np.uint8)
        bg = cv2.bitwise_and(bg, bg, mask=mask)
        result = cv2.add(fg, bg)
    
    if result.shape[2] == 4:
        return result
    else:
        return to_streamlit(result)

def get_method_description(method_name):
    """Get description for each background removal method."""
    descriptions = {
        "HSV Color Thresholding": "Menggunakan ruang warna HSV untuk mendeteksi dan menghapus background hijau.",
        "Blur Background": "Membuat background blur sementara foreground tetap tajam.",
        "Remove Background (Transparent)": "Menghapus background dan membuatnya transparan (format PNG).",
        "Grayscale Background": "Mengubah background menjadi grayscale sementara foreground tetap berwarna.",
        "Edge Detection Background": "Mengubah background menjadi deteksi tepi untuk efek artistik.",
        "Solid Red Background": "Mengganti background dengan warna merah solid.",
        "Solid Blue Background": "Mengganti background dengan warna biru solid.",
        "Solid Yellow Background": "Mengganti background dengan warna kuning solid.",
        "Solid Green Background": "Mengganti background dengan warna hijau solid.",
        "Solid Purple Background": "Mengganti background dengan warna ungu solid.",
        "Solid Orange Background": "Mengganti background dengan warna oranye solid.",
        "Solid Pink Background": "Mengganti background dengan warna pink solid.",
        "Solid Brown Background": "Mengganti background dengan warna coklat solid.",
        "Solid Black Background": "Mengganti background dengan warna hitam solid.",
        "Solid White Background": "Mengganti background dengan warna putih solid."
    }
    
    return descriptions.get(method_name, "Metode penghapusan background.")

def get_method_color(method_name):
    """Get color tag for each method."""
    colors = {
        "HSV Color Thresholding": "#4CAF50",
        "Blur Background": "#2196F3",
        "Remove Background (Transparent)": "#9C27B0",
        "Grayscale Background": "#607D8B",
        "Edge Detection Background": "#FF5722",
        "Solid Red Background": "#F44336",
        "Solid Blue Background": "#2196F3",
        "Solid Yellow Background": "#FFEB3B",
        "Solid Green Background": "#4CAF50",
        "Solid Purple Background": "#9C27B0",
        "Solid Orange Background": "#FF9800",
        "Solid Pink Background": "#E91E63",
        "Solid Brown Background": "#795548",
        "Solid Black Background": "#000000",
        "Solid White Background": "#FFFFFF"
    }
    
    text_color = "#FFFFFF" if method_name in ["Solid Black Background"] else "#000000"
    
    return colors.get(method_name, "#9E9E9E"), text_color

def safe_display_square_image(path):
    """Display image in square format with proper cropping."""
    if os.path.exists(path):
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img_cropped = img.crop((left, top, right, bottom))
            img_resized = img_cropped.resize((180, 180), Image.Resampling.LANCZOS)

            buffered = BytesIO()
            img_resized.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            st.markdown(f"""
            <div class="team-photo-container">
                <img src="data:image/jpeg;base64,{img_str}" alt="Team member"/>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
    else:
        st.markdown("""
        <div class="team-photo-container">
            <div style="width:100%; height:100%; display:flex; align-items:center; justify-content:center; background:#ddd; color:#666;">
                No Image
            </div>
        </div>
        """, unsafe_allow_html=True)

# Ensure images directory exists
os.makedirs("images", exist_ok=True)

# Get current language and theme
lang = st.session_state["language"]
t = translations[lang]
theme_mode = st.session_state["theme_mode"]

# ===================== HEADER BOX =====================
with st.container():
    st.markdown('<div class="header-box">', unsafe_allow_html=True)
    
    header_col1, header_col2, header_col3 = st.columns([6, 1, 1], vertical_alignment="center")
    
    with header_col1:
        st.markdown('<div class="text-box">', unsafe_allow_html=True)
        st.title(t["title"])
        st.markdown(f'<p style="color: #ffffff; font-size: 1.1em;">{t["subtitle"]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with header_col2:
        lang_button_text = "🇬🇧 EN" if lang == "id" else "🇮🇩 ID"
        if st.button(lang_button_text, key="lang_toggle", use_container_width=True, 
                    help="Switch language"):
            st.session_state["language"] = "en" if lang == "id" else "id"
            st.rerun()
    
    with header_col3:
        theme_button_text = "🌙" if theme_mode == "light" else "☀️"
        if st.button(theme_button_text, key="theme_toggle", use_container_width=True,
                    help="Toggle dark/light mode"):
            st.session_state["theme_mode"] = "dark" if theme_mode == "light" else "light"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== CONCEPT BOXES =====================

# Tujuan Aplikasi Box
st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
st.markdown(f'### {t["app_goal"]}')
st.markdown(f'<div class="text-box">{t["app_goal_content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Two-column layout for concept boxes
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
    st.markdown(f'### {t["matrix_concept"]}')
    st.markdown(f'<div class="text-box">{t["matrix_concept_content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
    st.markdown(f'### {t["interactive_concept"]}')
    st.markdown(f'<div class="text-box">{t["interactive_concept_content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Konsep Singkat Box
st.markdown('<div class="quick-concepts-box">', unsafe_allow_html=True)
st.markdown(f'### {t["quick_concepts"]}')
st.markdown(f'<div class="text-box">{t["quick_concepts_text"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ===================== PAGE TOGGLE BOX =====================
st.markdown('<div class="page-toggle-box">', unsafe_allow_html=True)
toggle_col1, toggle_col2 = st.columns([1, 1], gap="medium")

with toggle_col1:
    if st.button(t["menu_tools"], key="menu_tools_btn", 
                 type="primary" if st.session_state["current_page"] == "tools" else "secondary",
                 use_container_width=True):
        st.session_state["current_page"] = "tools"
        st.rerun()

with toggle_col2:
    if st.button(t["menu_team"], key="menu_team_btn",
                 type="primary" if st.session_state["current_page"] == "team" else "secondary",
                 use_container_width=True):
        st.session_state["current_page"] = "team"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ===================== PAGE CONTENT =====================
page = st.session_state["current_page"]

if page == "tools":
    # ===================== UPLOAD GUIDE BOX =====================
    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
    st.markdown(f'### {t["upload_explanation_title"]}')
    st.markdown(f'<div class="text-box">{t["upload_explanation_text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===================== UPLOAD IMAGE BOX =====================
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    st.markdown(f'### {t["upload_title"]}')
    uploaded_file = st.file_uploader(
        label=t["upload_label"],
        type=["png", "jpg", "jpeg"],
        key="image_uploader"
    )
    
    if uploaded_file is not None:
        original_img = load_image(uploaded_file)
        st.session_state.original_img = original_img
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success(t["upload_success"])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
        st.image(
            original_img,
            caption=t["upload_preview"],
            use_column_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info(t["upload_info"])
        st.markdown('</div>', unsafe_allow_html=True)
        original_img = None
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        original_img = st.session_state.original_img
        
        # ===================== TOOLS TITLE BOX =====================
        st.markdown('<div class="title-box">', unsafe_allow_html=True)
        st.markdown(f'### {t["tools_title"]}')
        st.markdown(f'<div class="text-box">{t["tools_subtitle"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Main tools columns
        tools_col_left, tools_col_right = st.columns(2, gap="large")
        
        # ===================== GEOMETRIC TRANSFORMATIONS BOX =====================
        with tools_col_left:
            st.markdown('<div class="tools-box">', unsafe_allow_html=True)
            st.markdown(f'### {t["geo_title"]}')
            st.markdown(f'<div class="text-box">{t["geo_desc"]}</div>', unsafe_allow_html=True)
            
            # Transformation buttons
            trans_col1, trans_col2, trans_col3 = st.columns(3)
            with trans_col1:
                if st.button(t["btn_translation"], key="btn_trans_click", type="secondary", use_container_width=True):
                    st.session_state["geo_transform"] = "translation"
            with trans_col2:
                if st.button(t["btn_scaling"], key="btn_scale_click", type="secondary", use_container_width=True):
                    st.session_state["geo_transform"] = "scaling"
            with trans_col3:
                if st.button(t["btn_rotation"], key="btn_rot_click", type="secondary", use_container_width=True):
                    st.session_state["geo_transform"] = "rotation"
            
            trans_col4, trans_col5, _ = st.columns(3)
            with trans_col4:
                if st.button(t["btn_shearing"], key="btn_shear_click", type="secondary", use_container_width=True):
                    st.session_state["geo_transform"] = "shearing"
            with trans_col5:
                if st.button(t["btn_reflection"], key="btn_refl_click", type="secondary", use_container_width=True):
                    st.session_state["geo_transform"] = "reflection"
            
            # Transform parameter panel
            if st.session_state["geo_transform"] == "translation":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["trans_settings"]}</div>', unsafe_allow_html=True)
                dx = st.slider(t["trans_dx"], -200, 200, 0, key="trans_dx")
                dy = st.slider(t["trans_dy"], -200, 200, 0, key="trans_dy")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_trans", type="primary", use_container_width=True):
                    T = np.array([[1, 0, dx],
                                  [0, 1, dy],
                                  [0, 0, 1]], dtype=np.float32)
                    translated_img = apply_affine_transform(original_img, T)
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(translated_img, caption=t["trans_result"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(translated_img, fmt="PNG"),
                            file_name="translation_result.png",
                            mime="image/png",
                            key="dl_trans_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(translated_img, fmt="JPEG"),
                            file_name="translation_result.jpg",
                            mime="image/jpeg",
                            key="dl_trans_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif st.session_state["geo_transform"] == "scaling":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["scale_settings"]}</div>', unsafe_allow_html=True)
                sx = st.slider(t["scale_x"], 0.1, 3.0, 1.0, key="scale_x")
                sy = st.slider(t["scale_y"], 0.1, 3.0, 1.0, key="scale_y")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_scale", type="primary", use_container_width=True):
                    h, w = original_img.shape[:2]
                    S = np.array([[sx, 0, 0],
                                  [0, sy, 0],
                                  [0, 0, 1]], dtype=np.float32)
                    new_w = int(w * sx)
                    new_h = int(h * sy)
                    scaled_img = apply_affine_transform(original_img, S, output_size=(new_w, new_h))
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(scaled_img, caption=t["scale_result"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(scaled_img, fmt="PNG"),
                            file_name="scaling_result.png",
                            mime="image/png",
                            key="dl_scale_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(scaled_img, fmt="JPEG"),
                            file_name="scaling_result.jpg",
                            mime="image/jpeg",
                            key="dl_scale_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif st.session_state["geo_transform"] == "rotation":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["rot_settings"]}</div>', unsafe_allow_html=True)
                angle = st.slider(t["rot_angle"], -180, 180, 0, key="rot_angle")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_rot", type="primary", use_container_width=True):
                    h, w = original_img.shape[:2]
                    cx, cy = w / 2, h / 2
                    theta = np.deg2rad(angle)
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    R = np.array([[cos_t, -sin_t, 0],
                                  [sin_t,  cos_t, 0],
                                  [0,      0,     1]], dtype=np.float32)
                    T1 = np.array([[1, 0, -cx],
                                   [0, 1, -cy],
                                   [0, 0, 1]], dtype=np.float32)
                    T2 = np.array([[1, 0, cx],
                                   [0, 1, cy],
                                   [0, 0, 1]], dtype=np.float32)
                    M = T2 @ R @ T1
                    rotated_img = apply_affine_transform(original_img, M)
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(rotated_img, caption=t["rot_result"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(rotated_img, fmt="PNG"),
                            file_name="rotation_result.png",
                            mime="image/png",
                            key="dl_rot_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(rotated_img, fmt="JPEG"),
                            file_name="rotation_result.jpg",
                            mime="image/jpeg",
                            key="dl_rot_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif st.session_state["geo_transform"] == "shearing":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["shear_settings"]}</div>', unsafe_allow_html=True)
                shear_x = st.slider(t["shear_x"], -1.0, 1.0, 0.0, key="shear_x")
                shear_y = st.slider(t["shear_y"], -1.0, 1.0, 0.0, key="shear_y")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_shear", type="primary", use_container_width=True):
                    Sh = np.array([[1,      shear_x, 0],
                                   [shear_y, 1,      0],
                                   [0,      0,      1]], dtype=np.float32)
                    sheared_img = apply_affine_transform(original_img, Sh)
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(sheared_img, caption=t["shear_result"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(sheared_img, fmt="PNG"),
                            file_name="shearing_result.png",
                            mime="image/png",
                            key="dl_shear_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(sheared_img, fmt="JPEG"),
                            file_name="shearing_result.jpg",
                            mime="image/jpeg",
                            key="dl_shear_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif st.session_state["geo_transform"] == "reflection":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["refl_settings"]}</div>', unsafe_allow_html=True)
                axis = st.selectbox(t["refl_axis"], [t["axis_x"], t["axis_y"], t["axis_diag"]], key="refl_axis")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_refl", type="primary", use_container_width=True):
                    h, w = original_img.shape[:2]
                    if axis == t["axis_x"]:
                        Rf = np.array([[1, 0, 0],
                                       [0, -1, h],
                                       [0, 0, 1]], dtype=np.float32)
                    elif axis == t["axis_y"]:
                        Rf = np.array([[-1, 0, w],
                                       [0, 1, 0],
                                       [0, 0, 1]], dtype=np.float32)
                    else:
                        Rf = np.array([[0, 1, 0],
                                       [1, 0, 0],
                                       [0, 0, 1]], dtype=np.float32)
                    reflected_img = apply_affine_transform(original_img, Rf)
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(reflected_img, caption=t["refl_result"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(reflected_img, fmt="PNG"),
                            file_name="reflection_result.png",
                            mime="image/png",
                            key="dl_refl_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(reflected_img, fmt="JPEG"),
                            file_name="reflection_result.jpg",
                            mime="image/jpeg",
                            key="dl_refl_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close tools-box
        
        # ===================== FILTERING GAMBAR BOX =====================
        with tools_col_right:
            st.markdown('<div class="filter-box">', unsafe_allow_html=True)
            st.markdown(f'### {t["filter_title"]}')
            st.markdown(f'<div class="text-box">{t["filter_desc"]}</div>', unsafe_allow_html=True)
            
            # Filter buttons
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                if st.button(t["btn_blur"], key="btn_blur_click", type="secondary", use_container_width=True):
                    st.session_state["image_filter"] = "blur"
            with filter_col2:
                if st.button(t["btn_sharpen"], key="btn_sharpen_click", type="secondary", use_container_width=True):
                    st.session_state["image_filter"] = "sharpen"
            with filter_col3:
                if st.button(t["btn_background"], key="btn_bg_click", type="secondary", use_container_width=True):
                    st.session_state["image_filter"] = "background"
            
            filter_col4, filter_col5, filter_col6 = st.columns(3)
            with filter_col4:
                if st.button(t["btn_grayscale"], key="btn_gray_click", type="secondary", use_container_width=True):
                    st.session_state["image_filter"] = "grayscale"
            with filter_col5:
                if st.button(t["btn_edge"], key="btn_edge_click", type="secondary", use_container_width=True):
                    st.session_state["image_filter"] = "edge"
            with filter_col6:
                if st.button(t["btn_brightness"], key="btn_bright_click", type="secondary", use_container_width=True):
                    st.session_state["image_filter"] = "brightness"
            
            # Filter parameter panel
            if st.session_state["image_filter"] == "blur":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["blur_settings"]}</div>', unsafe_allow_html=True)
                kernel_size = st.selectbox(
                    t["blur_kernel"],
                    [3, 5, 7],
                    index=0,
                    key="blur_kernel_size"
                )
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_blur", type="primary", use_container_width=True):
                    k = kernel_size
                    blur_kernel = np.ones((k, k), dtype=np.float32) / (k * k)
                    blurred_rgb = manual_convolution_rgb(original_img, blur_kernel)
                    
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(blurred_rgb, caption=t["blur_result"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(blurred_rgb, fmt="PNG"),
                            file_name="blur_result.png",
                            mime="image/png",
                            key="dl_blur_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(blurred_rgb, fmt="JPEG"),
                            file_name="blur_result.jpg",
                            mime="image/jpeg",
                            key="dl_blur_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif st.session_state["image_filter"] == "sharpen":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["sharpen_settings"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["sharpen_desc"]}</div>', unsafe_allow_html=True)
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_sharpen", type="primary", use_container_width=True):
                    gray = rgb_to_gray(original_img)
                    sharpen_kernel = np.array(
                        [[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]],
                        dtype=np.float32
                    )
                    sharpened_gray = manual_convolution_gray(gray, sharpen_kernel)
                    
                    if sharpened_gray.ndim == 2:
                        sharpened_rgb = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2RGB)
                    else:
                        sharpened_rgb = sharpened_gray
                    
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(sharpened_rgb, caption=t["sharpen_result"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(sharpened_rgb, fmt="PNG"),
                            file_name="sharpen_result.png",
                            mime="image/png",
                            key="dl_sharp_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(sharpened_rgb, fmt="JPEG"),
                            file_name="sharpen_result.jpg",
                            mime="image/jpeg",
                            key="dl_sharp_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif st.session_state["image_filter"] == "background":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["bg_settings"]}</div>', unsafe_allow_html=True)
                
                # Background removal methods
                bg_methods = [
                    "HSV Color Thresholding",
                    "Blur Background",
                    "Remove Background (Transparent)",
                    "Grayscale Background",
                    "Edge Detection Background",
                    "Solid Red Background",
                    "Solid Blue Background",
                    "Solid Yellow Background",
                    "Solid Green Background",
                    "Solid Purple Background",
                    "Solid Orange Background",
                    "Solid Pink Background",
                    "Solid Brown Background",
                    "Solid Black Background",
                    "Solid White Background"
                ]
                
                method = st.selectbox(
                    t["bg_method"],
                    bg_methods,
                    key="bg_method"
                )
                
                # Show method description
                bg_color, text_color = get_method_color(method)
                method_desc = get_method_description(method)
                
                st.markdown(f"""
                <div class="method-desc">
                    <span class="color-tag" style="background-color: {bg_color}; color: {text_color};">
                        {method}
                    </span>
                    <br>
                    {method_desc}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.info("⚠️ **Catatan:** Untuk hasil terbaik, gunakan gambar dengan background hijau (green screen).")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_bg", type="primary", use_container_width=True):
                    # Map method name to method type
                    method_map = {
                        "HSV Color Thresholding": "hsv",
                        "Blur Background": "blur_bg",
                        "Remove Background (Transparent)": "transparent",
                        "Grayscale Background": "grayscale_bg",
                        "Edge Detection Background": "edge_detection_bg",
                        "Solid Red Background": "red",
                        "Solid Blue Background": "blue",
                        "Solid Yellow Background": "yellow",
                        "Solid Green Background": "green",
                        "Solid Purple Background": "purple",
                        "Solid Orange Background": "orange",
                        "Solid Pink Background": "pink",
                        "Solid Brown Background": "brown",
                        "Solid Black Background": "black",
                        "Solid White Background": "white"
                    }
                    
                    method_type = method_map.get(method, "hsv")
                    
                    with st.spinner(f"Memproses {method}..."):
                        bg_removed_img = advanced_background_removal(original_img, method_type, (255, 255, 255))
                    
                    if bg_removed_img is not None:
                        # Check if image is RGBA (transparent)
                        if bg_removed_img.shape[2] == 4:
                            # Convert RGBA to RGB for display
                            bg_display = cv2.cvtColor(bg_removed_img, cv2.COLOR_RGBA2RGB)
                            st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                            st.image(bg_display, caption=f"{t['bg_result']} - {method}", use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                            st.image(bg_removed_img, caption=f"{t['bg_result']} - {method}", use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="download-box">', unsafe_allow_html=True)
                        col_png, col_jpg = st.columns(2)
                        with col_png:
                            st.download_button(
                                label="⬇️ Download PNG",
                                data=image_to_bytes(bg_removed_img, fmt="PNG"),
                                file_name=f"background_{method.lower().replace(' ', '_')}.png",
                                mime="image/png",
                                key="dl_bg_png",
                                use_container_width=True
                            )
                        with col_jpg:
                            # For RGBA images, convert to RGB for JPG
                            if bg_removed_img.shape[2] == 4:
                                bg_rgb = cv2.cvtColor(bg_removed_img, cv2.COLOR_RGBA2RGB)
                                st.download_button(
                                    label="⬇️ Download JPG",
                                    data=image_to_bytes(bg_rgb, fmt="JPEG"),
                                    file_name=f"background_{method.lower().replace(' ', '_')}.jpg",
                                    mime="image/jpeg",
                                    key="dl_bg_jpg",
                                    use_container_width=True
                                )
                            else:
                                st.download_button(
                                    label="⬇️ Download JPG",
                                    data=image_to_bytes(bg_removed_img, fmt="JPEG"),
                                    file_name=f"background_{method.lower().replace(' ', '_')}.jpg",
                                    mime="image/jpeg",
                                    key="dl_bg_jpg",
                                    use_container_width=True
                                )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.error("Gagal memproses penghapusan background. Pastikan gambar memiliki background hijau untuk hasil terbaik.")
                        st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif st.session_state["image_filter"] == "grayscale":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["gray_settings"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["gray_desc"]}</div>', unsafe_allow_html=True)
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_gray", type="primary", use_container_width=True):
                    gray_img = rgb_to_gray(original_img)
                    if gray_img.ndim == 2:
                        gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
                    else:
                        gray_rgb = gray_img
                    
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(gray_rgb, caption=t["gray_result"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(gray_rgb, fmt="PNG"),
                            file_name="grayscale_result.png",
                            mime="image/png",
                            key="dl_gray_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(gray_rgb, fmt="JPEG"),
                            file_name="grayscale_result.jpg",
                            mime="image/jpeg",
                            key="dl_gray_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif st.session_state["image_filter"] == "edge":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["edge_settings"]}</div>', unsafe_allow_html=True)
                method_edge = st.selectbox(
                    t["edge_method"], ["Sobel", "Canny"], key="edge_method"
                )
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_edge", type="primary", use_container_width=True):
                    img_bgr = to_opencv(original_img)
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    if method_edge == "Sobel":
                        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                        mag = cv2.magnitude(grad_x, grad_y)
                        mag = np.clip(mag, 0, 255).astype(np.uint8)
                        edge_bgr = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
                    else:
                        edges = cv2.Canny(gray, 100, 200)
                        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    edge_img = to_streamlit(edge_bgr)
                    
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(edge_img, caption=f"{t['edge_result']} ({method_edge})", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(edge_img, fmt="PNG"),
                            file_name="edge_result.png",
                            mime="image/png",
                            key="dl_edge_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(edge_img, fmt="JPEG"),
                            file_name="edge_result.jpg",
                            mime="image/jpeg",
                            key="dl_edge_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif st.session_state["image_filter"] == "brightness":
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="text-box">{t["bright_settings"]}</div>', unsafe_allow_html=True)
                brightness = st.slider(t["bright_brightness"], -100, 100, 0, key="brightness_value")
                contrast = st.slider(t["bright_contrast"], -100, 100, 0, key="contrast_value")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_bright", type="primary", use_container_width=True):
                    adjusted_img = adjust_brightness_contrast(original_img, brightness, contrast)
                    
                    st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                    st.image(adjusted_img, caption=t["bright_result"], use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="download-box">', unsafe_allow_html=True)
                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(adjusted_img, fmt="PNG"),
                            file_name="brightness_contrast_result.png",
                            mime="image/png",
                            key="dl_bright_png",
                            use_container_width=True
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(adjusted_img, fmt="JPEG"),
                            file_name="brightness_contrast_result.jpg",
                            mime="image/jpeg",
                            key="dl_bright_jpg",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close filter-box
        
        # ===================== HISTOGRAM BOX =====================
        st.markdown('<div class="histogram-box" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown(f'### {t["hist_title"]}')
        st.markdown(f'<div class="text-box">{t["hist_desc"]}</div>', unsafe_allow_html=True)
        if st.button(t["btn_histogram"], key="btn_histogram", type="secondary", use_container_width=True):
            if original_img is not None:
                hist_fig = compute_histogram(original_img)
                st.markdown('<div class="image-preview-box">', unsafe_allow_html=True)
                st.pyplot(hist_fig)
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(hist_fig)
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning(t["hist_warning"])
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ===================== TEAM MEMBERS PAGE =====================
elif page == "team":
    st.markdown('<div class="title-box">', unsafe_allow_html=True)
    st.markdown(f'### {t["team_title"]}')
    st.markdown(f'<div class="text-box">{t["team_subtitle"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    members = [
        {
            "img": "images/aditya.jpg", 
            "name": "ADITYA ANGGARA PAMUNGKAS", 
            "sid": "04202400051", 
            "role": "Leader", 
            "contribution": "Project Manager, Geometric Transformations Module"
        },
        {
            "img": "images/maula_aqiel.jpg", 
            "name": "MAULA AQIEL NURI", 
            "sid": "04202400023", 
            "role": "Member", 
            "contribution": "Image Filtering Module, UI/UX Design"
        },
        {
            "img": "images/syafiq_nur.jpg", 
            "name": "SYAFIQ NUR RAMADHAN", 
            "sid": "04202400073", 
            "role": "Member", 
            "contribution": "Background Removal Module, Image Upload & Download"
        },
        {
            "img": "images/rifat_fitrotu.jpg", 
            "name": "RIFAT FITROTU SALMAN", 
            "sid": "04202400106", 
            "role": "Member", 
            "contribution": "Histogram Module, Image Processing Functions"
        },
    ]
    
    # Use grid layout for team members
    cols = st.columns(2, gap="large")
    
    for idx, member in enumerate(members):
        with cols[idx % 2]:
            st.markdown('<div class="team-card">', unsafe_allow_html=True)
            
            # Center photo
            safe_display_square_image(member["img"])
            
            # Member info with clean formatting
            st.markdown(f'<div class="team-name">{member["name"]}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="team-info">', unsafe_allow_html=True)
            st.markdown(f'<div class="team-detail"><strong>{t["team_sid"]}:</strong> {member["sid"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-detail"><strong>{t["team_role"]}:</strong> {member["role"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-detail"><strong>{t["team_group"]}:</strong> 5</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-detail"><strong>{t["team_contribution"]}:</strong> {member["contribution"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("""
<hr style="border: 1px solid #4CAF50; margin: 30px 0;">
<div style="text-align: center; color: #ffffff; font-size: 0.9em; padding: 20px;">
    <strong>© 2025 Matrix Transformations in Image Processing</strong><br>
    <span style="color: #66BB6A;">Kelompok 5 - Linear Algebra</span><br>
    <small>Industrial Engineering - President University</small>
</div>
""", unsafe_allow_html=True)
