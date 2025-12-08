import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from io import BytesIO  # <- untuk download image

# ===================== CONFIG & THEME =====================

st.set_page_config(
    page_title="🧮 Matrix Transformations in Image Processing",
    layout="wide"
)

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

# ===================== TRANSLATIONS =====================

translations = {
    "id": {
        "title": "🧮 Transformasi Matriks dalam Pengolahan Citra",
        "subtitle": "📌 Demo interaktif transformasi matriks 2D dan filtering citra",
        "app_goal": "🎯 **Tujuan aplikasi:** Demo interaktif yang menunjukkan bagaimana transformasi matriks 2D dan filtering citra bekerja pada gambar menggunakan konsep aljabar linier.",
        "features": "- 🔁 Transformasi: translasi, scaling, rotasi, shearing, refleksi\n- 🧹 Filtering: blur, sharpen, deteksi tepi, penghapusan background, grayscale, brightness & contrast",
        "concept_1_title": "### 🧊 Transformasi matriks 2D",
        "concept_1_text1": "Sebuah citra 2D dapat dilihat sebagai kumpulan titik \\((x, y)\\) yang dapat dipindahkan menggunakan operasi linear seperti translasi, scaling, rotasi, shearing, dan refleksi, yang direpresentasikan oleh matriks 2×2 atau 3×3 (koordinat homogen).",
        "concept_1_text2": "Mengalikan koordinat piksel dengan matriks-matriks ini mengubah posisinya: scaling mengubah ukuran, rotasi memutar gambar di sekitar pusat, shearing membuat bentuk miring, dan refleksi membalik gambar pada sumbu yang dipilih.",
        "concept_2_title": "### 📐 Filtering citra (konvolusi)",
        "concept_2_text1": "Filtering menggunakan kernel kecil (matriks konvolusi) yang digeser pada gambar; di setiap posisi menghasilkan nilai piksel baru dari jumlah produk antara kernel dan piksel tetangganya.",
        "concept_2_text2": "Kernel dengan nilai positif seragam menghaluskan atau mengaburkan gambar, sementara kernel dengan pusat positif kuat dan tetangga negatif mempertajam dan menekankan tepi.",
        "concept_3_title": "### 🕹️ Mengapa interaktif?",
        "concept_3_text1": "Ini interaktif karena pengguna dapat mengubah parameter (sudut rotasi, faktor skala, nilai shear, tipe kernel blur/sharpen, dll.) dan langsung melihat efeknya pada gambar. ✨",
        "concept_3_text2": "Ini membantu menghubungkan struktur matriks atau kernel dengan efek visualnya, membuat konsep seperti transformasi linear dan konvolusi lebih intuitif.",
        "quick_concepts": "#### 📖 Konsep singkat",
        "quick_concepts_text": "- 🔁 Transformasi 2D: mengubah posisi piksel (translasi, scaling, rotasi, shearing, refleksi).\n- 🧮 Konvolusi: kernel kecil yang digeser pada gambar untuk menghasilkan nilai piksel baru.",
        "upload_title": "### 🖼️ Unggah Gambar",
        "upload_label": "Unggah gambar di sini (PNG/JPG/JPEG) 📂",
        "upload_success": "✅ Gambar berhasil diunggah!",
        "upload_preview": "🖼️ Pratinjau Gambar Asli",
        "upload_info": "⬆️ Silakan unggah gambar terlebih dahulu untuk menggunakan alat di bawah ini.",
        "tools_title": "### 🛠️ Alat Pengolahan Citra",
        "tools_subtitle": "🎛️ Pilih kotak di bawah untuk membuka pengaturan transformasi atau filter.",
        "geo_title": "#### 🔁 Transformasi Geometri",
        "geo_desc": "Transformasi geometri mengubah posisi, ukuran, dan orientasi piksel menggunakan operasi linear (matriks).",
        "btn_translation": "↔️ Translasi",
        "btn_scaling": "📏 Scaling",
        "btn_rotation": "🔄 Rotasi",
        "btn_shearing": "📐 Shearing",
        "btn_reflection": "🪞 Refleksi",
        "geo_info": "🔔 Silakan unggah gambar terlebih dahulu untuk mencoba transformasi.",
        "trans_settings": "**↔️ Pengaturan Translasi**",
        "trans_dx": "dx (pergeseran horizontal)",
        "trans_dy": "dy (pergeseran vertikal)",
        "btn_apply": "Terapkan",
        "trans_result": "Hasil Translasi",
        "scale_settings": "**📏 Pengaturan Scaling**",
        "scale_x": "Skala X",
        "scale_y": "Skala Y",
        "scale_result": "Hasil Scaling",
        "rot_settings": "**🔄 Pengaturan Rotasi**",
        "rot_angle": "Sudut rotasi (derajat)",
        "rot_result": "Hasil Rotasi",
        "shear_settings": "**📐 Pengaturan Shearing**",
        "shear_x": "Faktor shear X",
        "shear_y": "Faktor shear Y",
        "shear_result": "Hasil Shearing",
        "refl_settings": "**🪞 Pengaturan Refleksi**",
        "refl_axis": "Sumbu refleksi",
        "refl_result": "Hasil Refleksi",
        "hist_title": "#### 📊 Histogram Citra",
        "hist_desc": "Histogram menunjukkan distribusi intensitas piksel (gelap–terang), berguna untuk menganalisis brightness dan kontras.",
        "btn_histogram": "Tampilkan Histogram 📈",
        "hist_warning": "Silakan unggah gambar terlebih dahulu untuk menampilkan histogram.",
        "filter_title": "#### 🧮 Filtering Citra",
        "filter_desc": "Filtering mengubah nilai piksel berdasarkan tetangganya (konvolusi) untuk melakukan blur, sharpen, deteksi tepi, penghapusan background, dan lainnya.",
        "btn_blur": "🔲 Blur",
        "btn_sharpen": "✨ Sharpen",
        "btn_background": "🎯 Background",
        "btn_grayscale": "⚫ Grayscale",
        "btn_edge": "🔍 Edge",
        "btn_brightness": "☀️ Brightness",
        "filter_info": "🔔 Silakan unggah gambar terlebih dahulu untuk menggunakan filter.",
        "blur_settings": "**🔲 Pengaturan Filter Blur**",
        "blur_kernel": "Ukuran kernel",
        "blur_result": "Hasil Blur",
        "sharpen_settings": "**✨ Pengaturan Filter Sharpen**",
        "sharpen_desc": "Tingkatkan detail dan tepi pada gambar.",
        "sharpen_result": "Hasil Sharpen",
        "bg_settings": "**🎯 Pengaturan Penghapusan Background**",
        "bg_method": "Metode (demo menggunakan HSV saja untuk saat ini)",
        "bg_result": "Hasil Penghapusan Background",
        "gray_settings": "**⚫ Pengaturan Konversi Grayscale**",
        "gray_desc": "Konversi gambar ke grayscale (hitam putih).",
        "gray_result": "Hasil Grayscale",
        "edge_settings": "**🔍 Pengaturan Deteksi Tepi**",
        "edge_method": "Metode edge",
        "edge_result": "Gambar Edge",
        "bright_settings": "**☀️ Pengaturan Brightness & Kontras**",
        "bright_brightness": "Brightness",
        "bright_contrast": "Kontras",
        "bright_result": "Hasil Brightness/Kontras",
        "team_title": "### 👥 Anggota Tim",
        "team_subtitle": "Kelompok 5 – Anggota dan Peran",
        "team_sid": "**NIM:**",
        "team_role": "**Peran:**",
        "team_group": "**Kelompok:**",
        "axis_x": "Sumbu-X",
        "axis_y": "Sumbu-Y",
        "axis_diag": "Diagonal",
        "dark_mode": "Mode Gelap",
        "light_mode": "Mode Terang",
    },
    "en": {
        "title": "🧮 Matrix Transformations in Image Processing",
        "subtitle": "📌 Interactive demo of 2D matrix transformations and image filtering",
        "app_goal": "🎯 **App goal:** An interactive demo showing how 2D matrix transformations and image filtering work on images using linear algebra concepts.",
        "features": "- 🔁 Transformations: translation, scaling, rotation, shearing, reflection\n- 🧹 Filtering: blur, sharpen, edge detection, background removal, grayscale, brightness & contrast",
        "concept_1_title": "### 🧊 2D matrix transformations",
        "concept_1_text1": "A 2D image can be viewed as a set of points \\((x, y)\\) that can be moved using linear operations such as translation, scaling, rotation, shearing, and reflection, represented by 2×2 or 3×3 matrices (homogeneous coordinates).",
        "concept_1_text2": "Multiplying pixel coordinates by these matrices changes their positions: scaling changes the size, rotation turns the image around a center, shearing makes the shape slanted, and reflection flips the image across a chosen axis.",
        "concept_2_title": "### 📐 Image filtering (convolution)",
        "concept_2_text1": "Filtering uses a small kernel (convolution matrix) that slides over the image; at each position it produces a new pixel value from the sum of products between the kernel and its neighboring pixels.",
        "concept_2_text2": "Kernels with uniform positive values smooth or blur the image, while kernels with a strong positive center and negative neighbors sharpen and emphasize edges.",
        "concept_3_title": "### 🕹️ Why interactive?",
        "concept_3_text1": "It is interactive because users can change parameters (rotation angle, scale factors, shear values, blur/sharpen kernel types, etc.) and immediately see the effect on the image. ✨",
        "concept_3_text2": "This helps connect the structure of a matrix or kernel with its visual effect, making concepts like linear transformations and convolution more intuitive.",
        "quick_concepts": "#### 📖 Quick concepts",
        "quick_concepts_text": "- 🔁 2D transform: change pixel positions (translation, scaling, rotation, shearing, reflection).\n- 🧮 Convolution: a small kernel sliding over the image to produce new pixel values.",
        "upload_title": "### 🖼️ Upload Image",
        "upload_label": "Upload an image here (PNG/JPG/JPEG) 📂",
        "upload_success": "✅ Image uploaded successfully!",
        "upload_preview": "🖼️ Original Image Preview",
        "upload_info": "⬆️ Please upload an image first to use the tools below.",
        "tools_title": "### 🛠️ Image Processing Tools",
        "tools_subtitle": "🎛️ Choose a box below to open transformation or filter settings.",
        "geo_title": "#### 🔁 Geometric Transformations",
        "geo_desc": "Geometric transformations change the position, size, and orientation of pixels using linear operations (matrices).",
        "btn_translation": "↔️ Translation",
        "btn_scaling": "📏 Scaling",
        "btn_rotation": "🔄 Rotation",
        "btn_shearing": "📐 Shearing",
        "btn_reflection": "🪞 Reflection",
        "geo_info": "🔔 Please upload an image first to try transformations.",
        "trans_settings": "**↔️ Translation Settings**",
        "trans_dx": "dx (horizontal shift)",
        "trans_dy": "dy (vertical shift)",
        "btn_apply": "Apply",
        "trans_result": "Translation Result",
        "scale_settings": "**📏 Scaling Settings**",
        "scale_x": "Scale X",
        "scale_y": "Scale Y",
        "scale_result": "Scaling Result",
        "rot_settings": "**🔄 Rotation Settings**",
        "rot_angle": "Rotation angle (degrees)",
        "rot_result": "Rotation Result",
        "shear_settings": "**📐 Shearing Settings**",
        "shear_x": "Shear factor X",
        "shear_y": "Shear factor Y",
        "shear_result": "Shearing Result",
        "refl_settings": "**🪞 Reflection Settings**",
        "refl_axis": "Reflection axis",
        "refl_result": "Reflection Result",
        "hist_title": "#### 📊 Image Histogram",
        "hist_desc": "The histogram shows the distribution of pixel intensities (dark–bright), useful for analyzing brightness and contrast.",
        "btn_histogram": "Show Histogram 📈",
        "hist_warning": "Please upload an image first to display the histogram.",
        "filter_title": "#### 🧮 Image Filtering",
        "filter_desc": "Filtering changes pixel values based on their neighbors (convolution) to perform blur, sharpen, edge detection, background removal, and more.",
        "btn_blur": "🔲 Blur",
        "btn_sharpen": "✨ Sharpen",
        "btn_background": "🎯 Background",
        "btn_grayscale": "⚫ Grayscale",
        "btn_edge": "🔍 Edge",
        "btn_brightness": "☀️ Brightness",
        "filter_info": "🔔 Please upload an image first to use the filters.",
        "blur_settings": "**🔲 Blur Filter Settings**",
        "blur_kernel": "Kernel size",
        "blur_result": "Blur Result",
        "sharpen_settings": "**✨ Sharpen Filter Settings**",
        "sharpen_desc": "Enhance details and edges in the image.",
        "sharpen_result": "Sharpen Result",
        "bg_settings": "**🎯 Background Removal Settings**",
        "bg_method": "Method (demo uses HSV only for now)",
        "bg_result": "Background Removal Result",
        "gray_settings": "**⚫ Grayscale Conversion Settings**",
        "gray_desc": "Convert the image to grayscale (black and white).",
        "gray_result": "Grayscale Result",
        "edge_settings": "**🔍 Edge Detection Settings**",
        "edge_method": "Edge method",
        "edge_result": "Edge Image",
        "bright_settings": "**☀️ Brightness & Contrast Settings**",
        "bright_brightness": "Brightness",
        "bright_contrast": "Contrast",
        "bright_result": "Brightness/Contrast Result",
        "team_title": "### 👥 Team Members",
        "team_subtitle": "Group 5 – Members and Roles",
        "team_sid": "**SID:**",
        "team_role": "**Role:**",
        "team_group": "**Group:**",
        "axis_x": "X-axis",
        "axis_y": "Y-axis",
        "axis_diag": "Diagonal",
        "dark_mode": "Dark Mode",
        "light_mode": "Light Mode",
    }
}

# ===================== HEADER WITH TOGGLES =====================

# Get current language and theme
lang = st.session_state["language"]
t = translations[lang]
theme_mode = st.session_state["theme_mode"]

# Bungkus title + tombol dalam box
with st.container(border=True):
    # Title dan controls dalam 3 kolom sejajar
    header_col1, header_col2, header_col3 = st.columns([6, 1, 1], vertical_alignment="center")

    with header_col1:
        st.title(t["title"])

    with header_col2:
        # Language toggle - shows opposite language
        lang_button_text = "🇬🇧 EN" if lang == "id" else "🇮🇩 ID"
        if st.button(lang_button_text, key="lang_toggle", use_container_width=True):
            st.session_state["language"] = "en" if lang == "id" else "id"
            st.rerun()

    with header_col3:
        # Theme toggle - shows opposite mode icon
        theme_button_text = "🌙 Dark" if theme_mode == "light" else "☀️ Light"
        if st.button(theme_button_text, key="theme_toggle", use_container_width=True):
            st.session_state["theme_mode"] = "dark" if theme_mode == "light" else "light"
            st.rerun()

st.subheader(t["subtitle"])

# ----- Global layout + theme CSS -----

base_css = """
<style>
.block-container {
    max-width: 1200px;
    padding: 2.5rem 2rem 1.2rem 2rem;
}
section[data-testid="stExpander"]{
    border-radius:10px;
    padding:8px;
    box-shadow:0 1px 6px rgba(0,0,0,0.04);
    margin-bottom:10px;
    background-color: var(--stLightBlue-50);
}
section[data-testid="stExpander"] .streamlit-expanderHeader{
    font-size:16px;
}
.stImage > img{
    max-height:420px;
    object-fit:contain;
}
div[data-testid="column"] button {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    font-size: 14px !important;
    width: 100%;
    font-weight: 500 !important;
}
/* Green border for containers */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 2px solid #4CAF50 !important;
    border-radius: 12px !important;
}

/* Team member photo container - square with crop */
.team-photo-container {
    width: 140px;
    height: 140px;
    border-radius: 12px;
    overflow: hidden;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f0f0f0;
    border: 3px solid #4CAF50;
}

.team-photo-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
}
</style>
"""

light_css = """
<style>
body, .stApp {
    background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 25%, #e3f2fd 50%, #f5f5f5 75%, #e8f5e9 100%) !important;
    color: #1b5e20 !important;
}
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #1b5e20 !important;
}
/* Better visibility for toggles in light mode */
button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #1b5e20 !important;
    border: 2px solid #4CAF50 !important;
    font-weight: 600 !important;
}
button[kind="secondary"]:hover {
    background-color: #e8f5e9 !important;
    border-color: #2e7d32 !important;
}
.team-photo-container {
    background: #e8f5e9;
    border-color: #4CAF50;
}
</style>
"""

dark_css = """
<style>
body, .stApp {
    background: linear-gradient(135deg, #1b3a1b 0%, #0e1117 25%, #0d1b2a 50%, #1a1a1a 75%, #1b3a1b 100%) !important;
    color: #c8e6c9 !important;
}
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #c8e6c9 !important;
}
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(27, 58, 27, 0.3) !important;
}
/* Better visibility for toggles in dark mode */
button[kind="secondary"] {
    background-color: #1e3a1e !important;
    color: #c8e6c9 !important;
    border: 2px solid #66bb6a !important;
    font-weight: 600 !important;
}
button[kind="secondary"]:hover {
    background-color: #2d5a2d !important;
    border-color: #81c784 !important;
}
.team-photo-container {
    background: #1e3a1e;
    border-color: #66bb6a;
}
</style>
"""

st.markdown(base_css, unsafe_allow_html=True)

if theme_mode == "light":
    st.markdown(light_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

# ===================== APP GOAL =====================

with st.container(border=True):
    st.markdown(t["app_goal"])
    st.markdown(t["features"])

# ===================== THREE CONCEPT BOXES =====================

col1, col2, col3 = st.columns(3, vertical_alignment="top")

with col1:
    with st.container(border=True):
        st.markdown(t["concept_1_title"])
        st.markdown(t["concept_1_text1"])
        st.markdown(t["concept_1_text2"])

with col2:
    with st.container(border=True):
        st.markdown(t["concept_2_title"])
        st.markdown(t["concept_2_text1"])
        st.markdown(t["concept_2_text2"])

with col3:
    with st.container(border=True):
        st.markdown(t["concept_3_title"])
        st.markdown(t["concept_3_text1"])
        st.markdown(t["concept_3_text2"])

# ===================== HELPER FUNCTIONS =====================

def load_image(file):
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)
    return img_np

def to_opencv(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def to_streamlit(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_affine_transform(img_rgb, M, output_size=None):
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
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    padded = np.pad(img_gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    h, w = img_gray.shape
    output = np.zeros_like(img_gray, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)

    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def rgb_to_gray(img_rgb):
    img_bgr = to_opencv(img_rgb)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray

def adjust_brightness_contrast(img_rgb, brightness=0, contrast=0):
    img_bgr = to_opencv(img_rgb)
    beta = brightness
    alpha = 1 + (contrast / 100.0)
    adjusted = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return to_streamlit(adjusted)

def compute_histogram(img_rgb):
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

def simple_background_removal_hsv(img_rgb):
    img_bgr = to_opencv(img_rgb)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 200])
    mask_bg = cv2.inRange(hsv, lower, upper)
    mask_fg = cv2.bitwise_not(mask_bg)
    fg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_fg)
    fg_rgb = to_streamlit(fg)
    return fg_rgb

def image_to_bytes(img_rgb, fmt="PNG"):
    """Convert numpy RGB image to bytes for download."""
    pil_img = Image.fromarray(img_rgb.astype("uint8"))
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def create_square_image_html(image_path, size=140):
    """Create HTML for square cropped image"""
    return f"""
    <div class="team-photo-container">
        <img src="data:image/jpeg;base64,{{base64_img}}" alt="Team member"/>
    </div>
    """

def safe_display_square_image(path):
    """Display image in square format with proper cropping"""
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
            img_resized = img_cropped.resize((140, 140), Image.Resampling.LANCZOS)

            import base64
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

# Ensure images folder and placeholders exist
images_dir = "images"
os.makedirs(images_dir, exist_ok=True)
placeholder_files = [
    os.path.join(images_dir, "aditya.jpg"),
    os.path.join(images_dir, "maula_aqiel.jpg"),
    os.path.join(images_dir, "syafiq_nur.jpg"),
    os.path.join(images_dir, "rifat_fitrotu.jpg"),
]
for p in placeholder_files:
    if not os.path.exists(p):
        placeholder = Image.new("RGB", (400, 400), color=(200, 200, 200))
        placeholder.save(p, format="JPEG")

# ===================== CONCEPTS SHORT REMINDER =====================

with st.container(border=True):
    st.markdown(t["quick_concepts"])
    st.markdown(t["quick_concepts_text"])

# ===================== UPLOAD IMAGE =====================

with st.container(border=True):
    st.markdown(t["upload_title"])
    uploaded_file = st.file_uploader(
        label=t["upload_label"],
        type=["png", "jpg", "jpeg"],
        key="image_uploader"
    )

    if uploaded_file is not None:
        original_img = load_image(uploaded_file)
        st.session_state.original_img = original_img
        st.success(t["upload_success"])
        st.image(
            original_img,
            caption=t["upload_preview"],
            use_column_width=True
        )
    else:
        st.info(t["upload_info"])

original_img = st.session_state.original_img

# ===================== TOOLS TITLE =====================

st.markdown(t["tools_title"])
st.write(t["tools_subtitle"])

tools_col_left, tools_col_right = st.columns(2, vertical_alignment="top")

# ==================== LEFT: GEOMETRIC TRANSFORMATIONS ====================
with tools_col_left:
    with st.container(border=True):
        st.markdown(t["geo_title"])
        st.write(t["geo_desc"])
        st.markdown("---")

        trans_col1, trans_col2, trans_col3 = st.columns(3)
        with trans_col1:
            if st.button(t["btn_translation"], key="btn_trans_click", type="secondary"):
                st.session_state["geo_transform"] = "translation"
        with trans_col2:
            if st.button(t["btn_scaling"], key="btn_scale_click", type="secondary"):
                st.session_state["geo_transform"] = "scaling"
        with trans_col3:
            if st.button(t["btn_rotation"], key="btn_rot_click", type="secondary"):
                st.session_state["geo_transform"] = "rotation"

        trans_col4, trans_col5, _ = st.columns(3)
        with trans_col4:
            if st.button(t["btn_shearing"], key="btn_shear_click", type="secondary"):
                st.session_state["geo_transform"] = "shearing"
        with trans_col5:
            if st.button(t["btn_reflection"], key="btn_refl_click", type="secondary"):
                st.session_state["geo_transform"] = "reflection"

        # Transform parameter panel
        if original_img is None:
            st.info(t["geo_info"])
        else:
            if st.session_state["geo_transform"] == "translation":
                st.markdown(t["trans_settings"])
                dx = st.slider(t["trans_dx"], -200, 200, 0, key="trans_dx")
                dy = st.slider(t["trans_dy"], -200, 200, 0, key="trans_dy")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_trans", type="primary"):
                    T = np.array([[1, 0, dx],
                                  [0, 1, dy],
                                  [0, 0, 1]], dtype=np.float32)
                    translated_img = apply_affine_transform(original_img, T)
                    st.image(translated_img, caption=t["trans_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(translated_img, fmt="PNG"),
                            file_name="translation_result.png",
                            mime="image/png",
                            key="dl_trans_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(translated_img, fmt="JPEG"),
                            file_name="translation_result.jpg",
                            mime="image/jpeg",
                            key="dl_trans_jpg"
                        )

            elif st.session_state["geo_transform"] == "scaling":
                st.markdown(t["scale_settings"])
                sx = st.slider(t["scale_x"], 0.1, 3.0, 1.0, key="scale_x")
                sy = st.slider(t["scale_y"], 0.1, 3.0, 1.0, key="scale_y")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_scale", type="primary"):
                    h, w = original_img.shape[:2]
                    S = np.array([[sx, 0, 0],
                                  [0, sy, 0],
                                  [0, 0, 1]], dtype=np.float32)
                    new_w = int(w * sx)
                    new_h = int(h * sy)
                    scaled_img = apply_affine_transform(original_img, S, output_size=(new_w, new_h))
                    st.image(scaled_img, caption=t["scale_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(scaled_img, fmt="PNG"),
                            file_name="scaling_result.png",
                            mime="image/png",
                            key="dl_scale_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(scaled_img, fmt="JPEG"),
                            file_name="scaling_result.jpg",
                            mime="image/jpeg",
                            key="dl_scale_jpg"
                        )

            elif st.session_state["geo_transform"] == "rotation":
                st.markdown(t["rot_settings"])
                angle = st.slider(t["rot_angle"], -180, 180, 0, key="rot_angle")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_rot", type="primary"):
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
                    st.image(rotated_img, caption=t["rot_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(rotated_img, fmt="PNG"),
                            file_name="rotation_result.png",
                            mime="image/png",
                            key="dl_rot_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(rotated_img, fmt="JPEG"),
                            file_name="rotation_result.jpg",
                            mime="image/jpeg",
                            key="dl_rot_jpg"
                        )

            elif st.session_state["geo_transform"] == "shearing":
                st.markdown(t["shear_settings"])
                shear_x = st.slider(t["shear_x"], -1.0, 1.0, 0.0, key="shear_x")
                shear_y = st.slider(t["shear_y"], -1.0, 1.0, 0.0, key="shear_y")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_shear", type="primary"):
                    Sh = np.array([[1,      shear_x, 0],
                                   [shear_y, 1,      0],
                                   [0,       0,      1]], dtype=np.float32)
                    sheared_img = apply_affine_transform(original_img, Sh)
                    st.image(sheared_img, caption=t["shear_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(sheared_img, fmt="PNG"),
                            file_name="shearing_result.png",
                            mime="image/png",
                            key="dl_shear_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(sheared_img, fmt="JPEG"),
                            file_name="shearing_result.jpg",
                            mime="image/jpeg",
                            key="dl_shear_jpg"
                        )

            elif st.session_state["geo_transform"] == "reflection":
                st.markdown(t["refl_settings"])
                axis = st.selectbox(t["refl_axis"], [t["axis_x"], t["axis_y"], t["axis_diag"]], key="refl_axis")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_refl", type="primary"):
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
                    st.image(reflected_img, caption=t["refl_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(reflected_img, fmt="PNG"),
                            file_name="reflection_result.png",
                            mime="image/png",
                            key="dl_refl_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(reflected_img, fmt="JPEG"),
                            file_name="reflection_result.jpg",
                            mime="image/jpeg",
                            key="dl_refl_jpg"
                        )

    # Histogram box
    with st.container(border=True):
        st.markdown(t["hist_title"])
        st.write(t["hist_desc"])
        show_hist = st.button(t["btn_histogram"], key="btn_histogram", type="secondary")
        if show_hist:
            if original_img is not None:
                hist_fig = compute_histogram(original_img)
                st.pyplot(hist_fig)
                plt.close(hist_fig)
            else:
                st.warning(t["hist_warning"])

# ==================== RIGHT: IMAGE FILTERING ====================
with tools_col_right:
    with st.container(border=True):
        st.markdown(t["filter_title"])
        st.write(t["filter_desc"])
        st.markdown("---")

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            if st.button(t["btn_blur"], key="btn_blur_click", type="secondary"):
                st.session_state["image_filter"] = "blur"
        with filter_col2:
            if st.button(t["btn_sharpen"], key="btn_sharpen_click", type="secondary"):
                st.session_state["image_filter"] = "sharpen"
        with filter_col3:
            if st.button(t["btn_background"], key="btn_bg_click", type="secondary"):
                st.session_state["image_filter"] = "background"

        filter_col4, filter_col5, filter_col6 = st.columns(3)
        with filter_col4:
            if st.button(t["btn_grayscale"], key="btn_gray_click", type="secondary"):
                st.session_state["image_filter"] = "grayscale"
        with filter_col5:
            if st.button(t["btn_edge"], key="btn_edge_click", type="secondary"):
                st.session_state["image_filter"] = "edge"
        with filter_col6:
            if st.button(t["btn_brightness"], key="btn_bright_click", type="secondary"):
                st.session_state["image_filter"] = "brightness"

        if original_img is None:
            st.info(t["filter_info"])
        else:
            if st.session_state["image_filter"] == "blur":
                st.markdown(t["blur_settings"])
                kernel_size = st.selectbox(
                    t["blur_kernel"], [3, 5, 7], index=0, key="blur_kernel_size"
                )
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_blur", type="primary"):
                    gray = rgb_to_gray(original_img)
                    k = kernel_size
                    blur_kernel = np.ones((k, k), dtype=np.float32) / (k * k)
                    blurred_gray = manual_convolution_gray(gray, blur_kernel)
                    blurred_rgb = cv2.cvtColor(blurred_gray, cv2.COLOR_GRAY2RGB)
                    st.image(blurred_rgb, caption=t["blur_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(blurred_rgb, fmt="PNG"),
                            file_name="blur_result.png",
                            mime="image/png",
                            key="dl_blur_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(blurred_rgb, fmt="JPEG"),
                            file_name="blur_result.jpg",
                            mime="image/jpeg",
                            key="dl_blur_jpg"
                        )

            elif st.session_state["image_filter"] == "sharpen":
                st.markdown(t["sharpen_settings"])
                st.write(t["sharpen_desc"])
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_sharpen", type="primary"):
                    gray = rgb_to_gray(original_img)
                    sharpen_kernel = np.array(
                        [[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]],
                        dtype=np.float32
                    )
                    sharpened_gray = manual_convolution_gray(gray, sharpen_kernel)
                    sharpened_rgb = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2RGB)
                    st.image(sharpened_rgb, caption=t["sharpen_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(sharpened_rgb, fmt="PNG"),
                            file_name="sharpen_result.png",
                            mime="image/png",
                            key="dl_sharp_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(sharpened_rgb, fmt="JPEG"),
                            file_name="sharpen_result.jpg",
                            mime="image/jpeg",
                            key="dl_sharp_jpg"
                        )

            elif st.session_state["image_filter"] == "background":
                st.markdown(t["bg_settings"])
                method = st.selectbox(
                    t["bg_method"],
                    ["HSV Color Thresholding"],
                    key="bg_method"
                )
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_bg", type="primary"):
                    bg_removed_img = simple_background_removal_hsv(original_img)
                    st.image(bg_removed_img, caption=t["bg_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(bg_removed_img, fmt="PNG"),
                            file_name="background_result.png",
                            mime="image/png",
                            key="dl_bg_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(bg_removed_img, fmt="JPEG"),
                            file_name="background_result.jpg",
                            mime="image/jpeg",
                            key="dl_bg_jpg"
                        )

            elif st.session_state["image_filter"] == "grayscale":
                st.markdown(t["gray_settings"])
                st.write(t["gray_desc"])
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_gray", type="primary"):
                    gray_img = rgb_to_gray(original_img)
                    gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
                    st.image(gray_rgb, caption=t["gray_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(gray_rgb, fmt="PNG"),
                            file_name="grayscale_result.png",
                            mime="image/png",
                            key="dl_gray_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(gray_rgb, fmt="JPEG"),
                            file_name="grayscale_result.jpg",
                            mime="image/jpeg",
                            key="dl_gray_jpg"
                        )

            elif st.session_state["image_filter"] == "edge":
                st.markdown(t["edge_settings"])
                method_edge = st.selectbox(
                    t["edge_method"], ["Sobel", "Canny"], key="edge_method"
                )
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_edge", type="primary"):
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
                    st.image(edge_img, caption=f"{t['edge_result']} ({method_edge})", use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(edge_img, fmt="PNG"),
                            file_name="edge_result.png",
                            mime="image/png",
                            key="dl_edge_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(edge_img, fmt="JPEG"),
                            file_name="edge_result.jpg",
                            mime="image/jpeg",
                            key="dl_edge_jpg"
                        )

            elif st.session_state["image_filter"] == "brightness":
                st.markdown(t["bright_settings"])
                brightness = st.slider(t["bright_brightness"], -100, 100, 0, key="brightness_value")
                contrast = st.slider(t["bright_contrast"], -100, 100, 0, key="contrast_value")
                if st.button(f"{t['btn_apply']} ✅", key="btn_apply_bright", type="primary"):
                    adjusted_img = adjust_brightness_contrast(original_img, brightness, contrast)
                    st.image(adjusted_img, caption=t["bright_result"], use_column_width=True)

                    col_png, col_jpg = st.columns(2)
                    with col_png:
                        st.download_button(
                            label="⬇️ Download PNG",
                            data=image_to_bytes(adjusted_img, fmt="PNG"),
                            file_name="brightness_contrast_result.png",
                            mime="image/png",
                            key="dl_bright_png"
                        )
                    with col_jpg:
                        st.download_button(
                            label="⬇️ Download JPG",
                            data=image_to_bytes(adjusted_img, fmt="JPEG"),
                            file_name="brightness_contrast_result.jpg",
                            mime="image/jpeg",
                            key="dl_bright_jpg"
                        )

# ===================== TEAM MEMBERS =====================

st.markdown(t["team_title"])
st.write(t["team_subtitle"])

members = [
    {"img": "images/aditya.jpg", "name": "ADITYA ANGGARA PAMUNGKAS", "sid": "04202400051", "role": "Leader"},
    {"img": "images/maula_aqiel.jpg", "name": "MAULA AQIEL NURI", "sid": "04202400023", "role": "Member"},
    {"img": "images/syafiq_nur.jpg", "name": "SYAFIQ NUR RAMADHAN", "sid": "04202400073", "role": "Member"},
    {"img": "images/rifat_fitrotu.jpg", "name": "RIFAT FITROTU SALMAN", "sid": "04202400106", "role": "Member"},
]

cols_row1 = st.columns(2, vertical_alignment="top")
for i in range(2):
    with cols_row1[i]:
        with st.container(border=True):
            m = members[i]
            col_img, col_info = st.columns([1, 2], vertical_alignment="center")
            with col_img:
                safe_display_square_image(m["img"])
            with col_info:
                st.markdown(f"**{m['name']}**")
                st.markdown(f"{t['team_sid']} {m['sid']}")
                st.markdown(f"{t['team_role']} {m['role']}")
                st.markdown(f"{t['team_group']} 5")

cols_row2 = st.columns(2, vertical_alignment="top")
for i in range(2, 4):
    with cols_row2[i - 2]:
        with st.container(border=True):
            m = members[i]
            col_img, col_info = st.columns([1, 2], vertical_alignment="center")
            with col_img:
                safe_display_square_image(m["img"])
            with col_info:
                st.markdown(f"**{m['name']}**")
                st.markdown(f"{t['team_sid']} {m['sid']}")
                st.markdown(f"{t['team_role']} {m['role']}")
                st.markdown(f"{t['team_group']} 5")
