# app.py
import streamlit as st
import torch
import faiss
import numpy as np
import pickle
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from data_metadata import get_metadata_df
from openai import OpenAI # Atau library lain untuk Generative

# --- SETUP ---
st.set_page_config(page_title="Batik Nitik Smart Retrieval", layout="wide")
MODEL_ID = "openai/clip-vit-base-patch32"
INDEX_FILE = "batik_faiss.index"
PATHS_FILE = "image_paths.pkl"

# --- FUNGSI LOAD RESOURCES (Cached) ---
@st.cache_resource
def load_resources():
    print("Loading resources...")
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    index = faiss.read_index(INDEX_FILE)
    with open(PATHS_FILE, "rb") as f:
        image_paths = pickle.load(f)
    df_meta = get_metadata_df()
    return model, processor, index, image_paths, df_meta

model, processor, index, image_paths, df_meta = load_resources()

# --- FUNGSI SEARCH ---
def search(query_content, k=3, is_image=False):
    # Ekstraksi fitur query
    if is_image:
        inputs = processor(images=query_content, return_tensors="pt")
        with torch.no_grad():
            query_vector = model.get_image_features(**inputs)
    else:
        inputs = processor(text=[query_content], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_vector = model.get_text_features(**inputs)
            
    # Normalisasi
    query_vector = query_vector / query_vector.norm(p=2, dim=-1, keepdim=True)
    query_vector_np = query_vector.numpy().astype('float32')
    
    # Search FAISS
    distances, indices = index.search(query_vector_np, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        path = image_paths[idx]
        # Asumsi nama folder adalah nama kelas (misal: .../Sekar_Kemuning/img01.jpg)
        class_name = os.path.basename(os.path.dirname(path))
        
        # Ambil metadata
        meta = df_meta[df_meta['class_name'] == class_name]
        meaning = meta['meaning'].values[0] if not meta.empty else "Belum ada data filosofi."
        
        results.append({
            "path": path,
            "score": distances[0][i],
            "class": class_name,
            "meaning": meaning
        })
    return results

# --- FUNGSI GENERATIVE (LLM) ---
def generate_explanation(results, user_query):
    # Jika Anda punya API KEY OpenAI:
    # client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
    
    # Prompt Engineering (RAG Context)
    context_text = ""
    for r in results:
        context_text += f"- Motif: {r['class']}, Filosofi: {r['meaning']}\n"
        
    prompt = f"""
    Pengguna mencari tentang: '{user_query}'.
    Sistem menemukan Batik berikut yang relevan:
    {context_text}
    
    Tugas Anda: Jelaskan secara singkat dan puitis hubungan antara pencarian pengguna dengan filosofi batik yang ditemukan. Gunakan Bahasa Indonesia.
    """
    
    # Simulasi Generative (Jika tidak ada API Key, gunakan string statis ini dulu)
    # Untuk nilai tambah, ganti ini dengan panggilan `client.chat.completions.create(...)`
    simulated_response = f"**Analisis AI:**\nBerdasarkan pencarian '{user_query}', kami menemukan motif yang sangat relevan. \n\n"
    for r in results:
        simulated_response += f"Motif **{r['class']}** terpilih karena melambangkan nilai luhur. {r['meaning']}\n"
    
    return simulated_response + "\n\n*(Catatan: Sambungkan API OpenAI di kode untuk hasil dinamis)*"

# --- UI STREAMLIT ---
st.title("Batik Nitik 960: Retrieval & Generation")
st.markdown("Cari motif batik berdasarkan gambar atau teks, dan pahami filosofinya.")

# Sidebar untuk opsi
mode = st.sidebar.radio("Mode Pencarian", ["Text to Image", "Image to Image"])
k_slider = st.sidebar.slider("Jumlah Hasil (Top-K)", 1, 10, 3)

# Logic Input
results = []
query_display = None

if mode == "Text to Image":
    text_query = st.text_input("Masukkan deskripsi (cth: 'batik untuk upacara pernikahan' atau 'simbol kesucian')")
    if st.button("Cari") and text_query:
        results = search(text_query, k=k_slider, is_image=False)
        query_display = text_query

elif mode == "Image to Image":
    uploaded_file = st.file_uploader("Upload gambar batik", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Query", width=200)
        if st.button("Cari Serupa"):
            results = search(image, k=k_slider, is_image=True)
            query_display = "Gambar yang diunggah"

# Menampilkan Hasil
if results:
    st.divider()
    st.subheader("Hasil Pencarian & Generasi Pengetahuan")
    
    # 1. Tampilkan Gambar
    cols = st.columns(len(results))
    for i, res in enumerate(results):
        with cols[i]:
            img = Image.open(res['path'])
            st.image(img, use_column_width=True)
            st.caption(f"**{res['class']}**\nScore: {res['score']:.4f}")
            with st.expander("Lihat Filosofi"):
                st.write(res['meaning'])

    # 2. Generative Component
    st.divider()
    st.subheader("Penjelasan AI (Generative Context)")
    with st.spinner("Sedang menyusun narasi budaya..."):
        # Panggil fungsi generative
        explanation = generate_explanation(results, query_display)
        st.info(explanation)