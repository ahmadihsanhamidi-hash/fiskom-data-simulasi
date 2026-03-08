import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Hasil Tes", layout="wide")
st.title("📊 Dashboard Analisis Hasil Tes Siswa")

# ==========================================================
# LOAD DATA
# ==========================================================
uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    # ==========================================================
    # AMBIL DATA SOAL
    # ==========================================================
    soal_cols = [col for col in df.columns if "Soal_" in col]

    indikator = df[soal_cols]

    # ==========================================================
    # KPI NILAI RATA-RATA
    # ==========================================================
    rata_skor = df["Total_Skor"].mean()
    skor_maks = len(soal_cols)

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Siswa", len(df))
    col2.metric("Rata-rata Skor", f"{rata_skor:.2f}")
    col3.metric("Skor Maksimum", skor_maks)

    st.divider()

    # ==========================================================
    # ANALISIS KESULITAN SOAL
    # ==========================================================
    st.header("📈 Analisis Tingkat Kesulitan Soal")

    tingkat_benar = indikator.mean()

    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.bar(tingkat_benar.index, tingkat_benar.values)
    ax1.set_ylabel("Proporsi Jawaban Benar")
    ax1.set_title("Tingkat Kesulitan Soal")

    plt.xticks(rotation=90)

    st.pyplot(fig1)

    st.info(f"Soal paling sulit: **{tingkat_benar.idxmin()}**")

    st.divider()

    # ==========================================================
    # KORELASI ANTAR SOAL
    # ==========================================================
    st.header("🔗 Korelasi Antar Soal")

    corr = indikator.corr()

    fig2, ax2 = plt.subplots(figsize=(8,6))
    im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    plt.colorbar(im)

    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))

    ax2.set_xticklabels(corr.columns, rotation=90)
    ax2.set_yticklabels(corr.columns)

    st.pyplot(fig2)

    st.divider()

    # ==========================================================
    # REGRESI (PENGARUH SOAL TERHADAP TOTAL SKOR)
    # ==========================================================
    st.header("📊 Analisis Regresi")

    X = sm.add_constant(indikator)
    y = df["Total_Skor"]

    model = sm.OLS(y, X).fit()

    coef = model.params[1:]

    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.bar(coef.index, coef.values)

    ax3.set_title("Pengaruh Soal terhadap Total Skor")

    plt.xticks(rotation=90)

    st.pyplot(fig3)

    st.info(f"Nilai R²: **{model.rsquared:.3f}**")

    st.divider()

    # ==========================================================
    # CLUSTERING SISWA
    # ==========================================================
    st.header("🎯 Segmentasi Kemampuan Siswa")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(indikator)

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster = kmeans.fit_predict(X_scaled)

    df["Cluster"] = cluster

    cluster_mean = df.groupby("Cluster")["Total_Skor"].mean()

    fig4, ax4 = plt.subplots()

    cluster_mean.plot(kind="bar", ax=ax4)

    ax4.set_ylabel("Rata-rata Skor")
    ax4.set_title("Segmentasi Kemampuan Siswa")

    st.pyplot(fig4)

    st.success("Segmentasi siswa berhasil dibuat")
