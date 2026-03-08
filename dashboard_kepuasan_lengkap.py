import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(page_title="Dashboard Analisis Tes", layout="wide")

st.title("📊 Dashboard Analisis Hasil Tes Siswa")
st.markdown("Dashboard interaktif untuk membaca performa siswa dan analisis soal")

# =====================================================
# UPLOAD DATA
# =====================================================
uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df)

    # =====================================================
    # IDENTIFIKASI KOLOM SOAL
    # =====================================================
    soal_cols = [col for col in df.columns if "Soal_" in col]

    indikator = df[soal_cols]

    # =====================================================
    # KPI UTAMA
    # =====================================================
    rata_skor = df["Total_Skor"].mean()
    skor_max = len(soal_cols)
    skor_min = df["Total_Skor"].min()

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Siswa", len(df))
    col2.metric("Rata-rata Skor", f"{rata_skor:.2f}")
    col3.metric("Skor Maksimum", skor_max)

    st.divider()

    # =====================================================
    # DISTRIBUSI SKOR
    # =====================================================
    st.header("📈 Distribusi Nilai Siswa")

    fig_hist, ax_hist = plt.subplots()

    ax_hist.hist(df["Total_Skor"], bins=10)
    ax_hist.set_xlabel("Total Skor")
    ax_hist.set_ylabel("Jumlah Siswa")
    ax_hist.set_title("Distribusi Skor Siswa")

    st.pyplot(fig_hist)

    # =====================================================
    # ANALISIS TINGKAT KESULITAN SOAL
    # =====================================================
    st.header("📚 Analisis Tingkat Kesulitan Soal")

    tingkat_benar = indikator.mean()

    fig_soal, ax_soal = plt.subplots(figsize=(10,4))

    ax_soal.bar(tingkat_benar.index, tingkat_benar.values)

    ax_soal.set_ylabel("Proporsi Benar")
    ax_soal.set_title("Persentase Jawaban Benar per Soal")

    plt.xticks(rotation=90)

    st.pyplot(fig_soal)

    soal_sulit = tingkat_benar.idxmin()
    soal_mudah = tingkat_benar.idxmax()

    st.warning(f"Soal paling sulit: **{soal_sulit}**")
    st.success(f"Soal paling mudah: **{soal_mudah}**")

    st.divider()

    # =====================================================
    # FILTER INTERAKTIF SOAL
    # =====================================================
    st.header("🔍 Analisis Detail per Soal")

    selected_soal = st.selectbox("Pilih Soal", soal_cols)

    benar = df[selected_soal].sum()
    salah = len(df) - benar

    fig_pie, ax_pie = plt.subplots()

    ax_pie.pie([benar, salah], labels=["Benar", "Salah"], autopct="%1.1f%%")
    ax_pie.set_title(f"Distribusi Jawaban {selected_soal}")

    st.pyplot(fig_pie)

    # =====================================================
    # KORELASI ANTAR SOAL
    # =====================================================
    st.header("🔗 Korelasi Antar Soal")

    corr = indikator.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(8,6))

    im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    plt.colorbar(im)

    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))

    ax_corr.set_xticklabels(corr.columns, rotation=90)
    ax_corr.set_yticklabels(corr.columns)

    st.pyplot(fig_corr)

    st.divider()

    # =====================================================
    # REGRESI
    # =====================================================
    st.header("📊 Analisis Pengaruh Soal terhadap Skor")

    X = sm.add_constant(indikator)
    y = df["Total_Skor"]

    model = sm.OLS(y, X).fit()

    coef = model.params[1:]

    fig_reg, ax_reg = plt.subplots(figsize=(10,4))

    ax_reg.bar(coef.index, coef.values)

    ax_reg.set_title("Kontribusi Soal terhadap Total Skor")

    plt.xticks(rotation=90)

    st.pyplot(fig_reg)

    st.info(f"Nilai R² Model: {model.rsquared:.3f}")

    st.divider()

    # =====================================================
    # CLUSTERING SISWA
    # =====================================================
    st.header("🎯 Segmentasi Kemampuan Siswa")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(indikator)

    kmeans = KMeans(n_clusters=3, random_state=42)

    df["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_mean = df.groupby("Cluster")["Total_Skor"].mean()

    fig_cluster, ax_cluster = plt.subplots()

    cluster_mean.plot(kind="bar", ax=ax_cluster)

    ax_cluster.set_ylabel("Rata-rata Skor")
    ax_cluster.set_title("Segmentasi Kemampuan")

    st.pyplot(fig_cluster)

    st.dataframe(df.groupby("Cluster").mean())

    st.divider()

    # =====================================================
    # INSIGHT OTOMATIS
    # =====================================================
    st.header("💡 Insight Otomatis")

    insight = f"""
    - Rata-rata skor siswa adalah **{rata_skor:.2f}** dari maksimal **{skor_max}**
    - Soal paling sulit adalah **{soal_sulit}**
    - Soal paling mudah adalah **{soal_mudah}**
    - Model regresi memiliki R² sebesar **{model.rsquared:.3f}**
    """

    st.info(insight)

    # =====================================================
    # DOWNLOAD DATA
    # =====================================================
    st.header("⬇ Download Data Hasil Analisis")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Data",
        data=csv,
        file_name="hasil_analisis.csv",
        mime="text/csv"
    )
