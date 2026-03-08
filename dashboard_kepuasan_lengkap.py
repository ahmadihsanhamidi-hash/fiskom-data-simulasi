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
st.set_page_config(page_title="Dashboard Analisis Tes Siswa", layout="wide")

st.title("📊 Dashboard Analisis Hasil Tes")
st.write("Dashboard interaktif untuk analisis performa siswa dan kualitas soal")

# ==========================================================
# UPLOAD DATA
# ==========================================================
uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file:

    try:
        df = pd.read_excel(uploaded_file)
    except:
        st.error("File tidak bisa dibaca")
        st.stop()

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    # ==========================================================
    # VALIDASI DATA
    # ==========================================================
    if "Total_Skor" not in df.columns:
        st.error("Dataset harus memiliki kolom 'Total_Skor'")
        st.stop()

    # cari semua kolom soal otomatis
    soal_cols = [col for col in df.columns if "Soal_" in col]

    if len(soal_cols) == 0:
        st.error("Kolom soal tidak ditemukan")
        st.stop()

    indikator = df[soal_cols].apply(pd.to_numeric, errors="coerce")
    indikator = indikator.fillna(0)

    # ==========================================================
    # KPI UTAMA
    # ==========================================================
    st.header("📌 Statistik Utama")

    rata_skor = df["Total_Skor"].mean()
    max_skor = df["Total_Skor"].max()
    min_skor = df["Total_Skor"].min()

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Siswa", len(df))
    col2.metric("Rata-rata Skor", f"{rata_skor:.2f}")
    col3.metric("Skor Maksimum", max_skor)

    st.divider()

    # ==========================================================
    # DISTRIBUSI NILAI
    # ==========================================================
    st.header("📈 Distribusi Skor")

    fig_hist, ax_hist = plt.subplots()

    ax_hist.hist(df["Total_Skor"], bins=10)
    ax_hist.set_xlabel("Total Skor")
    ax_hist.set_ylabel("Jumlah Siswa")
    ax_hist.set_title("Distribusi Skor")

    st.pyplot(fig_hist)

    st.divider()

    # ==========================================================
    # ANALISIS KESULITAN SOAL
    # ==========================================================
    st.header("📚 Analisis Tingkat Kesulitan Soal")

    tingkat_benar = indikator.mean()

    fig_bar, ax_bar = plt.subplots(figsize=(10,4))

    ax_bar.bar(tingkat_benar.index, tingkat_benar.values)
    ax_bar.set_ylabel("Proporsi Benar")
    ax_bar.set_title("Persentase Jawaban Benar per Soal")

    plt.xticks(rotation=90)

    st.pyplot(fig_bar)

    soal_sulit = tingkat_benar.idxmin()
    soal_mudah = tingkat_benar.idxmax()

    st.warning(f"Soal paling sulit: {soal_sulit}")
    st.success(f"Soal paling mudah: {soal_mudah}")

    st.divider()

    # ==========================================================
    # ANALISIS DETAIL SOAL
    # ==========================================================
    st.header("🔍 Analisis Detail Soal")

    selected_soal = st.selectbox("Pilih Soal", soal_cols)

    benar = indikator[selected_soal].sum()
    salah = len(indikator) - benar

    fig_pie, ax_pie = plt.subplots()

    ax_pie.pie([benar, salah], labels=["Benar","Salah"], autopct="%1.1f%%")
    ax_pie.set_title(f"Distribusi Jawaban {selected_soal}")

    st.pyplot(fig_pie)

    st.divider()

    # ==========================================================
    # KORELASI ANTAR SOAL
    # ==========================================================
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

    # ==========================================================
    # REGRESI
    # ==========================================================
    st.header("📊 Analisis Regresi")

    try:
        X = sm.add_constant(indikator)
        y = pd.to_numeric(df["Total_Skor"], errors="coerce")

        model = sm.OLS(y, X).fit()

        coef = model.params[1:]

        fig_reg, ax_reg = plt.subplots(figsize=(10,4))

        ax_reg.bar(coef.index, coef.values)

        ax_reg.set_title("Pengaruh Soal terhadap Total Skor")

        plt.xticks(rotation=90)

        st.pyplot(fig_reg)

        st.info(f"Nilai R² Model: {model.rsquared:.3f}")

    except:
        st.warning("Regresi tidak dapat dihitung")

    st.divider()

    # ==========================================================
    # CLUSTERING SISWA
    # ==========================================================
    st.header("🎯 Segmentasi Kemampuan Siswa")

    try:

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(indikator)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

        df["Cluster"] = kmeans.fit_predict(X_scaled)

        cluster_mean = df.groupby("Cluster", as_index=True).mean(numeric_only=True)

        fig_cluster, ax_cluster = plt.subplots()

        cluster_mean["Total_Skor"].plot(kind="bar", ax=ax_cluster)

        ax_cluster.set_ylabel("Rata-rata Skor")
        ax_cluster.set_title("Segmentasi Kemampuan")

        st.pyplot(fig_cluster)

        st.subheader("Ringkasan Cluster")
        st.dataframe(cluster_mean)

    except:
        st.warning("Clustering gagal dilakukan")

    st.divider()

    # ==========================================================
    # INSIGHT OTOMATIS
    # ==========================================================
    st.header("💡 Insight Otomatis")

    insight = f"""
    Rata-rata skor siswa adalah {rata_skor:.2f}.
    Soal paling sulit adalah {soal_sulit}.
    Soal paling mudah adalah {soal_mudah}.
    """

    st.info(insight)

    st.divider()

    # ==========================================================
    # DOWNLOAD DATA
    # ==========================================================
    st.header("⬇ Download Data")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Data Hasil Analisis",
        data=csv,
        file_name="hasil_analisis.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload file Excel untuk memulai analisis.")
