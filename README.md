# Tugas Kelompok Sub-CLO 12-1-1: Evaluasi Sistem Rekomendasi

Prediksi rating dan rekomendasi film pada dataset **MovieLens `ml-latest-small`** dengan tiga metode, lalu dievaluasi memakai MAE, MSE, RMSE, NMSE, dan NDCG.

| | |
|---|---|
| **Kelompok** | 2 |
| **Mata kuliah** | Sistem Rekomendasi |
| **Deadline** | Jumat, 20 November 2026, 23.59 |
| **Source code** | https://github.com/tissuedry/usercf-itemcf-mf-movielens (akses diberikan ke `bayzal@gmail.com`) |

**Anggota**

| No | Nama | NIM |
|---|---|---|
| 1 | Yudha Setiawan Wicaksono | 103012300480 |
| 2 | Ghanif Hadiyana Akbar | 103012300018 |
| 3 | Dhafin Ghiffary | 103012300348 |
| 4 | Alif Ihsan | 103012330079 |
| 5 | Arif Rahmatiana | 103012300446 |

## Isi repositori

| File | Isi |
|---|---|
| [UserCF_MovieLens.ipynb](UserCF_MovieLens.ipynb) | User-based collaborative filtering (Pearson/mean-centered, k = 30 tetangga) |
| [ItemCF_MovieLens.ipynb](ItemCF_MovieLens.ipynb) | Item-based collaborative filtering (cosine dan adjusted cosine) |
| [mf_sgd.ipynb](mf_sgd.ipynb) | Matrix factorization dengan SGD, tanpa dan dengan bias |

Setiap notebook berisi alur lengkap: EDA/setup, preprocessing dan split, implementasi model, verifikasi pada contoh kecil, evaluasi, contoh rekomendasi Top-N, dan analisis.

## Dataset

[MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) (GroupLens):

- 610 user, 9.724 film yang dirating, 100.836 rating (skala 0,5 sampai 5,0, rata-rata 3,50)
- Sparsity 98,30%; tiap user minimal 20 rating (history user tersedia)
- Diunduh otomatis oleh notebook jika folder `ml-latest-small/` belum ada.

## Cara menjalankan

Notebook ditulis untuk **Google Colab** (sel setup memakai `wget`/`unzip`), tetapi bisa juga dijalankan lokal.

1. Buka notebook di Colab (Upload / buka dari GitHub), atau lokal: `jupyter notebook`.
2. Dependensi: `numpy`, `pandas`, `matplotlib` (tanpa library rekomendasi eksternal; semua model ditulis sendiri).
3. Jalankan semua sel secara berurutan (`Runtime > Run all`). Seed tetap 42 sehingga hasil bisa diulang.
4. Lokal di Windows: sel `!wget`/`!unzip` tidak berjalan; unduh `ml-latest-small.zip` secara manual, ekstrak ke folder `ml-latest-small/` di samping notebook. Notebook MF juga mencari data di `data/`.
5. Notebook akan menyimpan `hasil_ringkasan.csv`, `hasil_topn_contoh.csv`, dan `fig_*.png`.

## Metode

**User-based CF.** Similarity antar user dihitung dari rating yang di-center per user. Prediksi: `r̂(a,p) = r̄_a + Σ sim(a,b)·(r_bp − r̄_b) / Σ sim(a,b)` atas k = 30 tetangga dengan sim > 0, dipotong ke [0,5; 5]. Jika tidak ada tetangga, dipakai rata-rata user.

**Item-based CF.** Similarity antar film dihitung dari user yang merating kedua film, dengan dua varian: cosine dan adjusted cosine. Prediksi: `r̂(u,p) = Σ sim(i,p)·r_ui / Σ sim(i,p)` atas film yang sudah dirating user dan memiliki sim > 0. Kode diverifikasi terhadap contoh hitung manual "Alice" (sim = 0,7803; prediksi 4,0752).

**Matrix factorization (SGD).** `r̂ = μ + b_u + b_i + p_u·q_i`, dilatih dengan SGD dan regularisasi L2. Dua varian: tanpa bias dan dengan bias. Hyperparameter (k, β, α, jumlah epoch) dipilih dari data validasi (10% dari train), bukan dari data test. Algoritma dasarnya diuji lebih dulu pada matriks 5×4.

## Metrik evaluasi

| Metrik | Rumus |
|---|---|
| MAE | `Σ\|p − r\| / n` |
| MSE | `Σ(p − r)² / n` |
| RMSE | `√MSE` |
| NMSE | normalisasi MSE dengan rentang rating (lihat catatan di bawah) |
| NDCG@N | `DCG / IDCG`, `DCG = Σ rel_i / log2(i+1)`; N = 5 dan 10 |

NDCG dihitung pada item test tiap user (user dengan minimal 2 item test), dalam dua versi: gain linear (rel = rating) dan biner (rel = 1 jika rating ≥ 4).

## Hasil ringkas (test set)

| Model | MAE | MSE | RMSE | NDCG@5 (linear) | NDCG@10 (linear) |
|---|---|---|---|---|---|
| User-based CF (k = 30) | 0,6769 | 0,7849 | 0,8859 | 0,8778 | 0,9030 |
| Item-based CF, cosine | 0,7335 | 0,8905 | 0,9437 | 0,8210 | 0,8552 |
| Item-based CF, adjusted cosine | 0,6776 | 0,7769 | 0,8814 | 0,8686 | 0,8961 |
| MF-SGD tanpa bias (k=10, α=0,002, β=0,1) | 0,6909 | 0,7981 | 0,8934 | 0,8761 | 0,9006 |
| **MF-SGD dengan bias (k=100, α=0,01, β=0,2)** | **0,6509** | **0,7208** | **0,8490** | **0,8940** | **0,9152** |
| Baseline rata-rata user | 0,7357 | 0,8877 | 0,9422 | n/a | n/a |
| Baseline rata-rata global | 0,8284 | 1,0914 | 1,0447 | 0,8216 | 0,8555 |

### Catatan penting saat membandingkan angka

- **Definisi NMSE berbeda antar notebook.** UserCF dan ItemCF memakai `MSE / (r_max − r_min)` (dibagi 4,5), sedangkan MF memakai `MSE / (r_max − r_min)²` (dibagi 20,25). Karena itu NMSE UserCF/ItemCF (mis. 0,1744 dan 0,1726) tidak bisa dibandingkan langsung dengan NMSE MF (0,0356). Sebaiknya disatukan sebelum laporan final.
- **Split data berbeda.** UserCF dan ItemCF memakai split 80/20 acak global (train 80.669, test 20.167). MF memakai split 80/20 per user (test 20.164), dengan 10% train dipakai sebagai validasi. Selisih antar model karenanya sebagian berasal dari perbedaan data uji, bukan hanya perbedaan algoritma.
- Tabel di atas menyalin angka dari output notebook; jalankan ulang notebook untuk memastikan angka terbaru sebelum menyalinnya ke laporan.
- Metrik Precision/Recall/F1@10 hanya dihitung di notebook MF.

## Temuan utama

1. MF dengan bias memberi akurasi prediksi rating terbaik (RMSE 0,849). Bias global, user, dan item menyumbang sebagian besar perbaikan dibanding MF tanpa bias (0,893).
2. Pada item-based CF, adjusted cosine jauh lebih baik daripada cosine biasa (MAE 0,678 vs 0,734); cosine biasa bahkan hampir tidak lebih baik dari baseline rata-rata user.
3. User-based CF (MAE 0,677) hampir setara dengan item-based adjusted cosine, dan keduanya masih di bawah MF dengan bias.
4. Learning rate α adalah parameter paling sensitif pada SGD; α = 0,01 terbaik, α ≥ 0,05 mulai tidak stabil.
5. Akurasi prediksi rating yang bagus tidak menjamin Top-N yang bagus. Pada MF, F1@10 (0,011) jauh di bawah baseline populer (0,107) karena model merekomendasikan film langka dengan prediksi 5,0. Pada CF, Top-N dengan sedikit tetangga menghasilkan skor yang seri.
6. Keterbatasan: satu split saja (tanpa cross-validation), cold item sekitar 4% pasangan test, dan matriks similarity item berukuran besar.

## Checklist laporan PDF

- [ ] Nomor kelompok dan daftar anggota
- [ ] Link source code (akses diberikan ke `bayzal@gmail.com`)
- [ ] Penjelasan langkah tiap metode + screenshot kode
- [ ] Tabel hasil MAE, MSE, RMSE, NMSE, NDCG untuk ketiga metode
- [ ] Analisis dan kesimpulan
