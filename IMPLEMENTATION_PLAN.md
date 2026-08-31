# Physio-STF: Implementation Plan

## Spatiotemporal Multimodal Glucose Monitor (ST-MGM)

> **Repo Target:** `d:\zicofarry\GitHub\physio-stf`
> **Repo Referensi:** `D:\zicofarry\GitHub\physiocgm-hybrid` (kode awal)
> **IoT Referensi:** `D:\zicofarry\GitHub\IOT-prototype-deteksi-nyeri` (data validasi)

---

## 1. Arsitektur Komputasi ST-MGM Presisi (Sesuai Diagram Terbaru)

Seluruh sinyal fisiologis direkam secara simultan dalam window durasi **5 menit (300 detik)**. Model ini mengombinasikan fitur temporal (statistik DWT), morfologis spasial (1D-ResNet), dan representasi EDA (Conv1D + Bi-LSTM) melalui *Multi-Task Fusion*.

```
+---------------------------------------------------------------------------------------------------+
|                                  DATA ACQUISITION & PREPROCESSING                                 |
+---------------------------------------------------------------------------------------------------+
| ECG (5 min @ 250Hz)   --> Bandpass Filter (0.5-40Hz) --> Consecutive cardiac cycles / Individual beats |
| PPG (5 min @ 64Hz)    --> Bandpass Filter (0.5-8Hz)  --> Consecutive pulses / Individual pulses   |
| EDA (5 min @ 4Hz)     --> Z-Score Normalization                                                   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                SPATIOTEMPORAL FEATURE EXTRACTION                                  |
+---------------------------------------------------------------------------------------------------+
| A. Temporal Statistical:                                                                          |
|    DWT (db4, 7-level) -> 10 Stat Features x 8 sub-bands = 80 feat/modality (160 total)            |
|    --> Feature Selection (UFS + L1-FS + RFE) --> F_temporal (~18 features)                        |
|                                                                                                   |
| B. Spatial Morphological:                                                                         |
|    1D-ResNet (5-layer) [ECG + PPG] --> DNN Compression (512 -> Drop 0.3 -> 128 -> Drop 0.2 -> 40)|
|    --> F_morphological (40 features)                                                              |
|                                                                                                   |
| C. EDA Features:                                                                                  |
|    Conv1D (1->8, k=12, s=6) -> Conv1D (8->16, k=8, s=5) -> Bi-LSTM (hidden=32 per direction)        |
|    --> F_eda (64 features)                                                                        |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                  MULTI-TASK FUSION & PREDICTION                                   |
+---------------------------------------------------------------------------------------------------+
| Feature Fusion (Concatenation): F_fusion = {F_temporal, F_morphological, F_eda} (~122 features)   |
| Shared MLP: 64 (Dropout 0.3) --> 32 (Dropout 0.2)                                                 |
| Multi-Task Heads:                                                                                 |
|   1. Classification Head --> Hypo / Normal                                                        |
|   2. Regression Head     --> BG Value (mg/dL)                                                     |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Rincian Spesifikasi Input Sinyal & Durasi Waktu (ALL 5 MIN)

| Modalitas     | Durasi Waktu              | Sampling Rate | Jumlah Sampel | Preprocessing Filter           | Cabang Output Segmentasi                                   |
| ------------- | ------------------------- | ------------- | ------------- | ------------------------------ | ---------------------------------------------------------- |
| **ECG** | **5 menit (300 s)** | 250 Hz        | 75.000 sampel | Bandpass Filter (0.5 – 40 Hz) | 1. Consecutive cardiac cycles2. Individual segmented beats |
| **PPG** | **5 menit (300 s)** | 64 Hz         | 19.200 sampel | Bandpass Filter (0.5 – 8 Hz)  | 1. Consecutive pulses2. Individual segmented pulses        |
| **EDA** | **5 menit (300 s)** | 4 Hz          | 1.200 sampel  | Z-Score Normalization          | Sinyal ter-normalisasi utuh                                |

> **Catatan Penting Sesuai Diagram:**
>
> - **SEMUA SINYAL (ECG, PPG, EDA) BERDURASI SERAGAM 5 MENIT (300 DETIK)**.
> - Tidak ada Attention Pooling (Attention Pooling didepresiasi/dihapus).
> - Tidak ada sensor TEMP (Suhu). Model murni menggunakan EDA 1-channel.

---

## 3. Rincian Fitur & Ekstraksi Model

### A. Temporal Statistical Features ($F_{temporal}$)

- **Input:** Consecutive cardiac cycles (ECG) & Consecutive pulses (PPG) dari window 5 menit.
- **Metode Dekomposisi:** Discrete Wavelet Transform (DWT) menggunakan mother wavelet `db4` hingga 7 level.

  - Sub-bands ECG: $\{ECG_{raw}, ECG_{detail 1}, \dots, ECG_{detail 7}\}$
  - Sub-bands PPG: $\{PPG_{raw}, PPG_{detail 1}, \dots, PPG_{detail 7}\}$
- **Ekstraksi Fitur:** 10 fitur statistik dihitung untuk setiap sub-band:

  1. Kurtosis (Kur)
  2. Skewness (Ske)
  3. Spectral Mean (SM)
  4. Spectral Centroid (SC)
  5. Frequency Domain features (FD)
  6. Complexity Domain features (CD)
  7. Cosine / Correlation (CO)
  8. Power Spectral Entropy (PSE)
  9. Energy / Shannon Entropy (KE)
  10. Spectral Entropy (SE)

  - *Total awal:* 10 fitur $\times$ 8 sub-band $\times$ 2 modalitas = **160 fitur**.
- **Seleksi Fitur (Feature Selection):** Gabungan Unsupervised Feature Selection (UFS), L1-based Feature Selection (L1-FS), dan Recursive Feature Elimination (RFE) $\rightarrow$ Menghasilkan **$F_{temporal}$ ($\sim$18 fitur)**.

### B. Spatial Morphological Features ($F_{morphological}$)

- **Input:** Individual segmented beats (ECG) & Individual segmented pulses (PPG) dari window 5 menit.
- **Backbone Model:** Dual-branch 1D-ResNet (5-layer):
  - Branch ECG: `nn.Conv1d` $\rightarrow$ [`ConvBlock` + `IdentityBlock`] $\times 4 \rightarrow f(x_{ECG}) + x_{ECG}$
  - Branch PPG: `nn.Conv1d` $\rightarrow$ [`ConvBlock` + `IdentityBlock`] $\times 4 \rightarrow f(x_{PPG}) + x_{PPG}$
- **DNN Feature Compression:**
  - Concatenate embedding ECG & PPG
  - `Linear(in, 512)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(512, 128)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(128, 40)`
  - Menghasilkan **$F_{morphological}$ (40 fitur)**.

### C. EDA Features ($F_{eda}$)

- **Input:** Z-score normalized EDA (1200 sampel dari window 5 menit @ 4Hz).
- **Conv1D Encoder:**
  - Layer 1: `Conv1d(in_channels=1, out_channels=8, kernel_size=12, stride=6)`
  - Layer 2: `Conv1d(in_channels=8, out_channels=16, kernel_size=8, stride=5)`
- **Bi-LSTM Layer:**
  - `nn.LSTM(input_size=16, hidden_size=32, bidirectional=True, batch_first=True)`
- Menghasilkan **$F_{eda}$ (64 fitur)**.

### D. Fusion & Multi-Task Prediction

- **Feature Fusion (Concatenation):**
  $$
  F_{fusion} = \{F_{temporal} (\sim18), F_{morphological} (40), F_{eda} (64)\} \implies \sim122 \text{ fitur}
  $$
- **Shared MLP:**
  - `Linear(122, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(64, 32)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)`
- **Dual Output Heads:**
  1. **Classification Head:** `Linear(32, 1)` $\rightarrow$ `Sigmoid` (Prediksi: Hypo / Normal)
  2. **Regression Head:** `Linear(32, 1)` (Prediksi: Nilai Glukosa Darah dalam mg/dL)

---

## 4. Evaluasi Kompatibilitas Data IoT (`IOT-prototype-deteksi-nyeri`)

1. **Kesesuaian Durasi 5 Menit:**
   - Perangkat IoT direkam dalam durasi 5 menit per sesi. Karena ketiga modalitas (ECG, PPG, EDA) di arsitektur ST-MGM ini semuanya berdurasi 5 menit (300 detik), maka data IoT **langsung 1:1 kompatibel secara temporal**!
2. **Tanpa Sensor Suhu (TEMP):**
   - Model ST-MGM murni hanya butuh EDA, sehingga alat IoT yang tidak punya sensor suhu sudah **100% kompatibel**.
3. **Resampling / Downsampling Data IoT:**
   - File `BVP.csv` dan `EDA.csv` di-log pada 250Hz oleh firmware ESP32.
   - Skrip `preprocess_iot.py` akan me-resample:
     - BVP (PPG): 250 Hz $\rightarrow$ 64 Hz (menjadi 19.200 sampel)
     - EDA: 250 Hz $\rightarrow$ 4 Hz (menjadi 1.200 sampel)
     - ECG: 250 Hz (tetap 75.000 sampel)
4. **Bandpass Filter:**
   - ECG dari AD8232 disaring dengan Bandpass Filter 0.5–40 Hz.
   - PPG dari MAX30102 disaring dengan Bandpass Filter 0.5–8 Hz.

---

## 5. Rencana Modul & File Kode Python yang Akan Dibuat

```
physio-stf/
├── IMPLEMENTATION_PLAN.md        ← Dokumen acuan ini
├── requirements.txt              ← Dependency (PyTorch, PyWavelets/scikit-learn, NeuroKit2, scipy)
├── configs/
│   └── st_mgm_config.yaml        ← Parameter hiper-model & training
├── libs/
│   ├── __init__.py
│   ├── dsp.py                    ← Bandpass Filter (0.5-40Hz & 0.5-8Hz) & Z-score
│   ├── dwt_features.py           ← DWT db4 7-level + 10 Stat Features + Feature Selection
│   ├── resnet1d.py               ← 1D-ResNet (5-layer) untuk ECG & PPG + DNN Compression
│   ├── eda_encoder.py            ← Conv1D (2-layer) + Bi-LSTM
│   ├── st_mgm_model.py           ← Multi-Task Fusion Model & Shared MLP
│   └── dataloader.py             ← PyTorch Dataset untuk ST-MGM (5-min window)
├── preprocess_raw.py             ← Potong window 5 menit seragam (ECG, PPG, EDA) + Filtering
├── preprocess_features.py        ← Ekstraksi DWT & ResNet embeddings
├── preprocess_iot.py             ← Adapter data IoT (Resampling 250Hz -> 64Hz & 4Hz)
├── train.py                      ← Multi-task training loop (Loss = BCE + MSE)
├── test.py                       ← Evaluasi metrik Klasifikasi (AUC, F1) & Regresi (RMSE, MARD)
└── run_all.py                    ← Master script penjalankan seluruh pipeline
```

---

## 6. Prioritas Urutan Pengerjaan Kode

1. **Phase 1 (DSP & DWT Feature Extractor):** `libs/dsp.py` dan `libs/dwt_features.py` (Filter sinyal, DWT db4 7-level, 10 fitur statistik, UFS/L1-FS/RFE).
2. **Phase 2 (Deep Learning Modules):** `libs/resnet1d.py`, `libs/eda_encoder.py`, dan `libs/st_mgm_model.py` (1D-ResNet 5-layer, Bi-LSTM, Fusion Concat, Shared MLP, Classification & Regression Heads).
3. **Phase 3 (Preprocessing & Dataset Slicing):** `preprocess_raw.py` (ECG 5m @ 250Hz, PPG 5m @ 64Hz, EDA 5m @ 4Hz) dan `libs/dataloader.py`.
4. **Phase 4 (Multi-Task Training & Loss):** `train.py` dengan gabungan Loss (Binary Cross Entropy + Mean Squared Error).
5. **Phase 5 (IoT Adapter & Master Script):** `preprocess_iot.py` (Downsample 250Hz $\rightarrow$ 64Hz & 4Hz) dan `run_all.py`.
