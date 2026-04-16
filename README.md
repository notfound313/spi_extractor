# SPI Grid Extractor

Aplikasi Streamlit untuk mengekstrak klasifikasi **Standardized Precipitation Index (SPI)**
dari citra peta raster ke format grid GeoJSON bergeoreferensi, lengkap dengan anotasi nama provinsi.

---

## Struktur Proyek

```
spi_extractor/
├── app.py              # Antarmuka Streamlit (entry point)
├── config.py           # Semua konstanta: LEGEND, ISLAND_GEOREF, ISLAND_PROVINCES
├── georef.py           # Fungsi konversi piksel <-> koordinat geografis
├── masking.py          # Segmentasi daratan/lautan & ekstraksi poligon
├── spatial.py          # Unduh batas provinsi & spatial join (point-in-polygon)
├── classification.py   # Klasifikasi grid warna ke kategori SPI + build GeoJSON
├── visualization.py    # Render peta output PNG dengan legenda
└── requirements.txt    # Dependensi Python
```

---

## Instalasi

```bash
pip install -r requirements.txt
```

---

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

---

## Cara Penggunaan

1. **Pilih Pulau Target** di sidebar (JAWA, SUMATRA, KALIMANTAN, SULAWESI, PAPUA, NUSA TENGGARA).
2. **Unggah citra peta** SPI dalam format PNG, JPG, atau TIFF.
3. Atur **Ukuran Grid** (piksel) sesuai resolusi yang diinginkan.
4. Aktifkan/nonaktifkan **Spatial Join Provinsi** sesuai kebutuhan.
5. Tekan **Proses Ekstraksi**.
6. Setelah selesai, unduh:
   - **Peta PNG** — visualisasi SPI dengan legenda warna.
   - **Grid GeoJSON** — data grid untuk GIS / geojson.io.
   - **Semua File (ZIP)** — keduanya dalam satu arsip.

---

## Output GeoJSON

Setiap fitur (grid) memiliki properti:

| Properti       | Keterangan                             |
|----------------|----------------------------------------|
| `grid_id`      | Identifier unik sel grid (`G_x_y`)     |
| `provinsi`     | Nama provinsi hasil spatial join       |
| `spi_category` | Kategori SPI (mis. "Kering")           |
| `spi_range`    | Rentang nilai SPI                      |
| `spi_value`    | Nilai numerik representatif            |
| `lon_center`   | Longitude pusat sel                    |
| `lat_center`   | Latitude pusat sel                     |
| `fill`         | Warna HEX untuk GeoJSON.io             |
| `fill-opacity` | Opasitas warna                         |
| `stroke`       | Warna border                           |

---

## Konfigurasi

Semua parameter utama ada di **`config.py`**:

- `GRID_PX` — ukuran grid default dalam piksel
- `LEGEND` — definisi kategori dan warna SPI
- `ISLAND_GEOREF` — parameter georeferensi per pulau
- `ISLAND_PROVINCES` — daftar provinsi per pulau
- `GEOJSON_URLS` — sumber data batas provinsi

Untuk menambah pulau baru, tambahkan entri di `ISLAND_GEOREF` dan `ISLAND_PROVINCES`.

---

## Dependensi

- `streamlit` — antarmuka web
- `opencv-python-headless` — pengolahan citra
- `numpy` — komputasi numerik
