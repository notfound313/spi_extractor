from __future__ import annotations
import csv
import io
from typing import List


# Kolom yang diekspor ke CSV (urutan tetap)
CSV_COLUMNS: list[str] = [
    "provinsi",
    "spi_category",
    "spi_range",
    "spi_value",
    "lon_center",
    "lat_center",
]


def features_to_csv_bytes(features: List[dict]) -> bytes:
    """
    Konversi list GeoJSON Feature ke bytes CSV siap-unduh.

    Parameters
    ----------
    features : list GeoJSON Feature dari classify_grid()

    Returns
    -------
    bytes UTF-8 berisi CSV lengkap (header + baris data).
    """
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=CSV_COLUMNS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()

    for feat in features:
        props = feat.get("properties", {})
        writer.writerow(
            {
                "provinsi":     props.get("provinsi", ""),
                "spi_category": props.get("spi_category", ""),
                "spi_range":    props.get("spi_range", ""),
                "spi_value":    props.get("spi_value", ""),
                "lon_center":   props.get("lon_center", ""),
                "lat_center":   props.get("lat_center", ""),
            }
        )

    return buf.getvalue().encode("utf-8")


def features_to_records(features: List[dict]) -> List[dict]:
    """
    Konversi list GeoJSON Feature ke list of dict (untuk pandas / st.dataframe).

    Returns
    -------
    List of dict dengan kolom CSV_COLUMNS.
    """
    records = []
    for feat in features:
        props = feat.get("properties", {})
        records.append(
            {
                "provinsi":     props.get("provinsi", ""),
                "spi_category": props.get("spi_category", ""),
                "spi_range":    props.get("spi_range", ""),
                "spi_value":    props.get("spi_value", ""),
                "lon_center":   props.get("lon_center", ""),
                "lat_center":   props.get("lat_center", ""),
            }
        )
    return records