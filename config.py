#3px = 6-9 km
GRID_PX = 3          
OUTPUT_DIR = "output"

ISLAND_PROVINCES: dict[str, list[str]] = {
    "JAWA": [
        "DKI JAKARTA", "JAWA BARAT", "JAWA TENGAH", 
        "DAERAH ISTIMEWA YOGYAKARTA", "JAWA TIMUR", "BANTEN"
    ],
    "SUMATRA": [        
        "DI. ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "JAMBI", 
        "SUMATERA SELATAN", "BENGKULU", "LAMPUNG", 
        "BANGKA BELITUNG", "KEPULAUAN RIAU"
    ],
    "KALIMANTAN": [
        "KALIMANTAN BARAT", "KALIMANTAN TENGAH", "KALIMANTAN SELATAN", 
        "KALIMANTAN TIMUR", "KALIMANTAN UTARA"
    ],
    "SULAWESI": [
        "SULAWESI UTARA", "SULAWESI TENGAH", "SULAWESI SELATAN", 
        "SULAWESI TENGGARA", "GORONTALO", "SULAWESI BARAT"
    ],
    "PAPUA": [
        "PAPUA BARAT", "PAPUA","MALUKU", "MALUKU UTARA"
    ],
    "NUSA TENGGARA": [
        "BALI", "NUSATENGGARA BARAT", "NUSATENGGARA TIMUR"
    ],
}

ISLAND_GEOREF: dict[str, dict] = {
    "JAWA": {
        "lon": {"px1": 50, "lon1": 105.2, "px2": 550, "lon2": 115.9},
        "lat": {"py1": 30, "lat1": -5.7,  "py2": 230, "lat2": -8.80},
        "px_min": 40, "py_min": 20,
        "bbox_geo": [105.2,-8.8,115.9,-5.7],
    },
    "SUMATRA": {
        "lon": {"px1": 50, "lon1": 95.2,  "px2": 550, "lon2": 109.12},
        "lat": {"py1": 30, "lat1": 5.88,   "py2": 400, "lat2": -5.94},
        "px_min": 10, "py_min": 20,
        "bbox_geo": [ 95.2, -5.9, 109.1, 5.9],
    },
    "KALIMANTAN": {
        "lon": {"px1": 50, "lon1": 108.69, "px2": 550, "lon2": 118.99},
        "lat": {"py1": 30, "lat1": 4.41,   "py2": 420, "lat2": -4.7},
        "px_min": 15, "py_min": 40,
        "bbox_geo": [108.7, -4.7, 119.0, 4.4],
    },
    "SULAWESI": {
        "lon": {"px1": 50, "lon1": 117.66, "px2": 450, "lon2": 126.91},
        "lat": {"py1": 30, "lat1": 4.56,   "py2": 380, "lat2": -7.49},
        "px_min": 20, "py_min": 20,
        "bbox_geo": [117.7, -7.5, 126.9, 4.6],
    },
    "PAPUA": {
        "lon": {"px1": 50, "lon1": 124.3, "px2": 550, "lon2": 141.01},
        "lat": {"py1": 30, "lat1": 2.65,   "py2": 400, "lat2": -9.12},
        "px_min": 40, "py_min": 20,
        "bbox_geo": [124.3, -9.1, 141.0, 2.6]
    },
    "NUSA TENGGARA": {
        "lon": {"px1": 87, "lon1": 114.43, "px2": 558, "lon2": 125.64},
        "lat": {"py1": 78, "lat1": -8.06,  "py2": 240, "lat2": -10.92},
        "px_min": 18, "py_min": 72,
        "bbox_geo": [114.4, -10.9,  125.6, -8.1],
    },
}

LEGEND: dict[int, dict] = {
    1: {
        "name": "Sangat Basah",  "spi_range": "> 2.0",
        "spi_value": 2.5,
        "rgb": [0, 167, 229],    "hex": "#00A7E5",
        "stroke": "#005F8A",     "fill_opacity": 0.85,
        "stroke_width": 1.2,    "spin_rgb": [0, 120, 200],
    },
    2: {
        "name": "Basah",         "spi_range": "1.5 s/d 2.0",
        "spi_value": 1.75,
        "rgb": [56, 167, 0],     "hex": "#38A700",
        "stroke": "#1A5C00",     "fill_opacity": 0.85,
        "stroke_width": 1.2,    "spin_rgb": [30, 140, 0],
    },
    3: {
        "name": "Agak Basah",    "spi_range": "1.0 s/d 1.49",
        "spi_value": 1.25,
        "rgb": [152, 229, 0],    "hex": "#98E500",
        "stroke": "#5A8A00",     "fill_opacity": 0.82,
        "stroke_width": 1.0,    "spin_rgb": [100, 200, 0],
    },
    4: {
        "name": "Normal",        "spi_range": "-0.99 s/d 0.99",
        "spi_value": 0.0,
        "rgb": [255, 255, 190],  "hex": "#FFFFBE",
        "stroke": "#AAAA60",     "fill_opacity": 0.75,
        "stroke_width": 0.8,    "spin_rgb": [220, 220, 140],
    },
    5: {
        "name": "Agak Kering",   "spi_range": "-1.0 s/d -1.49",
        "spi_value": -1.25,
        "rgb": [230, 152, 0],    "hex": "#E69800",
        "stroke": "#8A5500",     "fill_opacity": 0.85,
        "stroke_width": 1.2,    "spin_rgb": [200, 110, 0],
    },
    6: {
        "name": "Kering",        "spi_range": "-1.5 s/d -2.0",
        "spi_value": -1.75,
        "rgb": [230, 0, 0],      "hex": "#E60000",
        "stroke": "#8A0000",     "fill_opacity": 0.88,
        "stroke_width": 1.5,    "spin_rgb": [200, 0, 0],
    },
    7: {
        "name": "Sangat Kering", "spi_range": "< -2.0",
        "spi_value": -2.5,
        "rgb": [114, 0, 0],      "hex": "#720000",
        "stroke": "#3A0000",     "fill_opacity": 0.90,
        "stroke_width": 1.8,    "spin_rgb": [90, 0, 0],
    },
}


GEOJSON_URLS: list[str] = [
    "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson",
]

BOUNDARY_DEFAULT: dict = {
    "stroke": "#333333",
    "stroke-width": 2.0,
    "fill": "none",
    "fill-opacity": 0,
}
