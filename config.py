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
        "BALI", "NUSATENGGARA BARAT", "NUSA TENGGARA TIMUR", "MALUKU"
    ],
}

ISLAND_GEOREF: dict[str, dict] = {
    "JAWA": {
        "master_size": (912, 412),
        "gcps": [
            {"px":   0, "py":   0, "lon": 105.20, "lat": -5.70},
            {"px": 910, "py":   0, "lon": 115.90, "lat": -5.70},
            {"px":   0, "py": 409, "lon": 105.20, "lat": -8.80},
            {"px": 910, "py": 409, "lon": 115.90, "lat": -8.80},
        ],
    },
    "KALIMANTAN": {
        "master_size": (383, 393),
        "gcps": [
            {"px":   0, "py":  37, "lon": 108.69, "lat":  4.41},
            {"px": 380, "py":  37, "lon": 118.99, "lat":  4.41},
            {"px":   0, "py": 390, "lon": 108.69, "lat": -4.70},
            {"px": 380, "py": 390, "lon": 118.99, "lat": -4.70},
        ],
    },
    "NUSA TENGGARA": {
        "master_size": (572, 262),
        "gcps": [
            {"px":   0, "py":   0, "lon": 114.43, "lat":  -8.06},
            {"px": 568, "py":   0, "lon": 125.64, "lat":  -8.06},
            {"px":   0, "py": 260, "lon": 114.43, "lat": -10.92},
            {"px": 568, "py": 260, "lon": 125.64, "lat": -10.92},
        ],
    },
    "PAPUA": {
        "master_size": (380, 398),
        "gcps": [
            {"px":   0, "py":   0, "lon": 124.30, "lat":   2.65},
            {"px": 378, "py":   0, "lon": 141.01, "lat":   2.65},
            {"px":   0, "py": 394, "lon": 124.30, "lat":  -9.12},
            {"px": 378, "py": 394, "lon": 141.01, "lat":  -9.12},
        ],
    },
    "SULAWESI": {
        "master_size": (384, 397),
        "gcps": [
            {"px":   0, "py":  34, "lon": 117.66, "lat":   4.56},
            {"px": 382, "py":  34, "lon": 126.91, "lat":   4.56},
            {"px":   0, "py": 394, "lon": 117.66, "lat":  -7.49},
            {"px": 382, "py": 394, "lon": 126.91, "lat":  -7.49},
        ],
    },
    "SUMATRA": {
        "master_size": (380, 397),
        "gcps": [
            {"px":   0, "py":   0, "lon":  95.20, "lat":   5.88},
            {"px": 378, "py":   0, "lon": 109.12, "lat":   5.88},
            {"px":   0, "py": 394, "lon":  95.20, "lat":  -5.94},
            {"px": 378, "py": 394, "lon": 109.12, "lat":  -5.94},
        ],
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
 