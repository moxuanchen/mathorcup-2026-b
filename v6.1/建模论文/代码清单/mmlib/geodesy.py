"""坐标转换：UTM 45N ↔ WGS84 经纬度（精确横轴墨卡托反算）。"""
from __future__ import annotations

import math

A = 6378137.0
F = 1 / 298.257223563
K0 = 0.9996
E2 = F * (2 - F)
EP2 = E2 / (1 - E2)


def _meridian_arc(phi: float) -> float:
    e4, e6 = E2 * E2, E2 * E2 * E2
    return A * ((1 - E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
                - (3 * E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
                + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
                - (35 * e6 / 3072) * math.sin(6 * phi))


def utm_to_lonlat(easting: float, northing: float, zone: int = 45, south: bool = False):
    x = easting - 500000.0
    y = northing - (10000000.0 if south else 0.0)
    mu = y / K0 / (A * (1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - E2)) / (1 + math.sqrt(1 - E2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    sin1, cos1, tan1 = math.sin(phi1), math.cos(phi1), math.tan(phi1)
    c1 = EP2 * cos1 ** 2
    t1 = tan1 ** 2
    n1 = A / math.sqrt(1 - E2 * sin1 ** 2)
    r1 = A * (1 - E2) / (1 - E2 * sin1 ** 2) ** 1.5
    d = x / (n1 * K0)
    lat = phi1 - (n1 * tan1 / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * EP2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * EP2 + 24 * t1 ** 2) * d ** 5 / 120) / cos1
    return math.degrees(lon) + (zone * 6 - 183), math.degrees(lat)


def lonlat_to_utm(lon: float, lat: float, zone: int = 45):
    phi, lam = math.radians(lat), math.radians(lon)
    lam0 = math.radians(zone * 6 - 183)
    n = A / math.sqrt(1 - E2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = EP2 * math.cos(phi) ** 2
    aa = (lam - lam0) * math.cos(phi)
    m = _meridian_arc(phi)
    east = K0 * n * (aa + (1 - t + c) * aa ** 3 / 6
                     + (5 - 18 * t + t ** 2 + 72 * c - 58 * EP2) * aa ** 5 / 120) + 500000.0
    north = K0 * (m + n * math.tan(phi) * (
        aa ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * aa ** 4 / 24
        + (61 - 58 * t + t ** 2 + 600 * c - 330 * EP2) * aa ** 6 / 720))
    return east, north
