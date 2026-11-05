"""吉隆泥石流建模：公共数据读写与预处理模块。

所有路径均以项目根目录为基准，可通过环境变量 JILONG_ROOT 覆盖。
原始附件解压目录默认为 <root>/临时产物/原始数据。
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径

def project_root() -> Path:
    """定位工程根目录（含 求解结果/ 与 临时产物/ 的目录）。

    优先取环境变量 JILONG_ROOT；否则自本文件向上逐级搜索，命中第一个同时含有
    `求解结果` 与 `临时产物` 的目录。**不能用固定的 parents[N]**：源码位于
    `支撑材料/源码/mmlib/`，parents[2] 是 `支撑材料` 而不是工程根，早期实现因此
    在未设置环境变量的全新 shell 中直接找不到原始数据（`python run_all.py`
    这条可复现命令实际不可用）。向上搜索使"在哪一层被拷入"都不影响定位。
    """
    env = os.environ.get("JILONG_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "求解结果").is_dir() and (cand / "临时产物").is_dir():
            return cand
    # 兜底：若目录结构尚未建立，退回三层（源码被放在 <root>/源码/ 的旧布局）
    return here.parents[3]


ROOT = project_root()
RAW = ROOT / "临时产物" / "原始数据"
RESULT = ROOT / "求解结果"
DATA_OUT = RESULT / "数据"
FIG_OUT = RESULT / "图表"
CACHE = DATA_OUT / "cache"

for _d in (DATA_OUT, FIG_OUT, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

STATION_DIR = RAW / "（1）西藏吉隆高寒山区地气相互作用逐小时综合观测数据集（2021–2026）"
TEMPERATURE_DIR = RAW / "（2）2005-2025年吉隆县及周边县市逐日平均气温数据" / "2005-2025年吉隆县周边县市逐日平均气温数据"
GLACIER_DHDT_DIR = RAW / "（3）冰川高程变化" / "西藏日喀则市吉隆县特大洪水—200km周围内的冰川和冰川厚度变化数据集"
GLACIER_NC = RAW / "（4）吉隆口岸山洪灾害-蓝塘里壤峰冰崩前冰川地形、冰厚及表面速度数据集" / (
    "Glacier topographic and dynamic drivers (Glacier-collapse and debris-flow disasters at Langtang Lirung, Nepal).nc")
LANDCOVER_DIR = RAW / "（5）西藏日喀则市吉隆县泥石流灾害区土地覆盖数据（2020-2023）"
HAZARD_DIR = RAW / "（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据" / "西藏日喀则市吉隆县泥石流灾害区历史灾点数据"
DEM_DIR = RAW / "（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据"
CENSUS_DIR = RAW / "（8）西藏日喀则市吉隆县泥石流灾害区吉隆县第七次人口普查数据" / "西藏日喀则市吉隆县第七次人口普查数据"

# 站点元信息（坐标与海拔均为假定/推断值，非实测安装元数据，见 问题与数据说明.md）
# elev_m 取气压反演值 3978.85 m；触发计算中的 3980.0 是同一推断值的取整工作常数，
# 两者差 1.15 m，远小于反演不确定度，不构成两套口径。
# 晴空辐射模型（meteo.clear_sky_rsd）为 Haurwitz 形式，不含海拔项，故辐射结果不使用 3980.0；
# 该参数仅为接口保留，不参与计算。
STATION = {
    "name": "吉隆高寒山区综合观测站",
    "lon": 85.30,
    "lat": 28.40,
    "elev_m": 3978.85,
    "elev_m_deprecated": 3955.0,
    "elev_m_note": (
        "elev_m 取本站气压反演值 3978.85 m；elev_m_deprecated = 3955.0 为 v2 遗留配置值，"
        "旧版推断值仅留作审计痕迹，不作为站点海拔。elev_m 不参与任何下游计算。"
    ),
    "period": ("2021-04-23 11:00", "2026-08-27 08:00"),
    "index_first": "2021-01-01 00:00",
    "note": (
        "坐标为吉隆沟高寒山区假定代表点（原始 CSV 无站点元数据）；"
        "海拔为标准大气由本站气压反演值，非实测安装高度；"
        "逐时序列按 2021-01-01 00:00 起的完整小时轴重采样，故 rows 大于 period 跨度对应的小时数；"
        "反演细节见 求解结果/数据/00_站点元信息.json 的 elevation_inferred_m"
    ),
}

# 事件时刻（题面给定）与口岸位置（非题面给定，见 note）
EVENT = {
    "name": "2026-08-26 吉隆口岸冰崩-泥石流",
    "datetime_local": "2026-08-26 10:30",
    # 口岸工作坐标：研究区 DEM 图上量取的近似位置，题面未给坐标。
    # 公开地理坐标为 28.2797N/85.3777E，与本工作坐标相差约 571 m。
    # 该距离约合 19--21 个 DEM 像元（像元 27.2 x 30.7 m），并非小于像元尺度；
    # 它小于口岸集水区与模拟域尺度，故空间结果统一以此工作坐标为准，
    # 并在敏感性分析中把路径吸附偏移量级一并披露。
    "port_lon": 85.3826,
    "port_lat": 28.2826,
    "port_note": "量图近似位置，非题面给定、非实测；候选位置仍待现场核实；重演见求解结果/数据/06_坐标敏感性.json",
}

TAG = 9999.0  # 缺测标记
PREPROCESS_VERSION = "qc-center-v4"

# ---------------------------------------------------------------- TIFF

def _lzw_decode(data: bytes) -> bytes:
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    out = bytearray()
    nbits, bitpos, total, prev = 9, 0, len(data) * 8, None
    while bitpos + nbits <= total:
        b = bitpos >> 3
        chunk = data[b:b + 4].ljust(4, b"\x00")
        shift = 32 - (bitpos & 7) - nbits
        if shift < 0:
            break
        code = (int.from_bytes(chunk, "big") >> shift) & ((1 << nbits) - 1)
        bitpos += nbits
        if code == 256:
            table, nbits, prev = table[:258], 9, None
            continue
        if code == 257:
            break
        if prev is None:
            entry = table[code]
            out += entry
            prev = entry
            continue
        if code < len(table):
            entry = table[code]
            table.append(prev + entry[:1])
        else:
            entry = prev + prev[:1]
            table.append(entry)
        out += entry
        prev = entry
        if len(table) == (1 << nbits) - 1 and nbits < 12:
            nbits += 1
    return bytes(out)


_DTYPE = {(3, 32): "f4", (3, 64): "f8", (1, 8): "u1", (1, 16): "u2",
          (1, 32): "u4", (2, 8): "i1", (2, 16): "i2", (2, 32): "i4"}


def read_geotiff(path: str | Path):
    """读取 GeoTIFF（支持 LZW/Deflate/无压缩，条带或瓦片），返回 (数组, 地理信息)。"""
    import tifffile
    path = str(path)
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = {t.code: t.value for t in page.tags.values()}
        width, height = int(tags[256]), int(tags[257])
        comp = int(tags.get(259, 1))
        spp = int(tags.get(277, 1))
        bits = tags.get(258, 32)
        bits = int(bits[0]) if isinstance(bits, (list, tuple)) else int(bits)
        sf = tags.get(339, 1)
        sf = int(sf[0]) if isinstance(sf, (list, tuple)) else int(sf)
        dtype = _DTYPE[(sf, bits)]
        bps = bits // 8
        fh = tf.filehandle
        tw = int(tags.get(322, 0) or 0)
        th = int(tags.get(323, 0) or 0)

        def unpack(blob: bytes) -> bytes:
            if comp == 1:
                return blob
            if comp == 5:
                return _lzw_decode(blob)
            if comp in (8, 32946):
                return zlib.decompress(blob)
            raise ValueError(f"不支持的压缩类型 {comp}")

        if tw and th:
            across = (width + tw - 1) // tw
            out = np.zeros(((height + th - 1) // th * th, across * tw), dtype=np.dtype(dtype))
            need = tw * th * spp * bps
            for idx, (off, cnt) in enumerate(zip(tags[324], tags[325])):
                fh.seek(off)
                blob = unpack(fh.read(cnt))
                if len(blob) < need:
                    blob += b"\x00" * (need - len(blob))
                arr = np.frombuffer(blob[:need], dtype=dtype).reshape(th, tw, spp) if spp > 1 else \
                    np.frombuffer(blob[:need], dtype=dtype).reshape(th, tw)
                r, c = divmod(idx, across)
                out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = arr
            arr = out[:height, :width]
        else:
            chunks = []
            for off, cnt in zip(tags[273], tags[279]):
                fh.seek(off)
                arr_c = np.frombuffer(unpack(fh.read(cnt)), dtype=dtype)
                chunks.append(arr_c)
            arr = np.concatenate(chunks)
            arr = arr.reshape(height, width, spp) if spp > 1 else arr[:height * width].reshape(height, width)
    geo = {}
    for key, code in (("pixel_scale", 33550), ("tiepoint", 33922), ("geokeys", 34735),
                      ("gcs_ascii", 34737), ("nodata", 42113)):
        if code in tags:
            geo[key] = tags[code]
    return np.asarray(arr), geo


def geotiff_grid(geo: dict) -> dict:
    """由 GeoTIFF 标签给出 (lon0, lat0, dlon, dlat) 栅格定位（左上角像元中心为锚点）。"""
    sx, sy = float(geo["pixel_scale"][0]), float(geo["pixel_scale"][1])
    tp = geo["tiepoint"]
    keys = list(geo.get("geokeys", []))
    raster_type = 1
    for i in range(4, len(keys) - 3, 4):
        if keys[i] == 1025:
            raster_type = int(keys[i + 3])
    half = 0.5 if raster_type == 1 else 0.0
    return {"lon0": float(tp[3]) + (half - float(tp[0])) * sx,
            "lat0": float(tp[4]) - (half - float(tp[1])) * sy,
            "dlon": sx, "dlat": -sy, "anchor": "pixel_center",
            "raster_type": raster_type, "preprocess_version": PREPROCESS_VERSION}


def lonlat_to_rowcol(grid: dict, lon, lat):
    return (np.asarray(lat) - grid["lat0"]) / grid["dlat"], (np.asarray(lon) - grid["lon0"]) / grid["dlon"]


# ---------------------------------------------------------------- 站点观测

_MET_FMT = "%Y-%m-%d %H:%M"
_FLUX_FMT = "%Y/%m/%d %H:%M"


def _load_family(tag: str, fmt: str) -> pd.DataFrame:
    files = sorted(STATION_DIR.glob(f"{tag}_Gyirong_*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到 {tag} 观测文件：{STATION_DIR}")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8")
        df["ts"] = pd.to_datetime(df["Timestamp"], format=fmt)
        frames.append(df.drop(columns=["Timestamp"]))
    data = pd.concat(frames, ignore_index=True).set_index("ts").sort_index()
    data = data[~data.index.duplicated(keep="last")]
    num = data.select_dtypes("number").columns
    data[num] = data[num].mask(data[num].abs() >= TAG)
    stems = {_clean_name(c).lower(): c for c in data if not c.startswith("QC_")}
    for qc in [c for c in data if c.startswith("QC_")]:
        target = stems.get(qc[3:].lower())
        if target:
            data[target] = data[target].where(data[qc].eq(0))
    return data


def _clean_name(c: str) -> str:
    return (c.replace("(degree C)", "").replace("(degree)", "").replace("(m/s)", "").replace("(%)", "")
             .replace("(hPa)", "").replace("(mm)", "").replace("(W/m2)", "").replace("(m3/m3)", "")
             .replace("(kPa)", "").replace("(g/m3)", "").replace("(mg/m2*s)", "").replace("(kg/m*s2)", "")
             .replace("(m2/s2)", "").replace("(m)", "").strip())


def load_station(use_cache: bool = True) -> pd.DataFrame:
    """合并 MET/RADM/SOIL/FLUX 四类逐小时观测，返回带统一列名的 DataFrame。"""
    cache = CACHE / "station_hourly.parquet"
    if use_cache and cache.exists():
        old = pd.read_parquet(cache)
        if old.attrs.get("preprocess_version") == PREPROCESS_VERSION:
            return old

    met = _load_family("MET", _MET_FMT)
    met = met.drop(columns=[c for c in met.columns if c.startswith("QC_")])
    met.columns = [_clean_name(c) for c in met.columns]

    radm = _load_family("RADM", _MET_FMT)
    radm = radm.drop(columns=[c for c in radm.columns if c.startswith("QC_")])
    radm.columns = [_clean_name(c) for c in radm.columns]

    soil = _load_family("SOIL", _MET_FMT)
    soil = soil.drop(columns=[c for c in soil.columns if c.startswith("QC_")])
    soil.columns = [_clean_name(c) for c in soil.columns]

    flux = _load_family("FLUX", _FLUX_FMT)
    flux.columns = [_clean_name(c) for c in flux.columns]

    df = met.join(radm, how="outer").join(soil, how="outer").join(flux, how="outer")
    df.index.name = "ts"
    df = df.loc[:"2026-08-27 08:00"]
    # 统一为浮点，剔除同一列名冲突（RADM 的 QC_rsu 等已在上游删除）
    df = df.loc[:, ~df.columns.duplicated()]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.astype("float64")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h", name="ts"))
    for c in df:
        if c.startswith("RH_"):
            df[c] = df[c].where(df[c].between(0, 100))
        elif c.startswith("WS_") or c == "Precip":
            df[c] = df[c].where(df[c] >= 0)
        elif c.startswith("Ta_"):
            df[c] = df[c].where(df[c].between(-60, 50))
    df.attrs["preprocess_version"] = PREPROCESS_VERSION
    df.to_parquet(cache)
    return df


def load_temperature(use_cache: bool = True) -> pd.DataFrame:
    """五个县 2005–2025 年逐日 2 m 平均气温。"""
    cache = CACHE / "temperature_daily.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for f in sorted(TEMPERATURE_DIR.glob("2米气温数据_*.xlsx")):
        df = pd.read_excel(f, sheet_name=0)
        df = df.rename(columns={"时间": "date"})
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("date").set_index("date").sort_index()
    out.to_parquet(cache)
    return out


# ---------------------------------------------------------------- 矢量

def _dbf_encoding(dbf_path: Path) -> str:
    cpg = dbf_path.with_suffix(".cpg")
    if cpg.exists():
        txt = cpg.read_text(encoding="ascii", errors="ignore").strip().lower()
        if "utf" in txt:
            return "utf-8"
        if "936" in txt or "gbk" in txt or "gb2312" in txt:
            return "gbk"
    raw = dbf_path.read_bytes()
    # 逐个字段名检验，不能只看文件头前 64 字节：首个字段常为 ASCII
    # （Name/flag/x_coord…），只看开头会把 GBK 表误判为 UTF-8，中文字段名
    # （河流/课题/沟比降…）在 utf-8+ignore 下塌缩为空串并在 dict 中互相
    # 覆盖——附件（6）正是这种情况（无 .cpg，首字段 ASCII，其余字段 GBK）。
    try:
        hlen = struct.unpack("<H", raw[8:10])[0]
        nf = (hlen - 33) // 32
        for i in range(nf):
            raw[32 + i * 32:32 + i * 32 + 11].split(b"\x00")[0].decode("utf-8")
        return "utf-8"
    except (UnicodeDecodeError, struct.error):
        return "gbk"


def read_dbf(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    enc = _dbf_encoding(path)
    raw = path.read_bytes()
    nrec, hlen, rlen = struct.unpack("<IHH", raw[4:12])
    nf = (hlen - 33) // 32
    fields = []
    for i in range(nf):
        off = 32 + i * 32
        name = raw[off:off + 11].split(b"\x00")[0].decode(enc, "ignore")
        fields.append((name, chr(raw[off + 11]), raw[off + 16]))
    rows = []
    for r in range(nrec):
        base = hlen + r * rlen
        rec = raw[base:base + rlen]
        p, row = 1, {}
        for name, _typ, ln in fields:
            row[name] = rec[p:p + ln].decode(enc, "ignore").strip()
            p += ln
        rows.append(row)
    return pd.DataFrame(rows)


def read_shp_polygons(path: str | Path):
    """解析 polygon shapefile，返回每个要素的顶点数组与 bbox。"""
    raw = Path(path).read_bytes()
    i, out = 100, []
    while i < len(raw) - 8:
        _rec, = struct.unpack(">i", raw[i:i + 4])
        clen, = struct.unpack(">i", raw[i + 4:i + 8])
        stype, = struct.unpack("<i", raw[i + 8:i + 12])
        if stype in (5, 15):
            bbox = struct.unpack("<4d", raw[i + 12:i + 44])
            nparts, npoints = struct.unpack("<ii", raw[i + 44:i + 52])
            parts = struct.unpack("<%di" % nparts, raw[i + 52:i + 52 + 4 * nparts])
            p0 = i + 52 + 4 * nparts
            pts = np.frombuffer(raw[p0:p0 + 16 * npoints], dtype="<f8").reshape(npoints, 2).copy()
            out.append({"bbox": bbox, "parts": list(parts), "pts": pts})
        i += 8 + clen * 2
    return out


def load_hazard_points(use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE / "hazard_points.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    df = read_dbf(HAZARD_DIR / "西藏日喀则市吉隆县泥石流灾害区历史灾点数据.dbf")
    for c in ("x_coord", "y_coord", "height", "Area", "DisFrmFlt", "沟比降", "主沟长", "高差",
              "沟口H", "高程Max", "扇宽", "扇长"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"x_coord": "lon", "y_coord": "lat", "Area": "area_km2",
                            "沟比降": "grad", "主沟长": "length_m", "高差": "relief_m",
                            "沟口H": "mouth_m", "高程Max": "zmax_m", "扇宽": "fan_w",
                            "扇长": "fan_l"})
    df.to_parquet(cache)
    return df


def load_census() -> dict:
    df = read_dbf(CENSUS_DIR / "西藏日喀则市吉隆县第七次人口普查数据.dbf")
    row = df.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


# ---------------------------------------------------------------- 栅格

def load_dem() -> tuple[np.ndarray, dict]:
    cache = CACHE / "dem.npy"
    if cache.exists():
        g = json.loads((CACHE / "dem_grid.json").read_text(encoding="utf-8"))
        if g.get("preprocess_version") == PREPROCESS_VERSION:
            return np.load(cache), g
    arr, geo = read_geotiff(DEM_DIR / "DEM.tif")
    grid = geotiff_grid(geo)
    arr = arr.astype("float32")
    arr[arr <= -1000] = np.nan
    np.save(cache, arr)
    (CACHE / "dem_grid.json").write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    return arr, grid


def load_landcover(year: int = 2023) -> tuple[np.ndarray, dict]:
    cache = CACHE / f"lc{year}.npy"
    if cache.exists():
        g = json.loads((CACHE / f"lc{year}_grid.json").read_text(encoding="utf-8"))
        if g.get("preprocess_version") == PREPROCESS_VERSION:
            return np.load(cache), g
    arr, geo = read_geotiff(LANDCOVER_DIR / f"jilong_LUCD_{year}.tif")
    grid = geotiff_grid(geo)
    np.save(cache, arr)
    (CACHE / f"lc{year}_grid.json").write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    return arr, grid


LC_NAMES = {
    k: f"覆被原码{k}" for k in range(11)
}


# ---------------------------------------------------------------- 冰川

def load_glacier_nc() -> dict:
    """读取蓝塘里壤峰源区冰川地形/厚度/速度 NetCDF。

    HDF5 的 C 接口在超长中文路径下会打开失败，故首次读取时先复制到短路径缓存。
    """
    import h5py
    import shutil
    short = CACHE / "glacier_source.nc"
    if not short.exists():
        shutil.copyfile(GLACIER_NC, short)
    out = {}
    with h5py.File(str(short), "r") as f:
        for key in ("dem", "slope", "thickness", "v", "vx", "vy", "v_amp", "glacier_mask"):
            if key in f:
                out[key] = np.asarray(f[key][()], dtype="float64")
        out["x"] = np.asarray(f["x"][()], dtype="float64")
        out["y"] = np.asarray(f["y"][()], dtype="float64")
        out["crs"] = str(f["spatial_ref"].attrs.get("crs_wkt", ""))
        gt = f["spatial_ref"].attrs.get("GeoTransform", [])
        try:
            out["geotransform"] = [float(v) for v in gt]
        except (TypeError, ValueError):
            out["geotransform"] = [float(v) for v in str(gt).split()]
    return out


def load_glacier_outlines(version: int = 6) -> pd.DataFrame:
    cache = CACHE / f"glacier_rgi{version}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    dbf = GLACIER_DHDT_DIR / f"RGI_V{version}_JilongFlood_Buffer_200km.dbf"
    raw = dbf.read_bytes()
    nrec, hlen, rlen = struct.unpack("<IHH", raw[4:12])
    nf = (hlen - 33) // 32
    fields = []
    for i in range(nf):
        off = 32 + i * 32
        name = raw[off:off + 11].split(b"\x00")[0].decode("utf-8", "ignore")
        fields.append((name, raw[off + 16]))
    rows = []
    for r in range(nrec):
        base = hlen + r * rlen
        rec = raw[base:base + rlen]
        p, row = 1, {}
        for name, ln in fields:
            row[name] = rec[p:p + ln].decode("utf-8", "ignore").strip()
            p += ln
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("area_km2", "cenlon", "cenlat", "zmin_m", "zmax_m", "zmed_m", "slope_deg", "lmax_m"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(cache)
    return df


def glacier_dhdt_stats() -> list[dict]:
    """统计四个时段、各瓦片冰川表面高程变化率，用于源区/区域冰川减薄背景。"""
    cache = CACHE / "dhdt_stats.json"
    if cache.exists():
        old = json.loads(cache.read_text(encoding="utf-8"))
        if old and all(x.get("aggregation") == "valid_cell_weighted_v2" for x in old):
            return old
    recs = []
    for period_dir in sorted(GLACIER_DHDT_DIR.glob("13_14_15_rgi60_*")):
        period = period_dir.name.replace("13_14_15_rgi60_", "")
        vals, errs, cells, err_cells = [], [], 0, 0
        value_sum, error_sum = 0.0, 0.0
        for f in sorted((period_dir / "dhdt_UTM").glob("*.tif")):
            arr, _ = read_geotiff(f)
            arr = arr.astype("float64")
            arr[~np.isfinite(arr) | (np.abs(arr) > 1e3)] = np.nan
            if np.isfinite(arr).any():
                vals.append(float(np.nanmean(arr)))
                cells += int(np.isfinite(arr).sum())
                value_sum += float(np.nansum(arr))
        for f in sorted((period_dir / "dhdt_err_UTM").glob("*.tif")):
            arr, _ = read_geotiff(f)
            arr = arr.astype("float64")
            arr[~np.isfinite(arr) | (np.abs(arr) > 1e3)] = np.nan
            if np.isfinite(arr).any():
                errs.append(float(np.nanmean(arr)))
                err_cells += int(np.isfinite(arr).sum())
                error_sum += float(np.nansum(arr))
        recs.append({"period": period, "n_tiles": len(vals),
                     "mean_dhdt_m_per_yr": value_sum / cells if cells else None,
                     "mean_err_m_per_yr": error_sum / err_cells if err_cells else None,
                     "valid_cells": cells, "error_valid_cells": err_cells,
                     "aggregation": "valid_cell_weighted_v2",
                     "interpretation": "regional static background; mean error is not standard error of regional mean"})
    CACHE.joinpath("dhdt_stats.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    return recs


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
