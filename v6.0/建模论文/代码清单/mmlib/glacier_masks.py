"""Rasterize supplied RGI6 boundaries on DEM pixel centres, retaining holes."""
import hashlib
import numpy as np
from matplotlib.path import Path as PolygonPath
from . import io_data as io

def rgi6_mask(grid, shape):
    source = io.GLACIER_DHDT_DIR / 'RGI_V6_JilongFlood_Buffer_200km.shp'
    if not source.exists():
        raise FileNotFoundError(f'RGI6 supplied boundary required; run prepare_inputs.py: {source}')
    prj = source.with_suffix('.prj')
    wkt = prj.read_text(encoding='utf-8-sig')
    if 'GEOGCS' not in wkt.upper() or 'PROJCS' in wkt.upper():
        raise ValueError('RGI6 must use a geographic CRS; no silent projected-coordinate interpretation')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    mask_path, meta_path = io.CACHE/'rgi6_dem_mask.npy', io.CACHE/'rgi6_dem_mask.json'
    import json
    signature = {'boundary_sha256': digest, 'shape': list(shape), 'grid': grid, 'version': 'rgi6-centre-xor-v1'}
    if mask_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding='utf-8'))
        if old.get('signature') == signature:
            return np.load(mask_path), old
    mask = np.zeros(shape, dtype=bool)
    nx, ny = shape[1], shape[0]
    xmin, xmax = sorted([grid['lon0'], grid['lon0']+(nx-1)*grid['dlon']])
    ymin, ymax = sorted([grid['lat0'], grid['lat0']+(ny-1)*grid['dlat']])
    used = 0
    for feature in io.read_shp_polygons(source):
        bx0, by0, bx1, by1 = feature['bbox']
        if bx1 < xmin or bx0 > xmax or by1 < ymin or by0 > ymax: continue
        rf = [(by0-grid['lat0'])/grid['dlat'], (by1-grid['lat0'])/grid['dlat']]
        cf = [(bx0-grid['lon0'])/grid['dlon'], (bx1-grid['lon0'])/grid['dlon']]
        r0,r1=max(0,int(np.floor(min(rf)))-1),min(ny,int(np.ceil(max(rf)))+2)
        c0,c1=max(0,int(np.floor(min(cf)))-1),min(nx,int(np.ceil(max(cf)))+2)
        rr,cc=np.mgrid[r0:r1,c0:c1]
        points=np.column_stack([(grid['lon0']+cc*grid['dlon']).ravel(),(grid['lat0']+rr*grid['dlat']).ravel()])
        inside=np.zeros(len(points),dtype=bool)
        ends=feature['parts']+[len(feature['pts'])]
        for start,end in zip(ends[:-1],ends[1:]):
            ring=feature['pts'][start:end]
            if len(ring)>=3: inside ^= PolygonPath(ring).contains_points(points)
        mask[r0:r1,c0:c1] |= inside.reshape(rr.shape)
        used+=1
    if not mask.any(): raise ValueError('No supplied RGI6 polygons intersect the DEM; do not substitute elevation')
    meta={'signature':signature,'source':str(source.relative_to(io.ROOT)), 'crs_wkt':wkt,
          'features_intersecting_dem':used, 'pixels':int(mask.sum()),
          'scope':'RGI6 inventory epoch, not observed 2026 glacier/snow extent; centre rasterization with ring parity'}
    np.save(mask_path,mask); io.write_json(meta_path,meta)
    return mask,meta
