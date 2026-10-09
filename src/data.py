from __future__ import annotations

import os
import hashlib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date as calendar_date, timedelta
from pathlib import Path

import requests


GMW_2020_ZIP_URL = "https://zenodo.org/records/6894273/files/gmw_v3_2020_gtiff.zip?download=1"
S2_SRF_V4_URL = "https://sentiwiki.copernicus.eu/__attachments/a_ece6183b6698587e7ecd9804974bece3d82e39e151aa5aeca9d8fd95e1f1558c/COPE-GSEG-EOPG-TN-15-0007%20-%20Sentinel-2%20Spectral%20Response%20Functions%202024%20-%204.0.xlsx"
S2_SRF_V4_SHA256 = "1a9edc27d692a570911a460d589f188da0fc3e27f0b0bd1ad322059c380519b0"
SANDERMAN_SOC_ZIP_URL = "https://zenodo.org/api/records/7727569/files/mangroves_SOC30m_0_100cm.zip/content"
SANDERMAN_SOC_ZIP_BYTES = 378284483
SANDERMAN_SOC_ZIP_MD5 = "bfb5236878b0b60dee11ca9157e39c38"


def ensure_s2_srf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with requests.get(S2_SRF_V4_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with path.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != S2_SRF_V4_SHA256:
        raise ValueError("Official Sentinel-2 SRF v4 checksum mismatch")
    return path


def ensure_gmw_tiles(destination: Path, cache: Path, tile_names: tuple[str, ...]) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    wanted = [destination / name for name in tile_names]
    if all(path.exists() for path in wanted):
        return wanted
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        partial = cache.with_suffix(cache.suffix + ".part")
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        with requests.get(GMW_2020_ZIP_URL, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            mode = "ab" if response.status_code == 206 and start else "wb"
            with partial.open(mode) as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        partial.replace(cache)
    with zipfile.ZipFile(cache) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        for path in wanted:
            if path.exists():
                continue
            if path.name not in members:
                raise FileNotFoundError(f"GMW tile absent from official archive: {path.name}")
            with archive.open(members[path.name]) as source, path.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    return wanted


def ensure_sanderman_soil_carbon_zip(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".part")
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        with requests.get(SANDERMAN_SOC_ZIP_URL, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            mode = "ab" if response.status_code == 206 and start else "wb"
            with partial.open(mode) as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        partial.replace(destination)
    if destination.stat().st_size != SANDERMAN_SOC_ZIP_BYTES:
        raise ValueError("Incomplete Sanderman soil carbon archive")
    digest = hashlib.md5()
    with destination.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != SANDERMAN_SOC_ZIP_MD5:
        raise ValueError("Sanderman soil carbon archive checksum mismatch")
    return destination


def find_emit_granule(scene_id: str, bbox: tuple[float, float, float, float], date: str):
    import earthaccess

    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError("Earthdata authentication failed")
    granules = earthaccess.search_data(
        short_name="EMITL2ARFL",
        version="001",
        bounding_box=bbox,
        temporal=(date, (calendar_date.fromisoformat(date) + timedelta(days=1)).isoformat()),
    )
    return next(granule for granule in granules if granule["meta"]["native-id"] == scene_id)


def download_emit(scene_id: str, bbox: tuple[float, float, float, float], date: str, root: Path) -> list[Path]:
    import earthaccess

    destination = root / scene_id
    destination.mkdir(parents=True, exist_ok=True)
    wanted = (f"{scene_id}.nc", f"EMIT_L2A_MASK_001_{scene_id.removeprefix('EMIT_L2A_RFL_001_')}.nc")
    if all((destination / name).exists() for name in wanted):
        return [destination / name for name in wanted]
    granule = find_emit_granule(scene_id, bbox, date)
    urls = {url.rsplit("/", 1)[-1]: url for url in granule.data_links()}
    outputs = []
    for name in wanted:
        if name not in urls:
            raise RuntimeError(f"CMR lacks required asset {name}")
        final = destination / name
        partial = destination / f"{name}.part"
        if final.exists():
            outputs.append(final)
            continue
        session = earthaccess.get_requests_https_session()
        with session.get(urls[name], headers={"Range": "bytes=0-0"}, timeout=60) as response:
            response.raise_for_status()
            if response.status_code != 206 or "Content-Range" not in response.headers:
                raise RuntimeError(f"Provider does not support ranges for {name}")
            total = int(response.headers["Content-Range"].rsplit("/", 1)[-1])
        chunk_size = 32 * 1024 * 1024
        chunk_count = (total + chunk_size - 1) // chunk_size
        markers = destination / f"{name}.chunks"
        markers.mkdir(exist_ok=True)
        with partial.open("ab") as target:
            target.truncate(total)

        def fetch_chunk(index: int) -> None:
            marker = markers / f"{index:06d}.done"
            if marker.exists():
                return
            start = index * chunk_size
            end = min(total - 1, start + chunk_size - 1)
            worker_session = earthaccess.get_requests_https_session()
            with worker_session.get(urls[name], headers={"Range": f"bytes={start}-{end}"}, timeout=120) as response:
                response.raise_for_status()
                expected_range = f"bytes {start}-{end}/{total}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"Unexpected byte range for {name}, chunk {index}")
                content = response.content
            if len(content) != end - start + 1:
                raise RuntimeError(f"Short response for {name}, chunk {index}")
            with partial.open("r+b") as target:
                target.seek(start)
                target.write(content)
            marker.touch()

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(fetch_chunk, range(chunk_count)))
        if partial.stat().st_size != total or len(list(markers.glob("*.done"))) != chunk_count:
            raise RuntimeError(f"Incomplete download for {name}")
        os.replace(partial, final)
        for marker in markers.glob("*.done"):
            marker.unlink()
        markers.rmdir()
        outputs.append(final)
    return outputs
