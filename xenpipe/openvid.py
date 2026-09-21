from __future__ import annotations

import binascii
import hashlib
import json
import os
import struct
import time
import zlib
from pathlib import Path
from urllib.parse import quote

import requests

HF_BASE = "https://huggingface.co/datasets"
OPENVID_DEFAULT = "nkp37/OpenVid-1M"
OPENVID_JSON = "OpenVidHD/OpenVidHD.json"
_RANGE_TIMEOUT = 120
_SESSION = requests.Session()


def _cache_root(cache_dir: str | None) -> Path:
    root = Path(cache_dir or ".xen-cache") / "openvid"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _zip_url(dataset_id: str, revision: str, part: int) -> str:
    return (
        f"{HF_BASE}/{quote(dataset_id, safe='/')}/resolve/{revision}/"
        f"OpenVidHD/OpenVidHD_part_{part}.zip"
    )


def _json_url(dataset_id: str, revision: str) -> str:
    return (
        f"{HF_BASE}/{quote(dataset_id, safe='/')}/resolve/{revision}/"
        f"{OPENVID_JSON}"
    )


def _range_get(url: str, start: int, end: int, token: str | None = None) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last = None
    for attempt in range(4):
        try:
            r = _SESSION.get(url, headers=headers, timeout=_RANGE_TIMEOUT)
            r.raise_for_status()
            if r.status_code != 206:
                raise RuntimeError(f"Expected HTTP 206 for Range request, got {r.status_code}")
            cr = r.headers.get("Content-Range", "")
            if not cr.startswith("bytes "):
                raise RuntimeError("Server did not return a valid Content-Range header")
            if len(r.content) != end - start + 1:
                raise RuntimeError(
                    f"Range length mismatch: {len(r.content)} != {end-start+1}"
                )
            return r.content
        except Exception as exc:
            last = exc
            if attempt < 3:
                time.sleep(0.75 * (2 ** attempt))
    raise RuntimeError(f"Range request failed after retries: {last}") from last


def _download_json(dataset_id: str, revision: str, root: Path, token: str | None) -> list[dict]:
    path = root / "OpenVidHD.json"
    if not path.exists():
        r = _SESSION.get(_json_url(dataset_id, revision), headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=_RANGE_TIMEOUT)
        r.raise_for_status()
        path.write_bytes(r.content)
    return json.loads(path.read_text("utf-8"))


def _build_part_map(dataset_id: str, revision: str, root: Path, token: str | None) -> dict[str, int]:
    map_path = root / "filename_to_part.json"
    if map_path.exists():
        return {k: int(v) for k, v in json.loads(map_path.read_text("utf-8")).items()}

    mapping: dict[str, int] = {}
    for item in _download_json(dataset_id, revision, root, token):
        for key, files in item.items():
            part = int(key.removeprefix("part"))
            for name in files:
                mapping[str(name)] = part
    map_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return mapping


def _parse_zip64_eocd(tail: bytes) -> tuple[int, int]:
    p = tail.rfind(b"PK\x06\x06")
    if p < 0 or p + 56 > len(tail):
        raise RuntimeError("ZIP64 EOCD not found in archive tail")
    record_size = struct.unpack_from("<Q", tail, p + 4)[0]
    if record_size < 44:
        raise RuntimeError(f"Invalid ZIP64 EOCD record size: {record_size}")
    # ZIP64 EOCD: total entries +32, central-directory size +40, offset +48
    entries = struct.unpack_from("<Q", tail, p + 32)[0]
    cd_size = struct.unpack_from("<Q", tail, p + 40)[0]
    cd_offset = struct.unpack_from("<Q", tail, p + 48)[0]
    if not (0 < entries < 100_000_000 and 0 < cd_size < 20_000_000_000 and 0 < cd_offset < 100_000_000_000):
        raise RuntimeError(f"Invalid ZIP64 metadata: entries={entries}, cd_size={cd_size}, cd_offset={cd_offset}")
    return cd_offset, cd_size


def _suffix_get(url: str, size: int, token: str | None = None) -> bytes:
    headers = {"Range": f"bytes=-{size}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = _SESSION.get(url, headers=headers, timeout=_RANGE_TIMEOUT)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"Expected HTTP 206 for suffix Range request, got {r.status_code}")
    if len(r.content) != size:
        raise RuntimeError(f"Suffix range length mismatch: {len(r.content)} != {size}")
    return r.content


def _central_directory(dataset_id: str, revision: str, part: int, root: Path, token: str | None) -> bytes:
    cache = root / f"part_{part}_central.bin"
    if cache.exists():
        return cache.read_bytes()

    url = _zip_url(dataset_id, revision, part)
    tail = _suffix_get(url, 16 * 1024 * 1024, token)
    cd_offset, cd_size = _parse_zip64_eocd(tail)
    cd = _range_get(url, cd_offset, cd_offset + cd_size - 1, token)
    cache.write_bytes(cd)
    return cd


def _find_entry(cd: bytes, target: str) -> tuple[int, int, int, int, int]:
    pos = 0
    wanted = target.encode("utf-8")
    while pos + 46 <= len(cd):
        if cd[pos:pos+4] != b"PK\x01\x02":
            pos += 1
            continue
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", cd, pos + 28)
        name = cd[pos+46:pos+46+name_len]
        if name == wanted:
            method = struct.unpack_from("<H", cd, pos + 10)[0]
            compressed = struct.unpack_from("<I", cd, pos + 20)[0]
            uncompressed = struct.unpack_from("<I", cd, pos + 24)[0]
            crc = struct.unpack_from("<I", cd, pos + 16)[0]
            local_offset = struct.unpack_from("<I", cd, pos + 42)[0]
            extra = cd[pos+46+name_len:pos+46+name_len+extra_len]
            if compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
                ep = 0
                while ep + 4 <= len(extra):
                    hid, size = struct.unpack_from("<HH", extra, ep)
                    field = extra[ep+4:ep+4+size]
                    if hid == 0x0001:
                        fp = 0
                        if uncompressed == 0xFFFFFFFF:
                            uncompressed = struct.unpack_from("<Q", field, fp)[0]; fp += 8
                        if compressed == 0xFFFFFFFF:
                            compressed = struct.unpack_from("<Q", field, fp)[0]; fp += 8
                        if local_offset == 0xFFFFFFFF:
                            local_offset = struct.unpack_from("<Q", field, fp)[0]
                        break
                    ep += 4 + size
            return method, compressed, uncompressed, crc, local_offset
        pos += 46 + name_len + extra_len + comment_len
    raise FileNotFoundError(f"{target!r} not found in OpenVidHD part")


def extract_openvid_video(value: str, dataset_id: str, revision: str, cache_dir: str | None = None, token: str | None = None) -> str:
    root = _cache_root(cache_dir)
    mapping = _build_part_map(dataset_id, revision, root, token)
    part = mapping.get(value)
    if part is None:
        raise FileNotFoundError(f"OpenVid video reference not found in OpenVidHD.json: {value}")

    out_dir = root / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (hashlib.sha256(f"{dataset_id}@{revision}:{value}".encode()).hexdigest() + ".mp4")
    if out.exists():
        return str(out)

    cd = _central_directory(dataset_id, revision, part, root, token)
    method, compressed_size, uncompressed_size, expected_crc, local_offset = _find_entry(cd, value)

    if compressed_size <= 0 or uncompressed_size <= 0:
        raise RuntimeError(f"Invalid ZIP entry sizes for {value}")

    zip_url = _zip_url(dataset_id, revision, part)
    local = _range_get(zip_url, local_offset, local_offset + 4095, token)
    if local[:4] != b"PK\x03\x04":
        raise RuntimeError("Invalid ZIP local file header")
    name_len = struct.unpack_from("<H", local, 26)[0]
    extra_len = struct.unpack_from("<H", local, 28)[0]
    data_offset = local_offset + 30 + name_len + extra_len

    compressed = _range_get(zip_url, data_offset, data_offset + compressed_size - 1, token)
    if method == 0:
        video = compressed
    elif method == 8:
        video = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"Unsupported ZIP compression method {method} for {value}")

    if len(video) != uncompressed_size:
        raise RuntimeError(f"Size validation failed for {value}: {len(video)} != {uncompressed_size}")
    crc = binascii.crc32(video) & 0xFFFFFFFF
    if crc != expected_crc:
        raise RuntimeError(f"CRC validation failed for {value}: {crc:#x} != {expected_crc:#x}")

    tmp = out.with_suffix(".part")
    tmp.write_bytes(video)
    os.replace(tmp, out)
    return str(out)
