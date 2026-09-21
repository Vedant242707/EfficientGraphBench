"""Visible downloads with timeouts, validation, and HTTP range resume."""

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from torch_geometric.datasets import Flickr


def validate_download(path):
    return validate_download_with_suffix(path, Path(path).suffix)


def download_file(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if validate_download(destination):
            return
        if not partial.exists():
            destination.replace(partial)
        else:
            raise ValueError(
                f"Both invalid {destination} and {partial} exist; keep one partial file"
            )
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            append = response.status_code == 206 and offset > 0
            if response.status_code == 206:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", "")
                )
                if not match or int(match[1]) != offset:
                    raise ValueError("download server returned an inconsistent resume range")
            if not append:
                offset = 0
            length = response.headers.get("Content-Length")
            total = offset + int(length) if length else None
            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(destination.name, total=total, completed=offset)
                with partial.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            handle.write(chunk)
                            progress.update(task, advance=len(chunk))
            if total is not None and partial.stat().st_size != total:
                raise ValueError("download ended before the expected number of bytes arrived")
        # Validate using the intended extension, without accepting a partial as final.
        if not validate_download_with_suffix(partial, destination.suffix):
            raise ValueError("downloaded content is incomplete or is not the expected dataset file")
        partial.replace(destination)
    except requests.RequestException as exc:
        raise OSError(
            f"Download interrupted for {destination.name}. Partial data is kept; "
            "rerun the same command to retry. " + str(exc)
        ) from exc


def validate_download_with_suffix(path, suffix):
    if suffix == ".npy":
        try:
            with Path(path).open("rb") as handle:
                version = np.lib.format.read_magic(handle)
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _, dtype = reader(handle)
                return (
                    not dtype.hasobject
                    and Path(path).stat().st_size
                    == handle.tell() + int(np.prod(shape)) * dtype.itemsize
                )
        except (ValueError, OSError, EOFError):
            return False
    try:
        if suffix == ".npz":
            with zipfile.ZipFile(path) as archive:
                return archive.testzip() is None
        if suffix == ".json":
            with Path(path).open(encoding="utf-8") as handle:
                return isinstance(json.load(handle), dict)
    except (ValueError, OSError, zipfile.BadZipFile, UnicodeError):
        return False
    return False


class ReliableFlickr(Flickr):
    def _download(self):
        # PyG otherwise considers every existing file complete, even after interruption.
        self.download()

    def download(self):
        files = {
            "adj_full.npz": self.adj_full_id,
            "feats.npy": self.feats_id,
            "class_map.json": self.class_map_id,
            "role.json": self.role_id,
        }
        for filename, identifier in files.items():
            download_file(
                f"https://drive.usercontent.google.com/download?id={identifier}&confirm=t",
                Path(self.raw_dir) / filename,
            )
