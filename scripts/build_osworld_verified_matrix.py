from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import re
import struct
import urllib.error
import urllib.request
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREE = (
    ROOT
    / "analysis"
    / "public_backtest"
    / "osworld_verified"
    / "raw"
    / "hf_tree.json"
)
OUT_DIR = ROOT / "analysis" / "public_backtest" / "osworld_verified"
REPO = "xlangai/ubuntu_osworld_verified_trajs"


ZIP64_EOCD = b"PK\x06\x06"
EOCD = b"PK\x05\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


@dataclass(frozen=True)
class Archive:
    path: str
    size: int


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OSWorld-Verified task-level results with HTTP ranges."
    )
    parser.add_argument("--hf-tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument(
        "--include",
        default=r"(100step|100steps|OSWorldSurfer|evocua|kimi-k2[56]|mobile)",
        help="Regex for archive paths to consider before max-step filtering.",
    )
    parser.add_argument(
        "--min-complete",
        type=int,
        default=320,
        help="Minimum task-result files required before keeping a system.",
    )
    parser.add_argument(
        "--max-systems",
        type=int,
        default=None,
        help="Optional smoke-test cap after archive filtering.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--tail-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Only parse cached archive directories and fail on missing task bytes.",
    )
    parser.add_argument(
        "--fallback-result-txt",
        action="store_true",
        help="Fallback to per-task result.txt extraction when all_result.json is absent.",
    )
    return parser.parse_args()


def request_bytes(url: str, start: int, end: int, timeout: float) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    req.add_header("User-Agent", "aimlsys-osworld-matrix/1.0")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def read_tree(path: Path) -> list[Archive]:
    data = json.loads(path.read_text(encoding="utf-8"))
    archives: list[Archive] = []
    for row in data:
        name = str(row.get("path", ""))
        if name.endswith(".zip"):
            archives.append(Archive(path=name, size=int(row["size"])))
    return archives


def find_eocd(tail: bytes, archive_size: int) -> tuple[int, int, int]:
    idx = tail.rfind(EOCD)
    if idx < 0:
        raise ValueError("EOCD not found in archive tail")
    fields = struct.unpack("<IHHHHIIH", tail[idx : idx + 22])
    entry_count = fields[4]
    cd_size = fields[5]
    cd_offset = fields[6]
    if entry_count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        zip64_idx = tail.rfind(ZIP64_EOCD, 0, idx)
        if zip64_idx < 0:
            raise ValueError("Zip64 EOCD not found")
        zip64 = tail[zip64_idx : zip64_idx + 56]
        zip64_fields = struct.unpack("<IQHHIIQQQQ", zip64)
        entry_count = int(zip64_fields[7])
        cd_size = int(zip64_fields[8])
        cd_offset = int(zip64_fields[9])
    if cd_offset + cd_size > archive_size:
        raise ValueError("Central directory exceeds archive size")
    return int(entry_count), int(cd_size), int(cd_offset)


def parse_central_directory(cd: bytes, entry_count: int) -> list[ZipEntry]:
    entries: list[ZipEntry] = []
    pos = 0
    for _ in range(entry_count):
        if cd[pos : pos + 4] != CENTRAL:
            raise ValueError(f"Bad central directory signature at {pos}")
        fields = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos : pos + 46])
        method = int(fields[4])
        comp_size = int(fields[8])
        uncomp_size = int(fields[9])
        name_len = int(fields[10])
        extra_len = int(fields[11])
        comment_len = int(fields[12])
        local_offset = int(fields[16])
        name_start = pos + 46
        name = cd[name_start : name_start + name_len].decode("utf-8", "replace")
        extra = cd[
            name_start + name_len : name_start + name_len + extra_len
        ]
        if (
            comp_size == 0xFFFFFFFF
            or uncomp_size == 0xFFFFFFFF
            or local_offset == 0xFFFFFFFF
        ):
            uncomp_size, comp_size, local_offset = apply_zip64_extra(
                extra, uncomp_size, comp_size, local_offset
            )
        entries.append(
            ZipEntry(
                name=name,
                method=method,
                compressed_size=comp_size,
                uncompressed_size=uncomp_size,
                local_offset=local_offset,
            )
        )
        pos = name_start + name_len + extra_len + comment_len
    return entries


def apply_zip64_extra(
    extra: bytes,
    uncompressed_size: int,
    compressed_size: int,
    local_offset: int,
) -> tuple[int, int, int]:
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[pos : pos + 4])
        pos += 4
        data = extra[pos : pos + data_size]
        pos += data_size
        if header_id != 0x0001:
            continue
        cursor = 0
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = struct.unpack("<Q", data[cursor : cursor + 8])[0]
            cursor += 8
        if compressed_size == 0xFFFFFFFF:
            compressed_size = struct.unpack("<Q", data[cursor : cursor + 8])[0]
            cursor += 8
        if local_offset == 0xFFFFFFFF:
            local_offset = struct.unpack("<Q", data[cursor : cursor + 8])[0]
        return int(uncompressed_size), int(compressed_size), int(local_offset)
    raise ValueError("Zip64 extra field missing required size or offset")


def load_archive_directory(
    archive: Archive,
    repo: str,
    out_dir: Path,
    timeout: float,
    tail_bytes: int,
    no_network: bool,
) -> list[ZipEntry]:
    cache = out_dir / "zip_directories" / f"{safe_stem(archive.path)}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [ZipEntry(**row) for row in data]
    if no_network:
        raise FileNotFoundError(cache)
    tail_start = max(0, archive.size - tail_bytes)
    tail = request_bytes(resolve_url(repo, archive.path), tail_start, archive.size - 1, timeout)
    entry_count, cd_size, cd_offset = find_eocd(tail, archive.size)
    if cd_offset < tail_start:
        cd = request_bytes(
            resolve_url(repo, archive.path),
            cd_offset,
            cd_offset + cd_size - 1,
            timeout,
        )
    else:
        local_start = cd_offset - tail_start
        cd = tail[local_start : local_start + cd_size]
    entries = parse_central_directory(cd, entry_count)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps([entry.__dict__ for entry in entries], indent=2),
        encoding="utf-8",
    )
    return entries


def safe_stem(name: str) -> str:
    stem = name.removesuffix(".zip")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def extract_entry(
    archive: Archive,
    entry: ZipEntry,
    repo: str,
    timeout: float,
) -> bytes:
    # Result files are tiny. One 8 KiB read covers the local header, filename,
    # extra fields, compressed bytes, and any small descriptor for these entries.
    end = min(archive.size - 1, entry.local_offset + entry.compressed_size + 8192)
    block = request_bytes(resolve_url(repo, archive.path), entry.local_offset, end, timeout)
    if block[:4] != LOCAL:
        raise ValueError(f"Bad local signature for {entry.name}")
    fields = struct.unpack("<IHHHHHIIIHH", block[:30])
    method = int(fields[3])
    name_len = int(fields[9])
    extra_len = int(fields[10])
    data_start = 30 + name_len + extra_len
    payload = block[data_start : data_start + entry.compressed_size]
    if method == 0:
        return payload
    if method == 8:
        return zlib.decompress(payload, -15)
    raise ValueError(f"Unsupported compression method {method} for {entry.name}")


def parse_score(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value


def parse_result_map(text: str) -> dict[str, dict[str, float]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)
    if not isinstance(data, dict):
        raise ValueError("Result map is not a dictionary")
    out: dict[str, dict[str, float]] = {}
    for group, tasks in data.items():
        if not isinstance(tasks, dict):
            continue
        group_scores: dict[str, float] = {}
        for task_id, raw_score in tasks.items():
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if math.isnan(score):
                continue
            group_scores[str(task_id)] = score
        if group_scores:
            out[str(group)] = group_scores
    return out


def run_id_from_all_result_path(path: str) -> str:
    parts = path.split("/")
    if "local_results" in parts:
        idx = parts.index("local_results")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "default_run"


def rows_from_all_result(
    archive: Archive,
    entry: ZipEntry,
    repo: str,
    timeout: float,
    system_id: str,
    system_label: str,
    archive_args: dict[str, object],
) -> list[dict[str, object]]:
    raw = extract_entry(archive, entry, repo, timeout)
    result_map = parse_result_map(raw.decode("utf-8", "replace"))
    run_id = run_id_from_all_result_path(entry.name)
    rows: list[dict[str, object]] = []
    for group, tasks in result_map.items():
        for task_id, score in tasks.items():
            rows.append(
                {
                    "source": "OSWorld-Verified",
                    "system_id": system_id,
                    "system_label": system_label,
                    "archive_path": archive.path,
                    "run_id": run_id,
                    "task_key": f"{group}/{task_id}",
                    "task_id": task_id,
                    "coverage_group": group,
                    "outcome": score,
                    "split": "verified",
                    "benchmark_version": "OSWorld-Verified",
                    "result_path": entry.name,
                    "max_steps": archive_args.get("max_steps"),
                    "model": archive_args.get("model"),
                    "model_type": archive_args.get("model_type"),
                    "observation_type": archive_args.get("observation_type"),
                    "action_space": archive_args.get("action_space"),
                }
            )
    return rows


def task_from_result_path(path: str) -> tuple[str, str] | None:
    parts = path.split("/")
    if len(parts) < 4 or parts[-1] != "result.txt":
        return None
    return parts[-3], parts[-2]


def infer_system_id(archive: Archive, args: dict[str, object]) -> tuple[str, str]:
    model = str(args.get("model", "")).strip()
    max_steps = args.get("max_steps")
    stem = archive.path.removesuffix(".zip")
    if model:
        label = f"{model} ({max_steps} steps)" if max_steps else model
    else:
        label = stem
    return safe_stem(stem), label


def args_from_archive_path(archive: Archive) -> dict[str, object]:
    stem = archive.path.removesuffix(".zip")
    max_steps = None
    match = re.search(r"([0-9]+)\s*steps?", stem, re.IGNORECASE)
    if not match:
        match = re.search(r"([0-9]+)step", stem, re.IGNORECASE)
    if match:
        max_steps = int(match.group(1))
    return {"model": stem, "max_steps": max_steps}


def extract_archive_rows(
    archive: Archive,
    repo: str,
    out_dir: Path,
    timeout: float,
    tail_bytes: int,
    no_network: bool,
    fallback_result_txt: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    entries = load_archive_directory(
        archive, repo, out_dir, timeout, tail_bytes, no_network
    )
    by_name = {entry.name: entry for entry in entries}
    args_entries = [entry for entry in entries if entry.name.endswith("/args.json")]
    archive_args: dict[str, object] = args_from_archive_path(archive)
    if args_entries:
        raw = extract_entry(archive, args_entries[0], repo, timeout)
        archive_args.update(json.loads(raw.decode("utf-8")))
    system_id, system_label = infer_system_id(archive, archive_args)
    all_result_entries = [
        entry for entry in entries if entry.name.endswith("/all_result.json")
    ]
    result_entries = [entry for entry in entries if entry.name.endswith("/result.txt")]
    rows: list[dict[str, object]] = []
    for entry in all_result_entries:
        rows.extend(
            rows_from_all_result(
                archive, entry, repo, timeout, system_id, system_label, archive_args
            )
        )
    if not rows and fallback_result_txt:
        for entry in result_entries:
            task = task_from_result_path(entry.name)
            if task is None:
                continue
            group, task_id = task
            raw = extract_entry(archive, entry, repo, timeout)
            score = parse_score(raw.decode("utf-8", "replace"))
            if score is None:
                continue
            rows.append(
                {
                    "source": "OSWorld-Verified",
                    "system_id": system_id,
                    "system_label": system_label,
                    "archive_path": archive.path,
                    "run_id": run_id_from_all_result_path(entry.name),
                    "task_key": f"{group}/{task_id}",
                    "task_id": task_id,
                    "coverage_group": group,
                    "outcome": score,
                    "split": "verified",
                    "benchmark_version": "OSWorld-Verified",
                    "result_path": entry.name,
                    "max_steps": archive_args.get("max_steps"),
                    "model": archive_args.get("model"),
                    "model_type": archive_args.get("model_type"),
                    "observation_type": archive_args.get("observation_type"),
                    "action_space": archive_args.get("action_space"),
                }
            )
    metadata = {
        "archive_path": archive.path,
        "system_id": system_id,
        "system_label": system_label,
        "entry_count": len(entries),
        "all_result_files": len(all_result_entries),
        "result_files": len(result_entries),
        "parsed_results": len(rows),
        "max_steps": archive_args.get("max_steps"),
        "model": archive_args.get("model"),
        "model_type": archive_args.get("model_type"),
    }
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["system_id"]), str(row["task_key"]))].append(row)
    out: list[dict[str, object]] = []
    for (_, _), group in grouped.items():
        first = group[0]
        scores = [float(row["outcome"]) for row in group]
        out.append(
            {
                "source": first["source"],
                "system_id": first["system_id"],
                "system_label": first["system_label"],
                "archive_path": first["archive_path"],
                "task_key": first["task_key"],
                "task_id": first["task_id"],
                "coverage_group": first["coverage_group"],
                "outcome": sum(scores) / len(scores),
                "success_count": sum(scores),
                "trials_observed": len(scores),
                "run_ids": ";".join(sorted(str(row["run_id"]) for row in group)),
                "split": first["split"],
                "benchmark_version": first["benchmark_version"],
                "max_steps": first["max_steps"],
                "model": first["model"],
                "model_type": first["model_type"],
                "observation_type": first["observation_type"],
                "action_space": first["action_space"],
            }
        )
    return sorted(out, key=lambda row: (str(row["system_id"]), str(row["task_key"])))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    include = re.compile(args.include, re.IGNORECASE)
    archives = [archive for archive in read_tree(args.hf_tree) if include.search(archive.path)]
    if args.max_systems is not None:
        archives = archives[: args.max_systems]

    all_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    for idx, archive in enumerate(archives, start=1):
        print(f"[{idx}/{len(archives)}] extracting {archive.path}", flush=True)
        try:
            rows, metadata = extract_archive_rows(
                archive,
                args.repo,
                args.out_dir,
                args.timeout,
                args.tail_bytes,
                args.no_network,
                args.fallback_result_txt,
            )
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            metadata = {
                "archive_path": archive.path,
                "system_id": safe_stem(archive.path.removesuffix(".zip")),
                "system_label": archive.path.removesuffix(".zip"),
                "entry_count": 0,
                "result_files": 0,
                "parsed_results": 0,
                "max_steps": None,
                "model": None,
                "model_type": None,
                "error": repr(exc),
            }
            rows = []
        metadata["kept"] = len(rows) >= args.min_complete
        metadata_rows.append(metadata)
        if len(rows) >= args.min_complete:
            all_rows.extend(rows)

    aggregate_rows = aggregate_runs(all_rows)
    write_csv(args.out_dir / "osworld_verified_run_rows.csv", all_rows)
    write_csv(args.out_dir / "osworld_verified_task_rows.csv", aggregate_rows)
    write_csv(args.out_dir / "osworld_verified_archive_audit.csv", metadata_rows)
    print(
        f"wrote {len(aggregate_rows)} task rows from {len(all_rows)} run rows and "
        f"{sum(1 for row in metadata_rows if row.get('kept'))} systems",
        flush=True,
    )


if __name__ == "__main__":
    main()
