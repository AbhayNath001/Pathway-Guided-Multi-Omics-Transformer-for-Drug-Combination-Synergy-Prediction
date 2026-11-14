"""Download/setup the public data inputs used by the PATH-SYN DrugComb workflow.

Downloads DepMap Public 24Q2, DrugComb v1.5 summary data and MSigDB GMT files.
No machine-specific local repository path is required. Some providers may require
manual access; in that case, place the downloaded file in the printed directory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

DEPMAP_ARTICLE = "25880521"
DEPMAP_FILES = {
    "Model.csv": "https://ndownloader.figshare.com/files/46489732",
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv": "https://ndownloader.figshare.com/files/46490878",
    "OmicsCNGene.csv": "https://ndownloader.figshare.com/files/46489771",
    "OmicsSomaticMutations.csv": "https://ndownloader.figshare.com/files/46500367",
}
DRUGCOMB_ZENODO_RECORD = "15235991"


def download_file(url: str, destination: Path, retries: int = 5) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1000:
        print(f"[skip] {destination}")
        return True

    temp = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            existing = temp.stat().st_size if temp.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            response = requests.get(url, stream=True, timeout=90, headers=headers)
            response.raise_for_status()
            if existing and response.status_code == 200:
                existing = 0
                mode = "wb"
            else:
                mode = "ab" if existing else "wb"
            with temp.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            if temp.stat().st_size <= 1000:
                raise IOError("downloaded file is unexpectedly small")
            temp.replace(destination)
            print(f"[downloaded] {destination} ({destination.stat().st_size:,} bytes)")
            return True
        except Exception as exc:
            print(f"[warning] attempt {attempt}/{retries} failed for {destination.name}: {exc}")
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))
    return False


def resolve_depmap_urls() -> dict[str, str]:
    urls = dict(DEPMAP_FILES)
    try:
        response = requests.get(
            f"https://api.figshare.com/v2/articles/{DEPMAP_ARTICLE}", timeout=30
        )
        response.raise_for_status()
        for item in response.json().get("files", []):
            if item.get("name") in urls and item.get("download_url"):
                urls[item["name"]] = item["download_url"]
    except Exception as exc:
        print(f"[warning] Figshare metadata lookup failed; using published 24Q2 file URLs: {exc}")
    return urls


def download_drugcomb(folder: Path) -> bool:
    folder.mkdir(parents=True, exist_ok=True)
    present = sorted(folder.glob("*.csv"))
    if present:
        print(f"[skip] DrugComb CSV already present: {present[0]}")
        return True

    api = f"https://zenodo.org/api/records/{DRUGCOMB_ZENODO_RECORD}"
    try:
        response = requests.get(api, timeout=60)
        response.raise_for_status()
        files = response.json().get("files", [])
        candidates = [
            item for item in files
            if item.get("key", item.get("filename", "")).lower().endswith(".csv")
            and "summary" in item.get("key", item.get("filename", "")).lower()
        ]
        if not candidates:
            candidates = [
                item for item in files
                if item.get("key", item.get("filename", "")).lower().endswith(".csv")
            ]
        if not candidates:
            raise FileNotFoundError("No CSV file was listed in the DrugComb Zenodo record")
        selected = candidates[0]
        key = selected.get("key", selected.get("filename", "drugcomb_summary.csv"))
        url = selected.get("links", {}).get("self")
        if not url:
            url = f"https://zenodo.org/records/{DRUGCOMB_ZENODO_RECORD}/files/{key}?download=1"
        return download_file(url, folder / key)
    except Exception as exc:
        print(f"[warning] automatic DrugComb download failed: {exc}")
        print(f"Download the DrugComb v1.5 summary CSV from https://zenodo.org/records/{DRUGCOMB_ZENODO_RECORD}")
        print(f"Place the CSV in: {folder.resolve()}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/real")
    parser.add_argument("--msigdb-version", default="2024.1.Hs")
    parser.add_argument("--skip-depmap", action="store_true")
    parser.add_argument("--skip-drugcomb", action="store_true")
    parser.add_argument("--skip-msigdb", action="store_true")
    args = parser.parse_args()

    root = Path(args.out)
    depmap_dir = root / "depmap"
    drugcomb_dir = root / "drugcomb"
    msigdb_dir = root / "msigdb"
    for directory in (depmap_dir, drugcomb_dir, msigdb_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.skip_depmap:
        print("\n[DepMap] Public 24Q2")
        for filename, url in resolve_depmap_urls().items():
            download_file(url, depmap_dir / filename)

    if not args.skip_msigdb:
        print(f"\n[MSigDB] {args.msigdb_version}")
        base = (
            "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
            f"{args.msigdb_version}/"
        )
        gmt_files = [
            f"h.all.v{args.msigdb_version}.symbols.gmt",
            f"c2.cp.kegg_legacy.v{args.msigdb_version}.symbols.gmt",
            f"c2.cp.reactome.v{args.msigdb_version}.symbols.gmt",
        ]
        for filename in gmt_files:
            download_file(base + filename, msigdb_dir / filename)

    if not args.skip_drugcomb:
        print("\n[DrugComb] v1.5")
        download_drugcomb(drugcomb_dir)

    print("\nData directories:")
    print(f"  DepMap:  {depmap_dir.resolve()}")
    print(f"  MSigDB:  {msigdb_dir.resolve()}")
    print(f"  DrugComb:{drugcomb_dir.resolve()}")
    print("\nDrug-target annotations are a separate input. Supply the verified annotation CSV to run_real.py.")
    print("Check all downloaded file versions and headers before starting preprocessing.")


if __name__ == "__main__":
    main()
