"""Download real DepMap omics data + DrugComb synergy table and set up the data directory.

This script:
  1. Downloads DepMap 24Q2 omics files (expression, CNV, mutations, model metadata) via the DepMap API
  2. Copies gene-set GMT files from the depmap_omics-master repo
  3. Downloads the DrugComb summary synergy table from Zenodo
  4. Sets up the directory layout expected by pathsyn.preprocess

Usage:
    python scripts/setup_real_data.py
    python scripts/setup_real_data.py --out data/real
"""
import argparse
import csv
import io
import os
import shutil
import sys

import requests

# DepMap file names we need
DEPMAP_FILE_NAMES = [
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "OmicsCNGene.csv",
    "OmicsSomaticMutations.csv",
    "Model.csv",
]

# Zenodo record for DrugComb v1.5 summary data
DRUGCOMB_ZENODO_RECORD = "15235991"
DRUGCOMB_ZENODO_API = f"https://zenodo.org/api/records/{DRUGCOMB_ZENODO_RECORD}"


def download_file(url, dest, desc=None, max_retries=10):
    """Download a file with progress, automatic resume, and retry."""
    import time

    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  [skip] already exists: {os.path.basename(dest)} ({size_mb:.1f} MB)")
        return True

    tmp_path = dest + ".tmp"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    file_label = desc or os.path.basename(dest)
    print(f"  downloading {file_label} ...")

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                print(f"    (resuming from {downloaded / 1024 / 1024:.1f} MB, attempt {attempt}/{max_retries})")

            r = requests.get(url, stream=True, timeout=60, allow_redirects=True, headers=headers)

            if r.status_code == 200 and downloaded > 0:
                downloaded = 0
                mode = "wb"
            elif r.status_code in (200, 206):
                mode = "ab" if downloaded > 0 else "wb"
            else:
                r.raise_for_status()

            content_len = int(r.headers.get("content-length", 0)) if r.headers.get("content-length") else 0
            total = downloaded + content_len if content_len else 0

            with open(tmp_path, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r    {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="", flush=True)
                        else:
                            print(f"\r    {downloaded / 1024 / 1024:.1f} MB", end="", flush=True)
            print()

            if downloaded < 1000:
                print(f"  WARNING: file too small ({downloaded} bytes)")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return False

            if os.path.exists(dest):
                os.remove(dest)
            os.rename(tmp_path, dest)
            print(f"  OK: {downloaded / 1024 / 1024:.1f} MB")
            return True

        except (requests.RequestException, OSError) as e:
            print(f"\n  [attempt {attempt}/{max_retries} warning: {e}]")
            if attempt < max_retries:
                wait = min(30, 2 * attempt)
                print(f"  retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} attempts: {e}")
                return False
    return False


DEPMAP_24Q2_FIGSHARE_ARTICLE = "25880521"
DEPMAP_24Q2_URLS = {
    "Model.csv": "https://ndownloader.figshare.com/files/46489732",
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv": "https://ndownloader.figshare.com/files/46490878",
    "OmicsCNGene.csv": "https://ndownloader.figshare.com/files/46489771",
    "OmicsSomaticMutations.csv": "https://ndownloader.figshare.com/files/46500367",
}


def get_depmap_download_urls(file_names):
    """Query Figshare API or use known direct URLs for DepMap 24Q2."""
    urls = dict(DEPMAP_24Q2_URLS)
    try:
        r = requests.get(f"https://api.figshare.com/v2/articles/{DEPMAP_24Q2_FIGSHARE_ARTICLE}", timeout=30)
        if r.status_code == 200:
            for f in r.json().get("files", []):
                name = f.get("name")
                dl = f.get("download_url")
                if name in file_names and dl:
                    urls[name] = dl
    except Exception as e:
        print(f"  (Figshare API query: {e}, falling back to static 24Q2 endpoints)")
    return {f: urls[f] for f in file_names if f in urls}


def download_depmap_files(depmap_dir, file_names):
    """Download DepMap files using Figshare direct endpoints."""
    os.makedirs(depmap_dir, exist_ok=True)

    needed = []
    for f in file_names:
        p = os.path.join(depmap_dir, f)
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            continue
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
        needed.append(f)

    if not needed:
        print("  All DepMap files already downloaded.")
        return True

    urls = get_depmap_download_urls(needed)

    all_ok = True
    for fname in needed:
        if fname not in urls:
            print(f"  WARNING: No download URL found for {fname}")
            all_ok = False
            continue
        dest = os.path.join(depmap_dir, fname)
        ok = download_file(urls[fname], dest, fname)
        if not ok:
            all_ok = False

    return all_ok


def download_drugcomb_from_zenodo(drugcomb_dir):
    """Download DrugComb summary data from Zenodo."""
    os.makedirs(drugcomb_dir, exist_ok=True)

    # Check if we already have a summary CSV
    existing = [f for f in os.listdir(drugcomb_dir) if f.endswith(".csv") and "summary" in f.lower()]
    if existing:
        print(f"  [skip] already exists: {existing[0]}")
        return True

    print("  Querying Zenodo API for DrugComb files...")
    try:
        r = requests.get(DRUGCOMB_ZENODO_API, timeout=60)
        r.raise_for_status()
        record = r.json()
        files = record.get("files", [])

        # Find the summary CSV file
        summary_file = None
        for f in files:
            key = f.get("key", f.get("filename", ""))
            if "summary" in key.lower() and key.endswith(".csv"):
                summary_file = f
                break

        if summary_file is None:
            # Try any CSV file
            for f in files:
                key = f.get("key", f.get("filename", ""))
                if key.endswith(".csv"):
                    summary_file = f
                    break

        if summary_file is None:
            print("  WARNING: No CSV file found in Zenodo record")
            print(f"  Available files: {[f.get('key', '') for f in files]}")
            return False

        key = summary_file.get("key", summary_file.get("filename", "summary.csv"))
        url = summary_file.get("links", {}).get("self", "")
        if not url:
            url = f"https://zenodo.org/records/{DRUGCOMB_ZENODO_RECORD}/files/{key}?download=1"

        dest = os.path.join(drugcomb_dir, key)
        return download_file(url, dest, f"DrugComb {key}")

    except Exception as e:
        print(f"  WARNING: Zenodo API failed: {e}")
        # Fallback: try direct URL
        print("  Trying direct Zenodo download...")
        url = f"https://zenodo.org/records/{DRUGCOMB_ZENODO_RECORD}/files/summary_v_1_5.csv?download=1"
        return download_file(url, os.path.join(drugcomb_dir, "summary_v_1_5.csv"), "DrugComb summary")


def copy_genesets(repo_dir, out_msigdb):
    """Copy GMT gene set files from the depmap_omics repo and rename to match PathSyn's expected glob patterns."""
    os.makedirs(out_msigdb, exist_ok=True)

    gmt_mapping = {
        "hallmark.gmt": "h.all.v2024.Hs.symbols.gmt",
        "KEGG_2016.gmt": "c2.cp.kegg_legacy.v2024.Hs.symbols.gmt",
        "Reactome_2013.gmt": "c2.cp.reactome.v2024.Hs.symbols.gmt",
    }

    src_dir = os.path.join(repo_dir, "data", "genesets")
    if not os.path.isdir(src_dir):
        print(f"  ERROR: directory not found: {src_dir}")
        return False

    success = True
    for src_name, dst_name in gmt_mapping.items():
        src = os.path.join(src_dir, src_name)
        dst = os.path.join(out_msigdb, dst_name)
        if os.path.exists(dst):
            print(f"  [skip] already exists: {dst_name}")
            continue
        if not os.path.exists(src):
            print(f"  WARNING: source not found: {src}")
            success = False
            continue
        shutil.copy2(src, dst)
        print(f"  copied {src_name} -> {dst_name}")
    return success


def main():
    ap = argparse.ArgumentParser(description="Download real data for PathSyn pipeline")
    ap.add_argument("--out", default="data/real",
                    help="Root output directory for real data (default: data/real)")
    ap.add_argument("--depmap-repo",
                    default=r"C:\Users\user\Downloads\depmap_omics-master\depmap_omics-master",
                    help="Path to the depmap_omics-master repo (for gene sets)")
    ap.add_argument("--skip-depmap", action="store_true")
    ap.add_argument("--skip-drugcomb", action="store_true")
    a = ap.parse_args()

    depmap_dir = os.path.join(a.out, "depmap")
    msigdb_dir = os.path.join(a.out, "msigdb")
    drugcomb_dir = os.path.join(a.out, "drugcomb")
    processed_dir = os.path.join(a.out, "processed")
    for d in [depmap_dir, msigdb_dir, drugcomb_dir, processed_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- Step 1: Download DepMap omics data ----
    if not a.skip_depmap:
        print("\n=== Step 1: Downloading DepMap 24Q2 omics data ===")
        print("(These are large files, 100-600 MB each. Be patient...)\n")
        depmap_ok = download_depmap_files(depmap_dir, DEPMAP_FILE_NAMES)
        if not depmap_ok:
            print("\n[WARNING] Some DepMap files could not be downloaded automatically.")
            print("Please download them manually from: https://depmap.org/portal/download/")
            print(f"Place them in: {os.path.abspath(depmap_dir)}")
            print("Required files: " + ", ".join(DEPMAP_FILE_NAMES))
    else:
        print("\n=== Step 1: Skipping DepMap download ===")

    # ---- Step 2: Copy gene sets from depmap_omics repo ----
    print("\n=== Step 2: Setting up gene sets (MSigDB) from depmap_omics repo ===")
    if os.path.isdir(a.depmap_repo):
        copy_genesets(a.depmap_repo, msigdb_dir)
    else:
        print(f"  depmap_omics repo not found at: {a.depmap_repo}")
        print("  Trying to download MSigDB files directly...")
        msigdb_ver = "2024.1.Hs"
        base_url = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{msigdb_ver}/"
        for f in [f"h.all.v{msigdb_ver}.symbols.gmt",
                  f"c2.cp.kegg_legacy.v{msigdb_ver}.symbols.gmt",
                  f"c2.cp.reactome.v{msigdb_ver}.symbols.gmt"]:
            download_file(base_url + f, os.path.join(msigdb_dir, f))

    # ---- Step 3: Download DrugComb synergy data ----
    if not a.skip_drugcomb:
        print("\n=== Step 3: Downloading DrugComb synergy data from Zenodo ===")
        drugcomb_ok = download_drugcomb_from_zenodo(drugcomb_dir)
        if not drugcomb_ok:
            print("\n[WARNING] DrugComb download failed.")
            print("Download manually from: https://zenodo.org/records/15235991")
            print(f"Place the summary CSV in: {os.path.abspath(drugcomb_dir)}")
    else:
        print("\n=== Step 3: Skipping DrugComb download ===")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"\nDepMap data:  {os.path.abspath(depmap_dir)}")
    print(f"Gene sets:    {os.path.abspath(msigdb_dir)}")
    print(f"DrugComb:     {os.path.abspath(drugcomb_dir)}")

    depmap_present = all(
        os.path.exists(os.path.join(depmap_dir, f)) and os.path.getsize(os.path.join(depmap_dir, f)) > 1000
        for f in DEPMAP_FILE_NAMES
    )
    gmt_files = [f for f in os.listdir(msigdb_dir) if f.endswith(".gmt")]
    drugcomb_files = [f for f in os.listdir(drugcomb_dir) if f.endswith(".csv")]

    print(f"\nDepMap files present:  {'YES' if depmap_present else 'MISSING -- see warnings above'}")
    print(f"GMT gene sets:        {len(gmt_files)} files")
    print(f"DrugComb files:       {len(drugcomb_files)} files")

    if depmap_present and gmt_files and drugcomb_files:
        print("\nAll data ready! Next step: run `python run_real.py`")
    else:
        print("\n[ERROR] Some data is missing. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
