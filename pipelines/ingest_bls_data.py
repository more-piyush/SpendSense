#!/usr/bin/env python3
"""
BLS CE Survey ingestion - memory-efficient version.
Processes one year at a time to avoid OOM.
"""
import io, os, json, hashlib, zipfile, re
from datetime import datetime
import pandas as pd
from minio import Minio

client = Minio("localhost:9000", access_key="firefly-access-key",
               secret_key="firefly-secret-key", secure=False)

YEARS = [20, 21, 22, 23, 24]
TMP = "/tmp"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("/tmp/bls_parquet", exist_ok=True)

# Step 17: Parse UCC mapping
def parse_hg_file(path):
    ucc_map = {}
    try:
        with open(path, encoding='latin-1') as f:
            for line in f:
                line = line.rstrip()
                if len(line) < 10:
                    continue
                match = re.search(r'\b(\d{6})\b', line)
                if match:
                    ucc = match.group(1)
                    name = line[:match.start()].strip().strip('-').strip()
                    if name:
                        ucc_map[ucc] = name
    except Exception as e:
        print(f"  Warning: {e}")
    return ucc_map

print("Step 17: Parsing UCC mappings...")
ucc_master = {}
for yr in YEARS:
    hg_path = f"{TMP}/CE-HG-Integ-20{yr}.txt"
    if os.path.exists(hg_path):
        m = parse_hg_file(hg_path)
        ucc_master.update(m)
print(f"  Total UCC codes: {len(ucc_master)}")

# Process ONE year at a time
summary = []
for yr in YEARS:
    zip_path = f"{TMP}/intrvw{yr}.zip"
    if not os.path.exists(zip_path):
        print(f"  Skipping 20{yr}")
        continue

    print(f"\nProcessing year 20{yr}...")
    mtbi_frames, fmli_frames = [], []

    with zipfile.ZipFile(zip_path) as z:
        for fname in z.namelist():
            base = os.path.basename(fname).lower()
            if not base.endswith('.csv'):
                continue
            try:
                if base.startswith('mtbi'):
                    # Only load needed columns to save memory
                    df = pd.read_csv(io.BytesIO(z.read(fname)),
                                     low_memory=False,
                                     usecols=lambda c: c.upper() in [
                                         'NEWID','UCCSEQ','COST_','REF_MO',
                                         'REF_YR','EXPNAME','ALCNO','SEQNO'])
                    df['SOURCE_YEAR'] = f"20{yr}"
                    mtbi_frames.append(df)
                elif base.startswith('fmli'):
                    df = pd.read_csv(io.BytesIO(z.read(fname)),
                                     low_memory=False,
                                     usecols=lambda c: c.upper() in [
                                         'NEWID','FINCBTXM','FAM_SIZE',
                                         'AGE_REF','REGION','FINLWT21'])
                    fmli_frames.append(df)
            except Exception as e:
                print(f"  Warning {base}: {e}")

    if not mtbi_frames:
        print(f"  No MTBI files for 20{yr}")
        continue

    mtbi = pd.concat(mtbi_frames, ignore_index=True)
    mtbi.columns = [c.upper() for c in mtbi.columns]
    del mtbi_frames

    fmli = pd.concat(fmli_frames, ignore_index=True) if fmli_frames else pd.DataFrame()
    fmli.columns = [c.upper() for c in fmli.columns]
    del fmli_frames

    print(f"  MTBI: {len(mtbi):,} rows")
    print(f"  FMLI: {len(fmli):,} rows")

    # Step 19: Add UCC item names
    ucc_col = next((c for c in mtbi.columns if 'UCC' in c), None)
    if ucc_col:
        mtbi['UCC_STR'] = mtbi[ucc_col].astype(str).str.zfill(6)
        mtbi['ITEM_NAME'] = mtbi['UCC_STR'].map(ucc_master).fillna('Unknown')

    # Step 20: Join with FMLI demographics
    if not fmli.empty and 'NEWID' in mtbi.columns and 'NEWID' in fmli.columns:
        fmli_slim = fmli.drop_duplicates(subset=['NEWID'])
        mtbi = mtbi.merge(fmli_slim, on='NEWID', how='left')
        print(f"  After join: {len(mtbi):,} rows, {len(mtbi.columns)} cols")
    del fmli

    # Step 21: Save parquet
    parquet_path = f"/tmp/bls_parquet/bls_20{yr}_{timestamp}.parquet"
    mtbi.to_parquet(parquet_path, index=False)
    size_mb = os.path.getsize(parquet_path) // 1024 // 1024
    print(f"  Saved parquet: {size_mb} MB")

    # Step 22: Upload to MinIO
    key = f"bls_ce_survey/processed/bls_20{yr}_{timestamp}.parquet"
    with open(parquet_path, 'rb') as f:
        client.put_object("processed-data", key, f, os.path.getsize(parquet_path))
    print(f"  Uploaded → processed-data/{key}")

    # Upload raw zip
    raw_key = f"bls_ce_survey/raw/intrvw{yr}.zip"
    with open(zip_path, 'rb') as f:
        client.put_object("raw-data", raw_key, f, os.path.getsize(zip_path))
    print(f"  Uploaded → raw-data/{raw_key}")

    summary.append({"year": f"20{yr}", "mtbi_rows": len(mtbi),
                     "parquet_key": key, "size_mb": size_mb})
    del mtbi

# Upload HG files and save lineage
for yr in YEARS:
    hg_path = f"{TMP}/CE-HG-Integ-20{yr}.txt"
    if os.path.exists(hg_path):
        with open(hg_path, 'rb') as f:
            client.put_object("raw-data",
                f"bls_ce_survey/hg/CE-HG-Integ-20{yr}.txt", f,
                os.path.getsize(hg_path))

lineage = {
    "pipeline": "bls_ce_survey_ingestion",
    "version": "1.0.0",
    "timestamp": timestamp,
    "source": "BLS Consumer Expenditures Survey PUMD",
    "source_url": "https://www.bls.gov/cex/pumd_data.htm",
    "ucc_codes_mapped": len(ucc_master),
    "years": summary,
}
lj = json.dumps(lineage, indent=2).encode()
client.put_object("raw-data", f"bls_ce_survey/lineage_{timestamp}.json",
                  io.BytesIO(lj), len(lj))

print("\n" + "="*50)
print("INGESTION COMPLETE")
for s in summary:
    print(f"  {s['year']}: {s['mtbi_rows']:,} rows → {s['size_mb']} MB parquet")
print(f"  UCC codes mapped: {len(ucc_master)}")
print("="*50)
