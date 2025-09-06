# logging_utils.py
import csv, os
from datetime import datetime

def append_record(path, rec: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    rec = {"timestamp": timestamp, **rec}

    # If file exists, read header; else, write new header
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames) if reader.fieldnames else []
    else:
        existing_fields = []

    # Union of columns: keep existing order, append new keys at the end
    new_keys = [k for k in rec.keys() if k not in existing_fields]
    fieldnames = existing_fields + new_keys if existing_fields else list(rec.keys())

    # Write (or rewrite) header if needed
    write_header = (not existing_fields) or (fieldnames != existing_fields)

    # If header changed and file already had rows, we need to rewrite the file with the new header
    if write_header and existing_fields:
        # Load all rows
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Rewrite with new header and rows (filling missing keys as "")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                for k in fieldnames:
                    row.setdefault(k, "")
                writer.writerow(row)

    # Append the new record
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header and not existing_fields:
            writer.writeheader()
        # Fill missing fields in this rec
        row = {k: rec.get(k, "") for k in fieldnames}
        writer.writerow(row)

def now_timestamp():
    return datetime.utcnow().isoformat()