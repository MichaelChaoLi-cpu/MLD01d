#!/usr/bin/env python3
"""Prepare a clean DOCX while correctly accepting tracked table-row deletions.

The bundled clean-generation helper accepts run-level revisions but leaves empty
table rows behind when Word records a whole-row deletion in ``w:trPr/w:del``.
This wrapper removes those rows in a temporary copy only, then delegates all
other revision acceptance to the standard helper.  The markup source is never
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source(source: Path, prepared: Path) -> dict[str, int]:
    with zipfile.ZipFile(source, "r") as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
        infos = archive.infolist()

    document = etree.fromstring(entries["word/document.xml"])
    removed_rows = 0
    removed_empty_tables = 0
    for row in list(document.iter(W + "tr")):
        row_properties = row.find(W + "trPr")
        if row_properties is None or row_properties.find(W + "del") is None:
            continue
        parent = row.getparent()
        if parent is not None:
            parent.remove(row)
            removed_rows += 1

    for table in list(document.iter(W + "tbl")):
        if table.find(W + "tr") is None:
            parent = table.getparent()
            if parent is not None:
                parent.remove(table)
                removed_empty_tables += 1

    entries["word/document.xml"] = etree.tostring(
        document,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=None,
    )
    with zipfile.ZipFile(prepared, "w") as target:
        for info in infos:
            target.writestr(info, entries[info.filename])
    return {
        "tracked_table_rows_removed_in_temporary_copy": removed_rows,
        "empty_tables_removed_in_temporary_copy": removed_empty_tables,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--standard-helper", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    source_sha_before = sha256_path(source)

    fd, temp_name = tempfile.mkstemp(prefix=".row-accepted-", suffix=".docx")
    os.close(fd)
    prepared = Path(temp_name)
    try:
        table_receipt = prepare_source(source, prepared)
        result = subprocess.run(
            [
                os.fspath(Path(os.sys.executable)),
                args.standard_helper,
                "--source",
                os.fspath(prepared),
                "--output",
                os.fspath(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        receipt = json.loads(result.stdout)
    finally:
        prepared.unlink(missing_ok=True)

    source_sha_after = sha256_path(source)
    if source_sha_before != source_sha_after:
        raise SystemExit("markup source changed during clean generation")

    receipt.update(table_receipt)
    receipt["source"] = os.fspath(source)
    receipt["source_sha256_before"] = source_sha_before
    receipt["source_sha256_after"] = source_sha_after
    receipt["markup_untouched"] = True
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
