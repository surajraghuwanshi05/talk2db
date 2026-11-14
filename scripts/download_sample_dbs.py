"""
Download sample databases used for development and evaluation.

    uv run python -m scripts.download_sample_dbs

Fetches Chinook (a digital music store: artists, albums, tracks, invoices, ...)
from its official GitHub release, verifies SHA-256, and stores it in
sample_dbs/chinook/:

  • Chinook_Sqlite.sqlite   — zero-setup target for local dev + tests
  • Chinook_PostgreSql.sql  — loaded by docker-compose into the `chinook` Postgres DB

Files are git-ignored (they're third-party data); the checksums below pin the exact
version so everyone evaluates against identical data.
Chinook license: https://github.com/lerocha/chinook-database/blob/master/LICENSE.md
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/lerocha/chinook-database/releases/download/v1.4.5"

FILES = {
    "chinook/Chinook_Sqlite.sqlite": (
        f"{RELEASE}/Chinook_Sqlite.sqlite",
        "bdf635be69850bd3be09c9a2dbeef7ddfb80036bd3ef3381383cd03b61e4a61a",
    ),
    "chinook/Chinook_PostgreSql.sql": (
        f"{RELEASE}/Chinook_PostgreSql.sql",
        "e3fde5c1a5b51a2a91429a702c9ca6e69ba56e6c7f5e112724d70c3d03db695e",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    target_root = ROOT / "sample_dbs"
    failed = False
    for rel, (url, expected) in FILES.items():
        dest = target_root / rel
        if dest.exists() and _sha256(dest) == expected:
            print(f"ok       {rel} (already present)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {rel} ...")
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            resp = client.get(url)
            resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(resp.content)
        actual = _sha256(tmp)
        if actual != expected:
            tmp.unlink()
            print(f"FAILED   {rel}: checksum mismatch ({actual})", file=sys.stderr)
            failed = True
            continue
        tmp.replace(dest)
        print(f"ok       {rel} ({len(resp.content) // 1024} KiB, sha256 verified)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
