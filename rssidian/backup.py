"""
Backup and restore for the SQLite database and the Annoy vector index.

A backup is a *bundle* directory that keeps the two data stores side by side
and binds them together with a ``manifest.json``.  The manifest records the
actual on-disk versions (SHA-256, size) of both files and the entry range they
cover (article id range, number of indexed embeddings and a fingerprint of the
indexed id set), so a restore can guarantee that the same content resolves to
the same article through both the database and the vector index.

Legacy single-file backups (``YYYY-MM-DD.db`` next to the bundle directories)
remain supported: they contain no index, so restoring one isolates the
currently installed (and therefore foreign) index instead of loading it.
"""

import glob
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
DB_FILENAME = "rssidian.db"
INDEX_FILENAME = "annoy.idx"
MANIFEST_VERSION = 1

# Suffix used to move a foreign index out of the way so it can never be loaded.
ISOLATED_SUFFIX = ".isolated"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    if not os.path.exists(file_path):
        return 0
    return os.path.getsize(file_path) / (1024 * 1024)


def sha256_file(file_path: str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(file_path: str) -> Dict[str, Any]:
    """Return the version fingerprint (size + SHA-256) of a file."""
    stat = os.stat(file_path)
    return {
        "size": stat.st_size,
        "sha256": sha256_file(file_path),
    }


def _db_entry_range(db_path: str) -> Dict[str, Any]:
    """
    Describe the entry range covered by a database.

    Returns article id bounds, total article count, embedding count and a
    stable fingerprint of the set of embedded article ids.  Importing the ORM
    locally keeps this module lightweight at CLI startup.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM articles")).scalar() or 0
            min_id = conn.execute(text("SELECT MIN(id) FROM articles")).scalar()
            max_id = conn.execute(text("SELECT MAX(id) FROM articles")).scalar()
            embedded = conn.execute(
                text(
                    "SELECT COUNT(*) FROM articles "
                    "WHERE embedding_generated = 1 AND embedding_vector IS NOT NULL"
                )
            ).scalar() or 0
            ids = [
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT id FROM articles "
                        "WHERE embedding_generated = 1 AND embedding_vector IS NOT NULL "
                        "ORDER BY id"
                    )
                )
            ]
    finally:
        engine.dispose()

    id_digest = hashlib.sha256(",".join(str(i) for i in ids).encode("utf-8")).hexdigest()
    return {
        "article_count": total,
        "min_article_id": min_id,
        "max_article_id": max_id,
        "embedded_article_count": embedded,
        "embedded_ids_sha256": id_digest,
    }


def _index_entry_range(index_path: str, config: Config) -> Dict[str, Any]:
    """
    Describe the entry range covered by an Annoy index file.

    The index stores article ids as Annoy item keys.  Annoy's public API only
    exposes the number of items, not the key set, so the item count together
    with the manifest-bound file hash is the portable range description.  The
    matching database side records the exact embedded id set fingerprint.
    """
    from annoy import AnnoyIndex

    embedding_dim = 768  # all-mpnet-base-v2 dimension
    index = AnnoyIndex(embedding_dim, config.annoy_metric)
    index.load(index_path)
    try:
        actual_count = index.get_n_items()
    finally:
        index.unload()

    return {"item_count": actual_count}


def _build_manifest(config: Config, db_path: str, index_path: Optional[str]) -> Dict[str, Any]:
    """Build the manifest that binds a database file and its index together."""
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "annoy": {
            "metric": config.annoy_metric,
            "n_trees": config.annoy_n_trees,
            "embedding_dim": 768,
            "embedding_model": "all-mpnet-base-v2",
        },
        "database": {
            "filename": DB_FILENAME,
            **_file_fingerprint(db_path),
            **_db_entry_range(db_path),
        },
    }

    if index_path and os.path.exists(index_path):
        manifest["index"] = {
            "filename": INDEX_FILENAME,
            "included": True,
            **_file_fingerprint(index_path),
            **_index_entry_range(index_path, config),
        }
    else:
        manifest["index"] = {"filename": INDEX_FILENAME, "included": False}

    return manifest


def _bundle_dir(backup_dir: str, name: str) -> str:
    return os.path.join(backup_dir, name)


def _make_bundle_name(backup_dir: str) -> str:
    """Pick a unique bundle name for today, mirroring the legacy -N convention."""
    today = datetime.now().strftime("%Y-%m-%d")
    indices = []
    for path in glob.glob(os.path.join(backup_dir, f"{today}*")):
        base = os.path.basename(path).rstrip("/")
        suffix = base[len(today):]
        if suffix == "" or suffix == ".db":
            indices.append(1)
        elif suffix.startswith("-"):
            token = suffix[1:].split(".")[0]
            if token.isdigit():
                indices.append(int(token))
    index = (max(indices) + 1) if indices else 1
    return today if index == 1 else f"{today}-{index}"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def get_backup_list(config: Config) -> List[Dict[str, Any]]:
    """
    Get a list of available backups.

    Bundles (directories containing a manifest) and legacy single-file
    ``*.db`` backups are both returned, newest first.

    Returns:
        List of backup information dictionaries
    """
    backup_dir = config.backup_dir

    if not os.path.exists(backup_dir):
        return []

    backups: List[Dict[str, Any]] = []

    # New-style bundles
    for entry in sorted(os.listdir(backup_dir), reverse=True):
        entry_path = os.path.join(backup_dir, entry)
        manifest_path = os.path.join(entry_path, MANIFEST_FILENAME)
        if not os.path.isdir(entry_path) or not os.path.exists(manifest_path):
            continue

        info: Dict[str, Any] = {
            "date": entry,
            "path": entry_path,
            "is_bundle": True,
            "manifest_path": manifest_path,
            "filename": entry,
            "size_mb": 0.0,
            "has_index": False,
            "article_count": None,
            "index_item_count": None,
        }
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            db_info = manifest.get("database", {})
            idx_info = manifest.get("index", {})
            info["has_index"] = bool(idx_info.get("included"))
            info["article_count"] = db_info.get("article_count")
            info["index_item_count"] = idx_info.get("item_count")
            info["size_mb"] = (db_info.get("size", 0) + idx_info.get("size", 0)) / (
                1024 * 1024
            )
            info["manifest"] = manifest
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Error reading manifest for backup {entry}: {str(e)}")
            info["manifest"] = None
        backups.append(info)

    # Legacy single-file backups
    for backup_file in glob.glob(os.path.join(backup_dir, "*.db")):
        filename = os.path.basename(backup_file)
        date_str = filename.replace(".db", "")
        if "-" in date_str and date_str.split("-")[-1].isdigit():
            base_date = "-".join(date_str.split("-")[:-1])
            display_date = f"{base_date} ({date_str.split('-')[-1]})"
        else:
            display_date = date_str

        backups.append({
            "date": display_date,
            "path": backup_file,
            "is_bundle": False,
            "manifest_path": None,
            "manifest": None,
            "has_index": False,
            "article_count": None,
            "index_item_count": None,
            "size_mb": get_file_size_mb(backup_file),
            "filename": filename,
        })

    backups.sort(key=lambda x: x["filename"], reverse=True)
    return backups


def find_backup(backups: List[Dict[str, Any]], backup_date: str) -> Optional[Dict[str, Any]]:
    """
    Find a backup by date token against its real filename or bundle name.

    Matching against the display string used to fail for legacy files because
    the display date strips the ``.db`` suffix while the sort key kept it.
    """
    for backup in backups:
        if backup_date == backup["filename"] or backup_date in backup["filename"]:
            return backup
    # Fall back to display-date matching for suffixed same-day names.
    for backup in backups:
        if backup_date in backup["date"]:
            return backup
    return None


# ---------------------------------------------------------------------------
# Backup creation
# ---------------------------------------------------------------------------

def create_backup(config: Config) -> Optional[str]:
    """
    Create a backup bundle of the database together with the Annoy index.

    The bundle contains the database, the index (when present) and a manifest
    binding their actual file versions and entry ranges.  Files are copied
    into a temporary directory first and moved into place only once every
    copy has succeeded, so a failed backup never leaves a partial bundle.

    Args:
        config: Configuration

    Returns:
        Path to the created backup bundle, or None if failed
    """
    db_path = config.db_path
    index_path = config.annoy_index_path
    backup_dir = config.backup_dir

    if not os.path.exists(db_path):
        logger.error(f"Database file does not exist: {db_path}")
        return None

    os.makedirs(backup_dir, exist_ok=True)

    bundle_name = _make_bundle_name(backup_dir)
    bundle_path = _bundle_dir(backup_dir, bundle_name)
    staging_path = f"{bundle_path}.staging"
    if os.path.exists(staging_path):
        shutil.rmtree(staging_path)

    try:
        os.makedirs(staging_path)

        db_target = os.path.join(staging_path, DB_FILENAME)
        shutil.copy2(db_path, db_target)
        logger.info(f"Copied database into backup bundle: {db_path}")

        index_included = False
        if os.path.exists(index_path):
            index_target = os.path.join(staging_path, INDEX_FILENAME)
            shutil.copy2(index_path, index_target)
            index_included = True
            logger.info(f"Copied vector index into backup bundle: {index_path}")
        else:
            logger.warning(
                "No Annoy index found; backup will contain the database only. "
                "Restore will require rebuilding the vector index before semantic search works."
            )

        # Fingerprint the copies inside the bundle, not the live files, so the
        # manifest describes exactly what the bundle holds.
        manifest = _build_manifest(
            config,
            db_target,
            os.path.join(staging_path, INDEX_FILENAME) if index_included else None,
        )
        with open(os.path.join(staging_path, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        os.replace(staging_path, bundle_path)
        logger.info(f"Created backup bundle at {bundle_path}")
        return bundle_path
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        if os.path.exists(staging_path):
            shutil.rmtree(staging_path, ignore_errors=True)
        return None


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def _verify_manifest(bundle_path: str, manifest: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verify that the files inside a bundle match the manifest fingerprints.

    A mismatch means the bundle was tampered with or corrupted; the caller
    must refuse to restore from it.
    """
    errors: List[str] = []

    db_info = manifest.get("database", {})
    db_path = os.path.join(bundle_path, db_info.get("filename", DB_FILENAME))
    if not os.path.exists(db_path):
        errors.append(f"database file missing from bundle: {db_path}")
    else:
        actual = _file_fingerprint(db_path)
        if actual["sha256"] != db_info.get("sha256"):
            errors.append("database file does not match manifest (SHA-256 mismatch)")

    idx_info = manifest.get("index", {})
    if idx_info.get("included"):
        index_path = os.path.join(bundle_path, idx_info.get("filename", INDEX_FILENAME))
        if not os.path.exists(index_path):
            errors.append("manifest claims an index but the index file is missing")
        else:
            actual = _file_fingerprint(index_path)
            if actual["sha256"] != idx_info.get("sha256"):
                errors.append("index file does not match manifest (SHA-256 mismatch)")

    return (not errors), errors


def isolate_foreign_index(config: Config, reason: str) -> Optional[str]:
    """
    Move the currently installed Annoy index aside so it cannot be loaded.

    Used when the database is restored without a matching index: loading the
    pre-restore index against the restored database would resolve article ids
    to the wrong content.  The file is renamed rather than deleted so the
    user can inspect or remove it manually.

    Args:
        config: Configuration
        reason: Human-readable reason recorded in the log

    Returns:
        The path the index was moved to, or None if nothing was moved
    """
    index_path = config.annoy_index_path
    if not os.path.exists(index_path):
        return None

    isolated_path = f"{index_path}{ISOLATED_SUFFIX}"
    counter = 1
    while os.path.exists(isolated_path):
        isolated_path = f"{index_path}.{counter}{ISOLATED_SUFFIX}"
        counter += 1

    try:
        os.replace(index_path, isolated_path)
        logger.warning(
            f"Isolated foreign vector index ({reason}): {index_path} -> {isolated_path}. "
            "Rebuild it with 'rssidian backfill-embeddings' to restore semantic search."
        )
        return isolated_path
    except OSError as e:
        logger.error(f"Failed to isolate foreign index {index_path}: {str(e)}")
        return None


def restore_backup(backup_date: str, config: Config, force: bool = False) -> bool:
    """
    Restore a database (and, when available, its matching vector index).

    Full bundles restore both files atomically as a pair, so the same content
    is found through the same article ids after the restore.  Database-only
    backups restore just the database and isolate any installed index rather
    than risk loading an index built for a different database.

    Args:
        backup_date: Date string to identify the backup
        config: Configuration
        force: Whether to force restore without confirmation

    Returns:
        True if restore was successful, False otherwise
    """
    backups = get_backup_list(config)

    target_backup = find_backup(backups, backup_date)

    if not target_backup:
        logger.error(f"No backup found for date: {backup_date}")
        return False

    db_path = config.db_path
    index_path = config.annoy_index_path
    temp_db = f"{db_path}.temp_backup"
    temp_index = f"{index_path}.temp_backup"

    try:
        if target_backup["is_bundle"]:
            ok = _restore_bundle(target_backup, config, temp_db, temp_index, force)
        else:
            ok = _restore_legacy_db(target_backup, config, temp_db, force)
        return ok
    finally:
        for temp_path in (temp_db, temp_index):
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


def _restore_bundle(
    backup: Dict[str, Any],
    config: Config,
    temp_db: str,
    temp_index: str,
    force: bool,
) -> bool:
    """Restore a full manifest-bound bundle."""
    bundle_path = backup["path"]
    manifest = backup.get("manifest")

    if not manifest:
        logger.error(f"Backup manifest is unreadable: {bundle_path}")
        return False

    valid, errors = _verify_manifest(bundle_path, manifest)
    if not valid:
        for error in errors:
            logger.error(f"Refusing to restore {bundle_path}: {error}")
        return False

    db_path = config.db_path
    index_path = config.annoy_index_path
    bundle_db = os.path.join(
        bundle_path, manifest.get("database", {}).get("filename", DB_FILENAME)
    )

    idx_info = manifest.get("index", {})
    bundle_index = (
        os.path.join(bundle_path, idx_info.get("filename", INDEX_FILENAME))
        if idx_info.get("included")
        else None
    )

    # Keep rollback copies of whatever is currently live.
    if os.path.exists(db_path):
        shutil.copy2(db_path, temp_db)
    if bundle_index and os.path.exists(index_path):
        shutil.copy2(index_path, temp_index)

    try:
        shutil.copy2(bundle_db, db_path)

        if bundle_index:
            # Restore the index as a pair with its database.  Make sure the
            # target directory exists (config path may be brand new).
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            shutil.copy2(bundle_index, index_path)
            logger.info(
                f"Restored matching vector index ({idx_info.get('item_count')} items) "
                f"from {bundle_index}"
            )
        else:
            # Database-only bundle: never leave a foreign index in place.
            isolate_foreign_index(
                config,
                reason="backup bundle contains no matching index",
            )

        logger.info(f"Restored database from {bundle_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to restore backup: {str(e)}")
        _rollback(db_path, index_path, temp_db, temp_index)
        return False


def _restore_legacy_db(
    backup: Dict[str, Any],
    config: Config,
    temp_db: str,
    force: bool,
) -> bool:
    """Restore a legacy single-file database backup with no index partner."""
    backup_path = backup["path"]
    db_path = config.db_path

    if not os.path.exists(backup_path):
        logger.error(f"Backup file does not exist: {backup_path}")
        return False

    if os.path.exists(db_path):
        try:
            shutil.copy2(db_path, temp_db)
            logger.info(f"Created temporary backup at {temp_db}")
        except Exception as e:
            logger.error(f"Failed to create temporary backup: {str(e)}")
            if not force:
                return False

    try:
        shutil.copy2(backup_path, db_path)
        logger.info(f"Restored database from {backup_path}")

        # The installed index was built for the previous database and its
        # item ids no longer map to the restored rows — isolate it.
        isolate_foreign_index(
            config,
            reason="legacy database backup has no matching index",
        )
        return True
    except Exception as e:
        logger.error(f"Failed to restore backup: {str(e)}")
        _rollback(db_path, config.annoy_index_path, temp_db, None)
        return False


def _rollback(
    db_path: str,
    index_path: str,
    temp_db: Optional[str],
    temp_index: Optional[str],
) -> None:
    """Restore the live files from their pre-restore copies."""
    if temp_db and os.path.exists(temp_db):
        try:
            shutil.copy2(temp_db, db_path)
            logger.info("Restored database from temporary backup")
        except OSError as e:
            logger.error(f"Failed to restore database from temporary backup: {str(e)}")
    if temp_index and os.path.exists(temp_index):
        try:
            shutil.copy2(temp_index, index_path)
            logger.info("Restored vector index from temporary backup")
        except OSError as e:
            logger.error(f"Failed to restore vector index from temporary backup: {str(e)}")
