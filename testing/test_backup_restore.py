"""
End-to-end tests for the backup/restore and index-consistency design.

Covers the requirement that an offline backup binds the SQLite database and
the Annoy index to their actual versions and entry ranges, and that a
database-only restore never opens a foreign index: search degrades to
content/keyword matching with an explicit pending-rebuild signal instead.

These tests build real SQLite files and real Annoy indices but stub out the
embedding model so no heavy ML dependency is loaded.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from annoy import AnnoyIndex

from rssidian import backup
from rssidian.core import RSSProcessor
from rssidian.models import Article, Feed, init_db

EMBEDDING_DIM = 768  # must match RSSProcessor's all-mpnet-base-v2 dimension
METRIC = "angular"


class FakeConfig:
    """Minimal config standing in for rssidian.config.Config."""

    def __init__(self, root):
        self._root = root
        self.db_path = os.path.join(root, "data", "rssidian.db")
        self.backup_dir = os.path.join(root, "backups")
        self._index_path = os.path.join(root, "config", "annoy.idx")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    @property
    def annoy_index_path(self):
        return self._index_path

    annoy_n_trees = 10
    annoy_metric = METRIC
    excerpt_length = 300


def make_index(path, id_to_vector):
    index = AnnoyIndex(EMBEDDING_DIM, METRIC)
    for article_id, vector in id_to_vector.items():
        index.add_item(article_id, vector)
    index.build(10)
    index.save(path)


def seed_database(config, articles):
    """Create a feed + articles and return the session."""
    session = init_db(config.db_path)
    feed = Feed(title="Test Feed", url="https://example.com/feed")
    session.add(feed)
    session.commit()
    for article in articles:
        session.add(Article(
            feed_id=feed.id,
            title=article["title"],
            url=f"https://example.com/{article['id']}",
            guid=f"guid-{article['id']}",
            content=article["content"],
            summary=article.get("summary"),
            embedding_generated=article.get("embedded", True),
            embedding_vector=json.dumps(article["vector"]) if article.get("vector") else None,
        ))
    session.commit()
    return session


def vector_for(article_id):
    # Deterministic distinct vector per id.
    base = [(article_id + 1) % 7, (article_id + 2) % 5, article_id % 3]
    vector = [float((x + article_id) % 11) / 11.0 for x in range(EMBEDDING_DIM)]
    vector[0] = (base[0] + 1) / 8.0
    return vector


def build_index_from_db(config):
    session = init_db(config.db_path)
    rows = session.query(Article).filter(
        Article.embedding_generated == True,
        Article.embedding_vector.isnot(None)
    ).all()
    mapping = {row.id: json.loads(row.embedding_vector) for row in rows}
    make_index(config.annoy_index_path, mapping)
    session.close()


class TestBackupBundle:
    def setup_method(self):
        self.root = tempfile.mkdtemp(prefix="rssidian-test-")
        self.config = FakeConfig(self.root)

    def teardown_method(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_full_bundle_contains_db_index_and_manifest(self):
        articles = [
            {"id": i, "title": f"Article {i}", "content": f"content about topic {i}",
             "vector": vector_for(i)}
            for i in range(1, 6)
        ]
        session = seed_database(self.config, articles)
        session.close()
        build_index_from_db(self.config)

        bundle_path = backup.create_backup(self.config)
        assert bundle_path and os.path.isdir(bundle_path)

        files = set(os.listdir(bundle_path))
        assert files == {"rssidian.db", "annoy.idx", "manifest.json"}

        with open(os.path.join(bundle_path, "manifest.json")) as f:
            manifest = json.load(f)

        assert manifest["manifest_version"] == backup.MANIFEST_VERSION
        assert manifest["database"]["article_count"] == 5
        assert manifest["database"]["embedded_article_count"] == 5
        assert manifest["database"]["min_article_id"] == 1
        assert manifest["database"]["max_article_id"] == 5
        assert manifest["index"]["included"] is True
        # Manifest fingerprints describe the copies, not the live files.
        assert manifest["database"]["sha256"] == backup.sha256_file(
            os.path.join(bundle_path, "rssidian.db"))
        assert manifest["index"]["sha256"] == backup.sha256_file(
            os.path.join(bundle_path, "annoy.idx"))

    def test_manifest_entry_range_is_bound_to_both_sides(self):
        articles = [
            {"id": i, "title": f"T{i}", "content": f"body {i}", "vector": vector_for(i)}
            for i in (1, 2, 3)
        ]
        session = seed_database(self.config, articles)
        session.close()
        build_index_from_db(self.config)

        backup.create_backup(self.config)
        listed = backup.get_backup_list(self.config)
        assert len(listed) == 1
        entry = listed[0]
        assert entry["is_bundle"] is True
        assert entry["has_index"] is True
        assert entry["article_count"] == 3
        assert entry["index_item_count"] == 4  # Annoy span = max id (3) + 1

    def test_database_only_backup_when_index_missing(self):
        articles = [{"id": 1, "title": "Solo", "content": "lonely content",
                     "vector": vector_for(1)}]
        session = seed_database(self.config, articles)
        session.close()
        assert not os.path.exists(self.config.annoy_index_path)

        bundle_path = backup.create_backup(self.config)
        with open(os.path.join(bundle_path, "manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["index"]["included"] is False
        assert not os.path.exists(os.path.join(bundle_path, "annoy.idx"))

    def test_tampered_bundle_is_refused(self):
        articles = [{"id": 1, "title": "A", "content": "alpha", "vector": vector_for(1)}]
        session = seed_database(self.config, articles)
        session.close()
        build_index_from_db(self.config)

        bundle_path = backup.create_backup(self.config)
        # Corrupt the bundled database after the manifest was written.
        with open(os.path.join(bundle_path, "rssidian.db"), "ab") as f:
            f.write(b"corruption")

        restored = backup.restore_backup(
            os.path.basename(bundle_path), self.config, force=True
        )
        assert restored is False
        # Live database untouched.
        live_session = init_db(self.config.db_path)
        assert live_session.query(Article).count() == 1
        live_session.close()


class TestRestoreSemantics:
    def setup_method(self):
        self.root = tempfile.mkdtemp(prefix="rssidian-test-")
        self.config = FakeConfig(self.root)

    def teardown_method(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _make_processor(self):
        session = init_db(self.config.db_path)
        processor = RSSProcessor(self.config, session)
        # Stub the embedding model so semantic tests never load ML weights.
        processor._embedding_model = _DummyModel()
        return processor, session

    def test_full_restore_keeps_index_and_database_as_a_pair(self):
        # Snapshot at version 1.
        articles = [
            {"id": i, "title": f"Original {i}",
             "content": f"original content {i} quantum", "vector": vector_for(i)}
            for i in range(1, 4)
        ]
        session = seed_database(self.config, articles)
        session.close()
        build_index_from_db(self.config)
        bundle_name = os.path.basename(backup.create_backup(self.config))

        # Simulate later state: a different database with different id->content.
        os.remove(self.config.db_path)
        if os.path.exists(self.config.annoy_index_path):
            os.remove(self.config.annoy_index_path)
        later = [
            {"id": i, "title": f"Changed {i}",
             "content": f"totally different blockchain text {i}", "vector": vector_for(100 + i)}
            for i in range(1, 4)
        ]
        session = seed_database(self.config, later)
        session.close()
        build_index_from_db(self.config)

        assert backup.restore_backup(bundle_name, self.config, force=True) is True

        processor, session = self._make_processor()
        status = processor.index_status()
        assert status["available"] is True, status["reason"]
        assert status["mode"] == "semantic"

        # The restored pair must resolve the original content by id again.
        article = session.query(Article).filter_by(id=1).one()
        assert article.content == "original content 1 quantum"
        restored_vec = processor._vector_index.get_item_vector(1)
        expected = vector_for(1)
        # Annoy persists vectors as float32, so compare with tolerance.
        assert all(abs(a - b) < 1e-5 for a, b in zip(restored_vec, expected))
        session.close()

    def test_db_only_restore_isolates_foreign_index_and_falls_back(self):
        # Version 1 database + matching index.
        articles = [
            {"id": i, "title": f"Keep {i}",
             "content": f"renewable energy research {i}", "vector": vector_for(i)}
            for i in range(1, 4)
        ]
        session = seed_database(self.config, articles)
        session.close()
        build_index_from_db(self.config)
        assert os.path.exists(self.config.annoy_index_path)

        # Take a legacy-style database-only backup (flat file in backup dir).
        legacy_path = os.path.join(self.config.backup_dir, "2025-01-01.db")
        shutil.copy2(self.config.db_path, legacy_path)

        # Database evolves: new rows, same index file left in place (the
        # dangerous pre-fix scenario — index ids now map to other content).
        os.remove(self.config.db_path)
        changed = [
            {"id": i, "title": f"New {i}",
             "content": f"cryptocurrency market move {i}", "vector": vector_for(200 + i)}
            for i in range(1, 5)
        ]
        session = seed_database(self.config, changed)
        session.close()
        # Deliberately do NOT rebuild the index: stale index still on disk.
        assert os.path.exists(self.config.annoy_index_path)

        # Restore the old database while the foreign index is installed.
        assert backup.restore_backup("2025-01-01", self.config, force=True) is True

        # Foreign index must have been isolated, not left loadable.
        assert not os.path.exists(self.config.annoy_index_path)
        isolated = glob_isolated(self.config.annoy_index_path)
        assert isolated, "expected an .isolated copy of the foreign index"

        processor, session = self._make_processor()
        status = processor.index_status()
        assert status["available"] is False
        assert status["mode"] == "keyword_fallback"

        # Search still works against restored content, via the body text.
        results = processor.search("renewable energy", max_results=10)
        assert results, "keyword fallback should find restored articles"
        assert all(r["search_mode"] == "keyword_fallback" for r in results)
        titles = {r["title"] for r in results}
        assert "Keep 1" in titles
        # And it must not surface content that no longer exists.
        assert all("cryptocurrency" not in (r["excerpt"] or "") for r in results)
        session.close()

    def test_stale_index_without_restore_is_detected_and_isolated(self):
        # Build a valid pair first.
        articles = [
            {"id": i, "title": f"P{i}", "content": f"machine learning {i}",
             "vector": vector_for(i)}
            for i in range(1, 4)
        ]
        session = seed_database(self.config, articles)
        session.close()
        build_index_from_db(self.config)

        # Swap the database under the index without touching the index.
        os.remove(self.config.db_path)
        other = [
            {"id": i, "title": f"Q{i}", "content": f"deep space mission {i}",
             "vector": vector_for(300 + i)}
            for i in range(1, 6)
        ]
        session = seed_database(self.config, other)
        session.close()

        processor, session = self._make_processor()
        status = processor.index_status()
        assert status["available"] is False
        assert status["mode"] == "keyword_fallback"
        # Loading it must have isolated the stale file.
        assert not os.path.exists(self.config.annoy_index_path)
        assert glob_isolated(self.config.annoy_index_path)

        results = processor.search("deep space", max_results=10)
        assert results and all(r["search_mode"] == "keyword_fallback" for r in results)
        session.close()

    def test_rebuilt_index_is_accepted_again(self):
        articles = [
            {"id": i, "title": f"R{i}", "content": f"fusion reactor update {i}",
             "vector": vector_for(i)}
            for i in range(1, 4)
        ]
        session = seed_database(self.config, articles)
        session.close()
        build_index_from_db(self.config)

        processor, session = self._make_processor()
        assert processor.index_status()["available"] is True
        results = processor.search("fusion reactor", max_results=5)
        assert results
        # Semantic path does not tag the fallback marker.
        assert "search_mode" not in results[0]
        session.close()


class TestEdgeCases:
    def setup_method(self):
        self.root = tempfile.mkdtemp(prefix="rssidian-test-")
        self.config = FakeConfig(self.root)

    def teardown_method(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_empty_database_with_orphan_index_uses_fallback(self):
        # A database with no embedded rows but an index file on disk.
        session = init_db(self.config.db_path)
        session.close()
        make_index(self.config.annoy_index_path, {1: vector_for(1)})

        session = init_db(self.config.db_path)
        processor = RSSProcessor(self.config, session)
        status = processor.index_status()
        assert status["available"] is False
        assert status["mode"] == "keyword_fallback"
        assert processor.search("anything", max_results=5) == []
        session.close()

    def test_same_day_backups_get_unique_bundle_names(self):
        articles = [{"id": 1, "title": "A", "content": "alpha", "vector": vector_for(1)}]
        session = seed_database(self.config, articles)
        session.close()
        build_index_from_db(self.config)

        first = backup.create_backup(self.config)
        second = backup.create_backup(self.config)
        assert first != second
        assert os.path.isdir(first) and os.path.isdir(second)
        names = [b["filename"] for b in backup.get_backup_list(self.config)]
        assert len(names) == len(set(names))

    def test_keyword_search_finds_same_article_by_body_content(self):
        articles = [
            {"id": 1, "title": "Grid", "content": "renewable battery storage deployment",
             "vector": vector_for(1)},
            {"id": 2, "title": "Markets", "content": "cryptocurrency trading volatility",
             "vector": vector_for(2)},
        ]
        session = seed_database(self.config, articles)
        session.close()
        # No index at all: pure fallback.
        processor = RSSProcessor(self.config, init_db(self.config.db_path))
        results = processor.search("battery storage", max_results=10)
        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["search_mode"] == "keyword_fallback"

        # The same query on a freshly recreated processor returns the same id.
        processor2 = RSSProcessor(self.config, init_db(self.config.db_path))
        again = processor2.search("battery storage", max_results=10)
        assert [r["id"] for r in again] == [r["id"] for r in results]
        processor.db_session.close()
        processor2.db_session.close()

    def test_isolate_foreign_index_keeps_file_and_clears_live_path(self):
        make_index(self.config.annoy_index_path, {1: vector_for(1)})
        moved = backup.isolate_foreign_index(self.config, reason="unit test")
        assert moved and os.path.exists(moved)
        assert not os.path.exists(self.config.annoy_index_path)
        # Idempotent naming when several foreign indices accumulate.
        make_index(self.config.annoy_index_path, {2: vector_for(2)})
        moved_again = backup.isolate_foreign_index(self.config, reason="unit test")
        assert moved_again != moved
        assert os.path.exists(moved) and os.path.exists(moved_again)


class TestLegacyBackupListing:
    def setup_method(self):
        self.root = tempfile.mkdtemp(prefix="rssidian-test-")
        self.config = FakeConfig(self.root)

    def teardown_method(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_legacy_db_files_still_listed(self):
        articles = [{"id": 1, "title": "Legacy", "content": "old", "vector": vector_for(1)}]
        session = seed_database(self.config, articles)
        session.close()
        legacy = os.path.join(self.config.backup_dir, "2024-12-31.db")
        shutil.copy2(self.config.db_path, legacy)

        listed = backup.get_backup_list(self.config)
        assert len(listed) == 1
        assert listed[0]["is_bundle"] is False
        assert listed[0]["has_index"] is False


def glob_isolated(index_path):
    import glob
    return glob.glob(f"{index_path}*.isolated")


class _DummyModel:
    """Deterministic stand-in for SentenceTransformer in semantic-path tests."""

    def encode(self, text):
        # Hash text into a stable unit-ish vector of EMBEDDING_DIM elements.
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(EMBEDDING_DIM)]
        # Make queries about "fusion" land near article vectors well enough
        # that get_nns_by_vector returns rows; ordering is not asserted here.
        class Result(list):
            def tolist(self):
                return list(self)
        return Result(raw)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
