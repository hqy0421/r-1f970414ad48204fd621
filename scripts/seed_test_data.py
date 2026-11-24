"""Seed a tiny but real RSSidian database + Annoy index for manual CLI checks."""
import json
import os
import sys

from annoy import AnnoyIndex

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rssidian.config import Config
from rssidian.models import init_db, Article, Feed

DIM = 768


def main():
    config = Config()
    session = init_db(config.db_path)
    if session.query(Article).count() == 0:
        feed = Feed(title="Seed Feed", url="https://seed.example/feed")
        session.add(feed)
        session.commit()
        bodies = [
            "renewable energy storage breakthrough in grid scale batteries",
            "quantum computing error correction reaches a new milestone",
            "urban farming reduces food miles in dense cities",
        ]
        index = AnnoyIndex(DIM, config.annoy_metric)
        for i, body in enumerate(bodies, start=1):
            vector = [((i * 37 + j * 13) % 97) / 97.0 for j in range(DIM)]
            session.add(Article(
                feed_id=feed.id,
                title=f"Seed article {i}",
                url=f"https://seed.example/{i}",
                guid=f"seed-guid-{i}",
                content=body,
                summary=body,
                embedding_generated=True,
                embedding_vector=json.dumps(vector),
            ))
            index.add_item(i, vector)
        session.commit()
        index.build(config.annoy_n_trees)
        index.save(config.annoy_index_path)
        print(f"Seeded {len(bodies)} articles and index at {config.annoy_index_path}")
    else:
        print("Database already seeded")
    session.close()


if __name__ == "__main__":
    main()
