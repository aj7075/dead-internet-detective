"""RSS fact-check ingestion. Usage: python ingestion/factcheck_rss.py"""
import json
from pathlib import Path

import feedparser

FEEDS = [
    ("snopes", "https://www.snopes.com/feed/"),
    ("afp", "https://factcheck.afp.com/feed"),
    ("politifact", "https://www.politifact.com/rss/all/"),
]


def ingest_fact_checks(output_dir: str = "data/real_cases") -> list[dict]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cases = []
    for source, url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:50]:
            cases.append({
                "source": source,
                "claim": entry.get("title", ""),
                "verdict": (entry.get("tags") or [{}])[0].get("term", "unknown").lower(),
                "primary_source_url": entry.get("link", ""),
                "summary": entry.get("summary", ""),
            })

    output_path = Path(output_dir) / "factcheck_cases.json"
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved {len(cases)} cases to {output_path}")
    return cases


if __name__ == "__main__":
    ingest_fact_checks()
