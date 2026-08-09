"""
mock_api.py
-----------
A small simulated REST "News API" source -- the third heterogeneous source
required by Section B (crawling + dataset + API). It is intentionally a local,
dependency-free stand-in (a Python list of dicts, exactly the shape a real
`requests.get(...).json()` response would have) so the pipeline is 100%
reproducible on a machine with no internet access, such as the BITS virtual
lab. The function signature (`fetch_latest_articles`) is written the way a
real API client would be, so swapping in `requests` + a real API key later is
a one-line change (see the commented-out real implementation below).
"""
from __future__ import annotations

from typing import Dict, List

_MOCK_API_PAYLOAD: List[Dict] = [
    {"id": "API001", "category": "tech", "title": "Chipmaker unveils new AI accelerator",
     "text": "A leading chipmaker unveiled a new AI accelerator chip promising faster training "
             "for large language models while cutting datacenter power consumption significantly."},
    {"id": "API002", "category": "business", "title": "Central bank holds interest rates steady",
     "text": "The central bank held interest rates steady this quarter, citing cooling inflation "
             "and resilient employment figures across manufacturing and services sectors."},
    {"id": "API003", "category": "sport", "title": "Underdog wins season-opening tournament",
     "text": "An underdog team claimed the season-opening tournament title after a dramatic "
             "extra-time finish, ending the reigning champions' two-year winning streak."},
    {"id": "API004", "category": "entertainment", "title": "Streaming service renews hit drama",
     "text": "A major streaming service renewed its hit drama series for two more seasons after "
             "record viewership numbers during the show's latest finale week."},
    {"id": "API005", "category": "politics", "title": "Lawmakers debate new education funding bill",
     "text": "Lawmakers spent the session debating a new education funding bill that would "
             "increase per-student spending while shifting oversight to regional boards."},
    {"id": "API006", "category": "tech", "title": "Startup raises funding for cloud security tool",
     "text": "A cybersecurity startup raised a new funding round for its cloud security platform, "
             "which automatically flags misconfigured storage buckets across multiple providers."},
    {"id": "API007", "category": "business", "title": "Retailer reports strong quarterly earnings",
     "text": "A major retailer reported stronger than expected quarterly earnings, driven by "
             "online sales growth and improved margins in its logistics network."},
    {"id": "API008", "category": "sport", "title": "Star player signs long-term contract extension",
     "text": "A star player signed a long-term contract extension, ending months of transfer "
             "speculation ahead of the upcoming season opener."},
    {"id": "API009", "category": "entertainment", "title": "Box office tops estimates on opening weekend",
     "text": "A big-budget film topped box office estimates on its opening weekend, boosted by "
             "strong reviews and an aggressive international marketing campaign."},
    {"id": "API010", "category": "politics", "title": "City council approves new transit plan",
     "text": "The city council approved a new transit plan that expands bus rapid-transit lanes "
             "and adds funding for electric vehicle charging infrastructure downtown."},
]


def fetch_latest_articles() -> List[Dict]:
    """Simulates `GET /v1/articles/latest` on a news API.

    Returns records shaped like: {id, category, title, text}.
    Swap for a real client when internet access is available, e.g.:

        import requests
        resp = requests.get(API_ENDPOINT, params={"apiKey": API_KEY}, timeout=5)
        return resp.json()["articles"]
    """
    return list(_MOCK_API_PAYLOAD)


__all__ = ["fetch_latest_articles"]
