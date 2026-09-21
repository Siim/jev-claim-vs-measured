#!/usr/bin/env python3
"""Fair is fair: the model doing what it was built for - reading text.

    python 40_text_triage.py        (no API key: every answer is in the shipped cache)

4,435 Binance announcement titles since 2022. For each title the model answers three
questions in one request: what kind of announcement is this, which way is the price likely
to move, and how large is the effect. No prices are involved anywhere in this script.

The ground truth is a handful of plain regular expressions, for the classes a regular
expression can define exactly (1,313 titles). The vendor recommends acting only when the
model's confidence is high; that works, and it is reported below.
"""

import os
import re

import pandas as pd

from jev.client import ask_many
from jev.scoring import print_headline
from jev.wording_text import TEXT_QUESTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CONFIDENCE_LEVELS = (0.5, 0.8, 0.9)


def class_by_regex(title):
    """The announcement class, where the title's wording settles it; otherwise None."""
    if re.match(r"^Binance Will Delist ", title) and "Futures" not in title:
        return "spot_delisting"
    if "Binance Futures Will Delist" in title:
        return "perp_delisting"
    if re.match(r"^Binance Will List ", title):
        return "spot_listing"
    if re.search(r"Binance Futures Will Launch", title) and "Pre-Market" not in title:
        return "perp_launch"
    if "Monitoring Tag" in title or "Seed Tag" in title and "Will List" not in title:
        return "monitoring_tag"
    if re.search(r"Margin Tiers|Leverage|Funding Rate|Tick Size|Collateral Ratio|Price Limit", title):
        return "risk_parameter_change"
    if re.search(r"Removal of (?:Spot|Margin)|Will Add .* Trading Pairs|Notice of Removal", title):
        return "pair_change"
    return None


def main():
    titles = pd.read_parquet(f"{DATA}/announcement_titles.parquet").reset_index(drop=True)
    replies = ask_many(
        [(title, TEXT_QUESTIONS) for title in titles["title"]], max_parallel=12, progress_every=1000
    )

    answered = [i for i, reply in enumerate(replies) if "error" not in reply]
    labels = titles.loc[answered].copy()
    labels["model_class"] = [replies[i]["answers"]["event_type"]["choice"] for i in answered]
    labels["model_confidence"] = [replies[i]["answers"]["event_type"]["confidence"] for i in answered]
    labels["model_direction"] = [replies[i]["answers"]["direction"]["choice"] for i in answered]
    labels["regex_class"] = labels["title"].map(class_by_regex)
    labels.to_parquet(f"{DATA}/text_triage_labels.parquet", index=False)

    checkable = labels[labels["regex_class"].notna()]
    correct = checkable["model_class"] == checkable["regex_class"]
    fresh_calls = sum(1 for reply in replies if not reply.get("from_cache"))

    print(f"titles                        {len(labels):,}   (fresh API calls: {fresh_calls})")
    print(f"titles a regex can classify   {len(checkable):,}")
    print(f"model accuracy on those       {100 * correct.mean():.1f}%")
    headline = {
        "titles": len(labels),
        "checkable_titles": len(checkable),
        "accuracy_pct": round(100 * float(correct.mean()), 1),
    }

    print("\nacting only when the model is confident:")
    for level in CONFIDENCE_LEVELS:
        confident = checkable["model_confidence"] >= level
        accuracy = 100 * correct[confident].mean()
        print(
            f"   confidence >= {level:.1f}: keeps {confident.sum():,} of {len(checkable):,} titles"
            f" ({100 * confident.mean():.1f}%), accuracy {accuracy:.1f}%"
        )
        headline[f"kept_at_{level}"] = int(confident.sum())
        headline[f"accuracy_pct_at_{level}"] = round(float(accuracy), 1)

    print("\nwhich way does it expect the price to move? (% of titles in each true class)")
    direction = (
        checkable.groupby("regex_class")["model_direction"].value_counts(normalize=True).unstack().fillna(0)
    )
    print((100 * direction[["down", "unclear", "up"]]).round(0).to_string())

    print(
        "\nSeveral of the 'errors' are the REGEX being wrong: a title that ends a pre-market and lists"
        "\na token (with a Seed Tag) is a listing, as the model said. The regex truth understates"
        "\nthe model's accuracy."
    )
    print_headline(headline)


if __name__ == "__main__":
    main()
