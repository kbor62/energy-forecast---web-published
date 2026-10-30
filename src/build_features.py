"""Build the feature file: price, history features and targets.

Calendar features are not stored; they depend on the horizon and are built
at training time by ``features.assemble``.

    python src/build_features.py --inp data/entsoe_NO_1_2020-01-01_now.parquet
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

import features as F
from config import PRICE_COL, TZ, repo_path
from price_io import describe_series, load_price_series

DEFAULT_INPUT = "data/entsoe_NO_1_2020-01-01_now.parquet"
DEFAULT_OUTPUT = "data/features_price_only.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the price-only feature file")
    ap.add_argument("--inp", default=DEFAULT_INPUT, help="Input price CSV/parquet")
    ap.add_argument("--out", default=DEFAULT_OUTPUT, help="Output parquet")
    ap.add_argument("--start", default=None, help="Optional start timestamp")
    ap.add_argument("--end", default=None, help="Optional end timestamp")
    ap.add_argument("--horizon", type=int, default=48, help="Max target horizon in hours")
    ap.add_argument("--csv", action="store_true", help="Also write a CSV copy for inspection")
    args = ap.parse_args()

    inp_path = repo_path(args.inp)
    out_path = repo_path(args.out)

    price = load_price_series(inp_path, tz=TZ)

    if args.start:
        price = price.loc[pd.Timestamp(args.start, tz=TZ) :]
    if args.end:
        price = price.loc[: pd.Timestamp(args.end, tz=TZ)]

    horizons = list(range(1, args.horizon + 1))
    history = F.history_features(price)
    targets = F.make_targets(price, horizons)

    out = pd.concat([price.rename(PRICE_COL), history, targets], axis=1)
    out.index.name = "timestamp"

    # Drop the last rows, which have no target.
    out = out.dropna(subset=[F.target_column(1)], how="all")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=True)
    if args.csv:
        out.to_csv(out_path.with_suffix(".csv"), index=True)

    meta = {
        "built_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "source_file": str(inp_path),
        "series": describe_series(price),
        "horizons": [1, args.horizon],
        "history_columns": F.history_columns(),
        "n_history_columns": len(F.history_columns()),
        "calendar_features": "built per horizon at t+h by features.assemble",
        "rows": int(len(out)),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Rows: {len(out):,}   history columns: {len(F.history_columns())}")
    print(f"Span: {out.index.min()} -> {out.index.max()}")


if __name__ == "__main__":
    main()
