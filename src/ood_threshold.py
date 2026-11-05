"""Step 4b - Calibrate the applicability-domain (out-of-scope) similarity threshold.

In-domain probes : 2,451 terse queries (first 3 words of each English name, as in task T2).
Out-of-domain    : 20 items that are NOT French food products (non-food goods + Indian foods absent
                   from AGRIBALYSE). A threshold is good if it flags all out-of-domain probes while
                   flagging (almost) no in-domain query.
Output: results/e6_ood_threshold.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from embed import encode
from features import load

ROOT = Path(__file__).resolve().parents[1]
OOD = ["car tyre", "laptop", "cotton t-shirt", "cement bag", "steel beam", "plastic bottle", "smartphone",
       "wooden chair", "paint", "petrol", "paneer", "idli", "dhokla", "printer paper", "leather shoes",
       "aluminium can", "shampoo", "detergent", "LED bulb", "bicycle"]


def main():
    df, _, emb, _ = load()
    short = df.name_en.str.replace(",", " ").str.split().str[:3].str.join(" ").str.lower().tolist()
    ind = (encode(short, "query") @ emb.T).max(1)
    ood = (encode(OOD, "query") @ emb.T).max(1)
    rows = [{"threshold": t, "in_domain_flagged_%": 100 * (ind < t).mean(),
             "ood_caught": int((ood < t).sum()), "ood_total": len(OOD)}
            for t in [0.83, 0.84, 0.845, 0.85, 0.855, 0.86, 0.87, 0.88]]
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "e6_ood_threshold.csv", index=False)
    print(out.to_string(index=False))
    print("in-domain 1st/5th percentile:", np.percentile(ind, [1, 5]).round(3),
          "| max out-of-domain:", ood.max().round(3), OOD[int(ood.argmax())])


if __name__ == "__main__":
    main()
