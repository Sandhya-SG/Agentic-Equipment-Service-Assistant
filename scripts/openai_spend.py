"""Show or change the ledger of estimated OpenAI spend that backs OPENAI_SPEND_CAP_USD.

    python scripts/openai_spend.py                 # show the estimated spend and the cap
    python scripts/openai_spend.py --add 0.30      # count 30 cents that were spent before the ledger existed
    python scripts/openai_spend.py --reset         # start again from zero

The ledger only counts calls made through the application's shared client (the app and the evaluation scripts). It is
a best-effort estimate (tokens times a price table), not a guaranteed financial limit and not your OpenAI balance:
check the usage page of the OpenAI console for the real figure, and set a limit there too, because DeepEval and
Promptfoo use their own clients.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asa.components import llm  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--add", type=float, metavar="USD", help="add dollars spent before the ledger existed")
    parser.add_argument("--reset", action="store_true", help="set the ledger back to zero")
    args = parser.parse_args(argv)

    if args.reset:
        llm.reset_spend()
    if args.add:
        llm.add_spend(args.add, calls=0)
    ledger = llm.read_ledger()
    cap = llm.spend_cap()
    print(
        f"estimated spend: ${ledger['estimated_usd']:.4f} over {ledger['calls']} calls (ledger: {llm._ledger_path()})"
    )
    if cap is None:
        print("no cap set: set OPENAI_SPEND_CAP_USD to stop calls at an estimated amount")
    else:
        left = cap - ledger["estimated_usd"]
        print(f"cap: ${cap:.2f}, estimated room left: ${max(left, 0):.4f}" + ("  (CAP REACHED)" if left <= 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
