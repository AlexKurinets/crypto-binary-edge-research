"""edgelab: seed, research, paper-trade and report."""

from __future__ import annotations

import argparse
import json
import sys

from edgelab import db
from edgelab.costs import breakeven_win_rate, taker_fee
from edgelab.stats import min_detectable_edge, outcomes_needed


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="edgelab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="create the schemas and tables")
    s = sub.add_parser("seed", help="load the synthetic market")
    s.add_argument("--if-empty", action="store_true")
    s.add_argument("--research-days", type=int, default=112)
    s.add_argument("--paper-days", type=int, default=42)
    s.add_argument("--seed", type=int, default=20260518)
    rs = sub.add_parser("research", help="walk-forward test of every registered hypothesis, then freeze the survivor")
    rs.add_argument("--if-none", action="store_true", help="skip if a research run already exists")
    p = sub.add_parser("paper", help="paper-trade the frozen rule through the paper period")
    p.add_argument("--run", type=int, default=None)
    p.add_argument("--if-none", action="store_true", help="skip if the run already has a paper result")
    r = sub.add_parser("report", help="write the findings as Markdown")
    r.add_argument("--run", type=int, default=None)
    r.add_argument("--out", default=None)
    sub.add_parser("status", help="dataset and latest run")
    pw = sub.add_parser("power", help="how many independent outcomes an edge needs, and what fees cost")
    pw.add_argument("--edge", type=float, default=0.02, help="per-contract edge in dollars")
    pw.add_argument("--price", type=float, default=0.50, help="contract price in dollars")
    pw.add_argument("--contracts", type=int, default=25)
    args = ap.parse_args(argv)

    if args.cmd == "power":
        n = outcomes_needed(args.edge)
        _print(
            {
                "edge_per_contract": args.edge,
                "independent_outcomes_needed": n,
                "fifteen_minute_windows_per_day": 96,
                "days_of_one_trade_per_window": round(n / 96, 1),
                "min_detectable_edge_after_30_days": round(float(min_detectable_edge(30 * 96)), 4),
                "fee_per_contract_at_price": float(taker_fee(args.price, args.contracts) / args.contracts),
                "breakeven_win_rate_at_ask": round(breakeven_win_rate(args.price, args.contracts), 4),
            }
        )
        return 0

    with db.connect() as conn:
        if args.cmd == "migrate":
            _print({"applied": db.migrate(conn)})
        elif args.cmd == "seed":
            from edgelab.seed import is_empty, seed

            if args.if_empty and not is_empty(conn):
                _print({"seeded": False, "reason": "market.bars already has data"})
                return 0
            _print({"seeded": True, **seed(conn, args.research_days, args.paper_days, args.seed)})
        elif args.cmd == "research":
            from edgelab.research import run_research

            existing = conn.execute("select max(run_id) from research.runs").fetchone()[0]
            if args.if_none and existing is not None:
                _print({"skipped": True, "reason": f"research run {existing} already exists"})
                return 0
            _print(run_research(conn))
        elif args.cmd == "paper":
            from edgelab.paper import run_paper

            row = conn.execute("select run_id, summary ? 'paper' from research.runs order by run_id desc limit 1").fetchone()
            if args.if_none and row is not None and row[1]:
                _print({"skipped": True, "reason": f"research run {row[0]} already has a paper result"})
                return 0
            _print(run_paper(conn, args.run))
        elif args.cmd == "report":
            from edgelab.report import render

            text = render(conn, args.run)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as fh:
                    fh.write(text)
                _print({"written": args.out, "chars": len(text)})
            else:
                sys.stdout.write(text)
        elif args.cmd == "status":
            info = db.meta(conn)
            row = conn.execute("select run_id, created_at, summary from research.runs order by run_id desc limit 1").fetchone()
            _print({"dataset": info, "latest_run": None if row is None else {"run_id": row[0], "created_at": row[1], "summary": row[2]}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
