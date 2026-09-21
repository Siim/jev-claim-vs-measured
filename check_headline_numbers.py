#!/usr/bin/env python3
"""Re-run every experiment and check that each published number comes out the same.

    python check_headline_numbers.py        (no API key, no downloads; about 8 minutes)

Every experiment script ends by printing a "HEADLINE NUMBERS" block. This script runs them,
reads those blocks, and compares them with the values published in the README.
"""

import re
import subprocess
import sys

EXPECTED = {
    "01_latency_from_cache.py": {
        "api_calls": 35685,
        "median_latency_ms": 295,
        "share_under_100ms_pct": 0.0,
    },
    "11_intraday_price_all_states.py": {
        "states": 15625,
        "buy_states": 3312,
        "sell_states": 7013,
        "mean_p_up_15m": 0.45,
    },
    "12_intraday_evaluate.py": {
        "decisions": 523361,
        "trades_ret_15m_bp": 298549,
        "hit_rate_pct_ret_15m_bp": 48.37,
        "gross_bp_ret_15m_bp": -0.304,
        "gross_bp_ret_60m_bp": -0.467,
        "auc_ret_15m_bp": 0.4805,
        "null_z_ret_15m_bp": -3.88,
        "fitted_model_gross_bp_ret_15m_bp": 0.176,
        "fitted_model_gross_bp_ret_60m_bp": 0.533,
    },
    "21_tape_price_all_states.py": {
        "states": 15625,
        "buy_states": 827,
        "sell_states": 4665,
    },
    "22_tape_evaluate.py": {
        "decisions": 298787,
        "model_hit_pct_delay1_hold1": 53.4,
        "model_gross_bp_delay1_hold1": 0.04,
        "model_gross_bp_delay1_hold10": 0.077,
        "model_hit_pct_delay0_hold1": 58.73,
        "rule_hit_pct_delay1_hold1": 58.07,
        "oracle_bp_delay1_hold1": 1.01,
        "oracle_bp_delay1_hold10": 3.8,
        "oracle_bp_delay1_hold50": 8.73,
    },
    "31_tape_alts_evaluate.py": {
        "decisions": 925874,
        "coins_evaluated": 8,
        "rule_beats_model_at_1_bar": 7,
        "alts_clearing_4bp": 1,
        "BTC_model_hit_pct": 56.23,
        "AKE_model_gross_10_bp": 0.242,
        "AKE_rule_gross_10_bp": 3.108,
    },
    "40_text_triage.py": {
        "titles": 4435,
        "checkable_titles": 1313,
        "accuracy_pct": 88.2,
        "kept_at_0.8": 1145,
        "accuracy_pct_at_0.8": 99.3,
    },
}

HEADLINE_LINE = re.compile(r"^\s+([\w.]+) = (-?[\d.]+)\s*$")


def headline_numbers(script):
    completed = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"{script} failed:\n{completed.stderr[-1500:]}")
    block = completed.stdout.split("HEADLINE NUMBERS")[-1]
    return {m.group(1): float(m.group(2)) for line in block.splitlines() if (m := HEADLINE_LINE.match(line))}


def main():
    failures = 0
    for script, expected in EXPECTED.items():
        print(f"\n{script}", flush=True)
        measured = headline_numbers(script)
        for name, value in expected.items():
            ok = name in measured and abs(measured[name] - value) < 1e-9
            shown = f"{measured[name]:g}" if name in measured else "missing"
            failures += not ok
            print(
                f"   {'ok  ' if ok else 'FAIL'} {name:<36s} published {value:>10g}" f"   measured {shown:>10}"
            )
    print(
        "\n"
        + ("ALL PUBLISHED NUMBERS REPRODUCED" if failures == 0 else f"{failures} NUMBERS DID NOT REPRODUCE")
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
