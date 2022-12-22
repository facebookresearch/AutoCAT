import argparse
import json
import re

from tabulate import tabulate
from typing import Any, Dict, Optional, Union

from rlmeta.utils.stats_dict import StatsItem, StatsDict

JSON_REGEX = re.compile("{.+}")


def parse_json(line: str) -> Optional[Dict[str, Any]]:
    m = JSON_REGEX.search(line)
    return None if m is None else json.loads(m.group())


def show_table(stats: Dict[str, Any], info: Optional[str] = None) -> tabulate:
    if info is None:
        head = ["key", "mean", "std", "min", "max", "count"]
    else:
        head = ["info", "key", "mean", "std", "min", "max", "count"]

    data = []
    for k, v in stats.items():
        if isinstance(v, dict):
            row = [k, v["mean"], v["std"], v["min"], v["max"], v["count"]]
        else:
            row = [k, v, 0.0, v, v, 1]
        if info is not None:
            row = [info] + row
        data.append(row)

    return tabulate(data,
                    head,
                    numalign="right",
                    stralign="right",
                    floatfmt=".8f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="log file to plot")

    flags = parser.parse_intermixed_args()
    with open(flags.log_file, "r") as f:
        line = f.readline()
        exp_cfg = parse_json(line)
        print(f"Experiment Configs = {exp_cfg}")

        for line in f:
            stats = parse_json(line)
            info = stats.pop("info")
            stats.pop("phase")
            stats.pop("epoch")
            print("\n" + show_table(stats, info) + "\n")


if __name__ == "__main__":
    main()
