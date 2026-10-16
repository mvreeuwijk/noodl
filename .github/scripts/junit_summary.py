"""Write a pass/fail/skip table and the skip reasons of a pytest junit XML to stdout (Markdown).

Usage: python .github/scripts/junit_summary.py report.xml [title]
Append the output to $GITHUB_STEP_SUMMARY so skipped tests and their reasons are visible.
"""

import sys
import xml.etree.ElementTree as ET
from collections import Counter


def main(path, title="Test summary"):
    counts = Counter()
    skips = Counter()
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"### {title}\n\nNo junit report ({exc}).")
        return
    for case in root.iter("testcase"):
        if case.find("skipped") is not None:
            counts["skipped"] += 1
            reason = case.find("skipped").get("message", "").strip() or "(no reason)"
            skips[reason] += 1
        elif case.find("failure") is not None or case.find("error") is not None:
            counts["failed"] += 1
        else:
            counts["passed"] += 1
    print(f"### {title}\n")
    print("| passed | failed | skipped |\n| --- | --- | --- |")
    print(f"| {counts['passed']} | {counts['failed']} | {counts['skipped']} |\n")
    if skips:
        print("<details><summary>Skip reasons</summary>\n")
        print("| tests | reason |\n| --- | --- |")
        for reason, n in skips.most_common():
            print(f"| {n} | {reason.replace('|', '/')[:300]} |")
        print("\n</details>")


if __name__ == "__main__":
    main(*sys.argv[1:3])
