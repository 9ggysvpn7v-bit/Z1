#!/usr/bin/env python3
"""Собирает blockerList.json для Safari Content Blocker.

Источники:
  rules/domains.txt   — домены для блокировки (вместе с поддоменами)
  rules/selectors.txt — CSS-селекторы для скрытия рекламных блоков
  --hosts-url URL     — (опционально) внешний hosts-список, например
                        https://adaway.org/hosts.txt

Safari принимает не более 150 000 правил в одном расширении.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_RULES = 150_000
SELECTORS_PER_RULE = 50
DOMAIN_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9-]{1,63}\.)+[a-z]{2,63}$")


def read_lines(path: Path):
    """Строки файла без пустых и комментариев ("# ..."). "#id" — это селектор."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and line != "#" and not line.startswith("# "):
            yield line


def parse_hosts(text: str):
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("0.0.0.0", "127.0.0.1"):
            yield parts[1]
        elif len(parts) == 1:
            yield parts[0]


def normalize(domains):
    clean = {d.lower().strip(".") for d in domains}
    clean = {d for d in clean if DOMAIN_RE.match(d)}
    # Поддомен не нужен, если уже блокируется родительский домен.
    def covered(d):
        labels = d.split(".")
        return any(".".join(labels[i:]) in clean for i in range(1, len(labels) - 1))
    return sorted(d for d in clean if not covered(d))


def domain_rule(domain: str):
    escaped = re.escape(domain).replace("\\-", "-")
    return {
        "trigger": {
            "url-filter": rf"^[^:]+://([^/:]+\.)?{escaped}[/:]",
            "url-filter-is-case-sensitive": False,
        },
        "action": {"type": "block"},
    }


def selector_rules(selectors):
    for i in range(0, len(selectors), SELECTORS_PER_RULE):
        chunk = selectors[i:i + SELECTORS_PER_RULE]
        yield {
            "trigger": {"url-filter": ".*"},
            "action": {"type": "css-display-none", "selector": ", ".join(chunk)},
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts-url", action="append", default=[])
    ap.add_argument("--out", default=str(ROOT / "ZeroAdsBlocker" / "blockerList.json"))
    args = ap.parse_args()

    domains = list(read_lines(ROOT / "rules" / "domains.txt"))
    for url in args.hosts_url:
        print(f"Загружаю {url} ...", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=60) as resp:
            domains += list(parse_hosts(resp.read().decode("utf-8", "replace")))

    selectors = list(read_lines(ROOT / "rules" / "selectors.txt"))
    domains = normalize(domains)

    rules = list(selector_rules(selectors))
    budget = MAX_RULES - len(rules)
    if len(domains) > budget:
        print(f"Внимание: {len(domains)} доменов, обрезаю до {budget}", file=sys.stderr)
        domains = domains[:budget]
    rules = [domain_rule(d) for d in domains] + rules

    Path(args.out).write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {len(rules)} правил ({len(domains)} доменов, {len(selectors)} селекторов) -> {args.out}")


if __name__ == "__main__":
    main()
