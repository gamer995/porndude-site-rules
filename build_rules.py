import argparse
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlparse


def normalize_host(host):
    host = host.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def extract_host(line):
    line = line.strip()
    if not line:
        return None
    if "://" not in line:
        line = "https://" + line
    parsed = urlparse(line)
    host = normalize_host(parsed.hostname or "")
    if not host:
        return None
    try:
        ipaddress.ip_address(host)
        return None
    except ValueError:
        return host


def write_outputs(hosts, output_sites, output_rule):
    output_sites.write_text("\n".join(hosts) + "\n", encoding="utf-8")

    payload = ["payload:"]
    for host in hosts:
        payload.append(f"  - +.{host}")
    output_rule.parent.mkdir(parents=True, exist_ok=True)
    output_rule.write_text("\n".join(payload) + "\n", encoding="utf-8")


def load_weights(state_file):
    if not state_file:
        return {}
    state_path = Path(state_file)
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text("utf-8"))
    except Exception:
        return {}

    counts = state.get("site_link_counts") or {}
    if not counts:
        counts = {link: 1 for link in state.get("site_links_raw", [])}
    resolved = state.get("resolved_redirects", {})
    allowed = state.get("allowed_domain")

    weights = {}
    for link, count in counts.items():
        final = resolved.get(link, link)
        host = extract_host(final)
        if not host:
            continue
        if allowed and (host == allowed or host.endswith("." + allowed)):
            continue
        weights[host] = weights.get(host, 0) + count
    return weights


def load_allowed(state_file):
    if not state_file:
        return None
    state_path = Path(state_file)
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text("utf-8"))
    except Exception:
        return None
    allowed = state.get("allowed_domain")
    return allowed.lower().strip() if allowed else None


def main():
    parser = argparse.ArgumentParser(description="Build rule-providers from site list")
    parser.add_argument("--input", default="sites_raw.txt")
    parser.add_argument("--output-sites", default="sites.txt")
    parser.add_argument("--output-rule", default="rule-providers/sites.yaml")
    parser.add_argument("--output-weighted", default=None)
    parser.add_argument("--max-sites", type=int, default=2000)
    parser.add_argument("--state-file", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_sites = Path(args.output_sites)
    output_rule = Path(args.output_rule)

    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    hosts = set()
    for line in input_path.read_text("utf-8").splitlines():
        host = extract_host(line)
        if host:
            hosts.add(host)

    weights = load_weights(args.state_file)
    allowed = load_allowed(args.state_file)
    if allowed:
        hosts.add(allowed)
        weights[allowed] = max(weights.values(), default=0)
    ordered = sorted(hosts, key=lambda h: (-weights.get(h, 0), h))
    if args.max_sites and len(ordered) > args.max_sites:
        ordered = ordered[:args.max_sites]
        if allowed and allowed not in ordered:
            ordered[-1] = allowed

    write_outputs(ordered, output_sites, output_rule)
    if args.output_weighted:
        weighted_lines = [f"{host}\\t{weights.get(host, 0)}" for host in ordered]
        Path(args.output_weighted).write_text(
            "\n".join(weighted_lines) + "\n", encoding="utf-8"
        )
    print(f"hosts: {len(ordered)}")
    print(f"wrote: {output_sites}")
    print(f"wrote: {output_rule}")
    if args.output_weighted:
        print(f"wrote: {args.output_weighted}")


if __name__ == "__main__":
    main()
