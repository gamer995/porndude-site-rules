import argparse
import ipaddress
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


def main():
    parser = argparse.ArgumentParser(description="Build rule-providers from site list")
    parser.add_argument("--input", default="sites_raw.txt")
    parser.add_argument("--output-sites", default="sites.txt")
    parser.add_argument("--output-rule", default="rule-providers/sites.yaml")
    parser.add_argument("--max-sites", type=int, default=2000)
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

    ordered = sorted(hosts)
    if args.max_sites and len(ordered) > args.max_sites:
        ordered = ordered[:args.max_sites]

    write_outputs(ordered, output_sites, output_rule)
    print(f"hosts: {len(ordered)}")
    print(f"wrote: {output_sites}")
    print(f"wrote: {output_rule}")


if __name__ == "__main__":
    main()
