import argparse
import json
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
import itertools
import time
import urllib.robotparser

START_URL = "https://theporndude.com/zh"
ALLOWED_DOMAIN = "theporndude.com"   # 只在这个域名内爬
MAX_PAGES = 800
DEFAULT_DELAY = 0.5
DEFAULT_STATE_FILE = "crawl_state.json"
DEFAULT_LOG_FILE = "crawl.log"
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_SITE_OUTPUT = "sites.txt"
REDIRECT_HOSTS = {"pdude.link"}
SITE_LINK_CLASSES = {"link-analytics", "icon-site"}
SITE_LINK_PARENT_CLASS = "category-item"
REDIRECT_TIMEOUT = 5
REDIRECT_WORKERS = 8
DEFAULT_RESOLVE_LIMIT = 200
DEFAULT_MAX_PATH_DEPTH = 2

# 更细的 URL 去重策略：设置保留或忽略的 query 参数
# KEEP_QUERY_KEYS 非空时，仅保留这些参数；为空时忽略 DROP_QUERY_KEYS 和 utm_*。
KEEP_QUERY_KEYS = set()
DROP_QUERY_KEYS = {
    "fbclid", "gclid", "yclid", "mc_cid", "mc_eid",
    "_hsenc", "_hsmi", "spm", "ref"
}

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; DomainCrawler/1.0)"
}
USER_AGENT = headers["User-Agent"]

def log_line(msg, log_fp=None):
    print(msg)
    if log_fp:
        log_fp.write(msg + "\n")
        log_fp.flush()

def get_html(url, session=None, log_fp=None):
    try:
        client = session if session is not None else requests
        resp = client.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        log_line(f"[!] 请求出错: {url} -> {e}", log_fp)
        return None

def normalize_host(host):
    host = host.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host

def build_netloc(host, port, scheme):
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        return f"{host}:{port}"
    return host

def normalize_query_keys(keys):
    return {k.lower() for k in keys} if keys else set()

def canonicalize_url(url, keep_query_keys=None, drop_query_keys=None):
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            return None
        if not parsed.hostname:
            return None

        host = normalize_host(parsed.hostname)
        netloc = build_netloc(host, parsed.port, scheme)
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        query = ""
        if parsed.query:
            keep_keys = normalize_query_keys(keep_query_keys)
            drop_keys = normalize_query_keys(drop_query_keys)
            params = []
            for k, v in parse_qsl(parsed.query, keep_blank_values=False):
                key = k.lower()
                if keep_keys:
                    if key not in keep_keys:
                        continue
                else:
                    if key in drop_keys or key.startswith("utm_"):
                        continue
                params.append((k, v))
            if params:
                params.sort()
                query = urlencode(params, doseq=True)

        cleaned = parsed._replace(
            scheme=scheme,
            netloc=netloc,
            path=path,
            query=query,
            fragment=""
        )
        return cleaned.geturl()
    except Exception:
        return None

def normalize_domain(url):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None
        if not parsed.hostname:
            return None
        host = normalize_host(parsed.hostname)
        return host if host else None
    except Exception:
        return None

def is_site_link(tag):
    classes = set(tag.get("class") or [])
    if SITE_LINK_CLASSES.issubset(classes):
        return True
    parent = tag.parent
    while parent is not None and getattr(parent, "name", None):
        if parent.name == "li" and SITE_LINK_PARENT_CLASS in (parent.get("class") or []):
            return True
        parent = parent.parent
    return False

def normalize_site_url(url):
    clean = canonicalize_url(url, KEEP_QUERY_KEYS, DROP_QUERY_KEYS)
    if not clean:
        return None
    parsed = urlparse(clean)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"

def resolve_redirect(url, session, log_fp=None):
    try:
        resp = session.head(url, headers=headers, allow_redirects=False, timeout=REDIRECT_TIMEOUT)
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            return urljoin(url, location) if location else url
        if resp.status_code == 405:
            resp = session.get(url, headers=headers, allow_redirects=False, timeout=REDIRECT_TIMEOUT, stream=True)
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                return urljoin(url, location) if location else url
        return url
    except Exception as e:
        log_line(f"[!] 解析跳转失败: {url} -> {e}", log_fp)
        return url

def resolve_redirects_batch(urls, log_fp=None):
    results = {}
    if not urls:
        return results
    with ThreadPoolExecutor(max_workers=REDIRECT_WORKERS) as executor:
        future_map = {
            executor.submit(resolve_redirect, url, requests, log_fp): url
            for url in urls
        }
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                results[url] = future.result()
            except Exception as e:
                log_line(f"[!] 批量解析失败: {url} -> {e}", log_fp)
                results[url] = url
    return results

def is_allowed_domain(domain, allowed_domain):
    return domain == allowed_domain or domain.endswith("." + allowed_domain)

def should_follow_path(path, max_depth):
    if max_depth is None:
        return True
    segments = [s for s in path.split("/") if s]
    if len(segments) > max_depth:
        return False
    if len(segments) >= 2 and segments[1].isdigit():
        return False
    return True

def get_robot_parser(base_url, user_agent, robots_cache, session, log_fp=None):
    if base_url in robots_cache:
        return robots_cache[base_url]

    robots_url = urljoin(base_url, "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        resp = session.get(robots_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            rp.parse([])
    except Exception as e:
        log_line(f"[!] robots.txt 请求失败: {robots_url} -> {e}", log_fp)
        rp.parse([])

    robots_cache[base_url] = rp
    return rp

def can_fetch(url, user_agent, robots_cache, session, log_fp=None):
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{build_netloc(normalize_host(parsed.hostname), parsed.port, parsed.scheme)}"
    rp = get_robot_parser(base_url, user_agent, robots_cache, session, log_fp=log_fp)
    return rp.can_fetch(user_agent, url)

def get_crawl_delay(url, user_agent, robots_cache, session, default_delay, log_fp=None):
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{build_netloc(normalize_host(parsed.hostname), parsed.port, parsed.scheme)}"
    rp = get_robot_parser(base_url, user_agent, robots_cache, session, log_fp=log_fp)
    delay = rp.crawl_delay(user_agent)
    if delay is None:
        delay = rp.crawl_delay("*")
    return max(default_delay, delay or 0)

def throttle(url, last_request_time, delay):
    parsed = urlparse(url)
    host = normalize_host(parsed.hostname)
    last = last_request_time.get(host)
    now = time.monotonic()
    if last is not None:
        sleep_for = delay - (now - last)
        if sleep_for > 0:
            time.sleep(sleep_for)
    last_request_time[host] = time.monotonic()

def load_state(state_file, log_fp=None):
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log_line(f"[!] 状态文件不存在: {state_file}", log_fp)
    except Exception as e:
        log_line(f"[!] 读取状态失败: {state_file} -> {e}", log_fp)
    return None

def save_state(state_file, state, log_fp=None):
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=True, separators=(",", ":"))
    except Exception as e:
        log_line(f"[!] 写入状态失败: {state_file} -> {e}", log_fp)

def crawl_domains(
    start_url,
    allowed_domain,
    max_pages=500,
    default_delay=0.5,
    state_file=None,
    checkpoint_every=50,
    resume_state=None,
    log_fp=None,
    resolve_redirects=False,
    resolve_limit=0,
    resolve_only=False,
    max_path_depth=DEFAULT_MAX_PATH_DEPTH,
    priority=False
):
    visited_pages = set()
    queued_pages = set()
    site_urls = set()
    site_links_raw = set()
    site_link_counts = {}
    resolved_redirects = {}
    url_scores = {}
    priority_queue = []
    order_counter = itertools.count()
    robots_cache = {}
    last_request_time = {}
    session = requests.Session()

    allowed_domain = normalize_host(allowed_domain)
    q = deque()

    if resume_state:
        state_allowed = resume_state.get("allowed_domain")
        state_start = resume_state.get("start_url")
        if state_allowed and normalize_host(state_allowed) != allowed_domain:
            log_line(
                f"[!] 发现状态文件中的域名 {state_allowed}，将覆盖当前 allowed_domain",
                log_fp
            )
            allowed_domain = normalize_host(state_allowed)
        if state_start:
            start_url = state_start
        visited_pages = set(resume_state.get("visited_pages", []))
        queued_pages = set(resume_state.get("queued_pages", []))
        site_urls = set(resume_state.get("site_urls", []))
        site_links_raw = set(resume_state.get("site_links_raw", []))
        site_link_counts = dict(resume_state.get("site_link_counts", {}))
        resolved_redirects = dict(resume_state.get("resolved_redirects", {}))
        url_scores = dict(resume_state.get("url_scores", {}))
        priority_queue = [
            tuple(item) for item in resume_state.get("priority_queue", [])
        ]
        if priority_queue:
            heapq.heapify(priority_queue)
            max_order = max(item[1] for item in priority_queue)
            order_counter = itertools.count(start=max_order + 1)
        q = deque(resume_state.get("queue", []))
        log_line(
            f"[+] 恢复状态: visited={len(visited_pages)}, queue={len(q)}, sites={len(site_urls)}",
            log_fp
        )
        if "site_links_raw" not in resume_state and site_urls:
            log_line("[!] 状态文件缺少 site_links_raw，无法重新解析跳转；将仅继续追加新链接", log_fp)
        state_priority = resume_state.get("priority")
        if state_priority is not None and state_priority != priority:
            log_line("[!] 状态文件的 priority 与当前参数不一致，已按状态文件设置", log_fp)
            priority = state_priority
        if not site_link_counts and site_links_raw:
            site_link_counts = {link: 1 for link in site_links_raw}

    if priority and not priority_queue and q:
        for url in list(q):
            if url in visited_pages:
                continue
            url_scores[url] = max(url_scores.get(url, 0), 1)
            heapq.heappush(priority_queue, (-url_scores[url], next(order_counter), url))
        q = deque()

    if not q and not resolve_only:
        start_url = canonicalize_url(start_url, KEEP_QUERY_KEYS, DROP_QUERY_KEYS)
        if not start_url:
            log_line("[!] 起始 URL 非法", log_fp)
            return site_urls
        if priority:
            url_scores[start_url] = url_scores.get(start_url, 0) + 1
            heapq.heappush(priority_queue, (-url_scores[start_url], next(order_counter), start_url))
        else:
            q.append(start_url)
            queued_pages.add(start_url)

    if not resolve_only:
        def pop_next():
            while priority_queue:
                score, _, url = heapq.heappop(priority_queue)
                score = -score
                if url in visited_pages:
                    continue
                if url_scores.get(url, 0) != score:
                    continue
                return url
            return None

        while (priority_queue if priority else q) and len(visited_pages) < max_pages:
            url = pop_next() if priority else q.popleft()
            if not url:
                break
            if url in visited_pages:
                continue
            visited_pages.add(url)
            log_line(f"[+] 抓取页面: {url}", log_fp)

            if not can_fetch(url, USER_AGENT, robots_cache, session, log_fp=log_fp):
                log_line(f"[!] robots.txt 拒绝抓取: {url}", log_fp)
                continue

            delay = get_crawl_delay(url, USER_AGENT, robots_cache, session, default_delay, log_fp=log_fp)
            throttle(url, last_request_time, delay)

            html = get_html(url, session=session, log_fp=log_fp)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")

            # 提取所有链接
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()

                # 拼绝对 URL
                abs_url = urljoin(url, href)
                clean_url = canonicalize_url(abs_url, KEEP_QUERY_KEYS, DROP_QUERY_KEYS)
                if not clean_url:
                    continue
                parsed = urlparse(clean_url)

                # 收集页面中“站点介绍”链接
                if is_site_link(a):
                    site_link = canonicalize_url(abs_url, KEEP_QUERY_KEYS, DROP_QUERY_KEYS)
                    if site_link:
                        site_links_raw.add(site_link)
                        site_link_counts[site_link] = site_link_counts.get(site_link, 0) + 1

                # 只在允许域名的站内继续爬，避免子串误命中
                d = normalize_domain(clean_url)
                if d and is_allowed_domain(d, allowed_domain):
                    # 限制协议，只跟 http/https
                    if parsed.scheme in ("http", "https"):
                        if not should_follow_path(parsed.path, max_path_depth):
                            continue
                        if priority:
                            if clean_url in visited_pages:
                                continue
                            url_scores[clean_url] = url_scores.get(clean_url, 0) + 1
                            heapq.heappush(
                                priority_queue,
                                (-url_scores[clean_url], next(order_counter), clean_url)
                            )
                        else:
                            if clean_url not in visited_pages and clean_url not in queued_pages:
                                q.append(clean_url)
                                queued_pages.add(clean_url)

            if state_file and checkpoint_every > 0 and len(visited_pages) % checkpoint_every == 0:
                state = {
                    "start_url": start_url,
                    "allowed_domain": allowed_domain,
                    "max_pages": max_pages,
                    "visited_pages": list(visited_pages),
                "queued_pages": list(queued_pages),
                "queue": list(q),
                "site_urls": list(site_urls),
                "site_links_raw": list(site_links_raw),
                "site_link_counts": site_link_counts,
                "resolved_redirects": resolved_redirects,
                "priority": priority,
                "priority_queue": list(priority_queue),
                "url_scores": url_scores
            }
            save_state(state_file, state, log_fp=log_fp)

    if state_file:
        state = {
            "start_url": start_url,
            "allowed_domain": allowed_domain,
            "max_pages": max_pages,
            "visited_pages": list(visited_pages),
            "queued_pages": list(queued_pages),
            "queue": list(q),
            "site_urls": list(site_urls),
            "site_links_raw": list(site_links_raw),
            "site_link_counts": site_link_counts,
            "resolved_redirects": resolved_redirects,
            "priority": priority,
            "priority_queue": list(priority_queue),
            "url_scores": url_scores
        }
        save_state(state_file, state, log_fp=log_fp)

    resolved_sites = set(site_urls)
    host_weights = {}
    if site_link_counts:
        redirect_links = []
        for link, count in site_link_counts.items():
            host = urlparse(link).netloc.lower()
            if host in REDIRECT_HOSTS:
                redirect_links.append(link)
            else:
                normalized_site = normalize_site_url(link)
                if normalized_site:
                    site_host = normalize_domain(normalized_site)
                    if site_host and not is_allowed_domain(site_host, allowed_domain):
                        host_weights[site_host] = host_weights.get(site_host, 0) + count

        if resolve_redirects and redirect_links:
            to_resolve = [link for link in redirect_links if link not in resolved_redirects]
            if resolve_limit and len(to_resolve) > resolve_limit:
                to_resolve = to_resolve[:resolve_limit]
            if to_resolve:
                resolved_map = resolve_redirects_batch(to_resolve, log_fp=log_fp)
                resolved_redirects.update(resolved_map)

        unresolved = 0
        for link in redirect_links:
            final_link = resolved_redirects.get(link)
            if not final_link:
                unresolved += 1
                continue
            normalized_site = normalize_site_url(final_link)
            if normalized_site:
                site_host = normalize_domain(normalized_site)
                if site_host and not is_allowed_domain(site_host, allowed_domain):
                    host_weights[site_host] = host_weights.get(site_host, 0) + site_link_counts.get(link, 1)

        if unresolved:
            log_line(f"[!] 未解析跳转链接数量: {unresolved}", log_fp)

    if state_file:
        state = {
            "start_url": start_url,
            "allowed_domain": allowed_domain,
            "max_pages": max_pages,
            "visited_pages": list(visited_pages),
            "queued_pages": list(queued_pages),
            "queue": list(q),
            "site_urls": list(site_urls),
            "site_links_raw": list(site_links_raw),
            "site_link_counts": site_link_counts,
            "resolved_redirects": resolved_redirects,
            "priority": priority,
            "priority_queue": list(priority_queue),
            "url_scores": url_scores
        }
        save_state(state_file, state, log_fp=log_fp)

    if host_weights:
        resolved_sites.update(host_weights.keys())

    return resolved_sites

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain crawler")
    parser.add_argument("--start-url", default=START_URL)
    parser.add_argument("--allowed-domain", default=ALLOWED_DOMAIN)
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--site-output", default=DEFAULT_SITE_OUTPUT)
    parser.add_argument("--resolve-redirects", action="store_true")
    parser.add_argument("--resolve-limit", type=int, default=DEFAULT_RESOLVE_LIMIT)
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--max-path-depth", type=int, default=DEFAULT_MAX_PATH_DEPTH)
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    allowed_domain = normalize_host(args.allowed_domain)
    log_fp = None
    if args.log_file:
        log_fp = open(args.log_file, "a", encoding="utf-8")

    resume_state = None
    if args.resume and args.state_file and os.path.exists(args.state_file):
        resume_state = load_state(args.state_file, log_fp=log_fp)
    elif args.resolve_only and args.state_file and os.path.exists(args.state_file):
        resume_state = load_state(args.state_file, log_fp=log_fp)
    site_urls = crawl_domains(
        args.start_url,
        allowed_domain,
        max_pages=args.max_pages,
        default_delay=args.delay,
        state_file=args.state_file,
        checkpoint_every=args.checkpoint_every,
        resume_state=resume_state,
        log_fp=log_fp,
        resolve_redirects=args.resolve_redirects,
        resolve_limit=args.resolve_limit,
        resolve_only=args.resolve_only,
        max_path_depth=args.max_path_depth,
        priority=args.priority
    )

    log_line(f"\n[+] 共提取到站点数量: {len(site_urls)}", log_fp)
    # 写入文件
    with open(args.site_output, "w", encoding="utf-8") as f:
        for d in sorted(site_urls):
            f.write(d + "\n")

    log_line(f"[+] 已写入 {args.site_output}", log_fp)

    if log_fp:
        log_fp.close()
