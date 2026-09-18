#!/usr/bin/env python3
"""Build the public SILT landing site for Vercel.

Emits dist-vercel/ with:
  - index.html            (from docs/index.html)
  - studio/index.html     (from src/asea/studio/static/index.html + bridge injection)
  - web/studio-bridge.js
  - logo.svg, favicon.svg
  - sitemap.xml
  - robots.txt
  - PATENT.md, README.md

Vercel deploys docs/ directly (vercel.json outputDirectory: "docs",
no build command), so the extras that must be reachable in production
are ALSO synced into docs/:
  - docs/sitemap.xml, docs/robots.txt, docs/favicon.svg (generated/copied here)
  - docs/PATENT.md, docs/README.md      (copied from the repo root,
                                         which stays the single source)

dist-vercel/ is a LOCAL PREVIEW bundle only -- it is never deployed.
Note the deliberate deployment split: docs/studio/index.html (the
deployed /studio/ page) is a hand-authored "Run locally" launcher;
the full studio UI built into dist-vercel/studio/ with the bridge
injection ships only to the local engine at 127.0.0.1:8377.

The script is intentionally hermetic: no network, no secrets, no build-time
parameter injection. Canonical host is https://silt.inbharat.ai.
Run it whenever PATENT.md, README.md or the route set changes so the
docs/ copies stay in sync.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "dist-vercel"
DOCS = REPO / "docs"
CANONICAL = "https://silt.inbharat.ai"
GITHUB = "https://github.com/inbharatai/SILT/blob/main"
GITHUB_RAW = "https://raw.githubusercontent.com/inbharatai/SILT/main"

# Markdown link/image targets that are never repo-relative
_EXTERNAL = ("http://", "https://", "//", "#", "mailto:", "data:")

_MD_LINK = re.compile(r"\]\(([^)\s]+)(#[^)\s]*)?\)")
_MD_IMG = re.compile(r'src="(docs/[^"]+)"')


def rewrite_readme_for_serving(text: str) -> str:
    """Adapt root-README links for the raw-served docs/ copy.

    docs/ IS the deployment root, so a link like ``docs/CAPABILITIES.md``
    would 404 at /docs/CAPABILITIES.md, and repo-root targets like
    ``src/...`` or ``CHANGELOG.md`` do not exist under docs/ at all.
    Targets under docs/ become site-relative (they serve raw at /name);
    everything else becomes an absolute GitHub URL, which works everywhere.
    The root README.md itself is never modified.
    """
    def link(m: "re.Match[str]") -> str:
        target, anchor = m.group(1), m.group(2) or ""
        if target.startswith(_EXTERNAL):
            return m.group(0)
        if target.startswith("docs/"):
            rest = target[len("docs/"):]
            if (DOCS / rest).exists():
                return f"](/{rest}{anchor})"
        return f"]({GITHUB}/{target}{anchor})"

    def img(m: "re.Match[str]") -> str:
        return f'src="{GITHUB_RAW}/{m.group(1)}"'

    return _MD_IMG.sub(img, _MD_LINK.sub(link, text))


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def inject_bridge(html: str) -> str:
    """Inject the hosted bridge script before </head> if not already present."""
    marker = '<script src="/web/studio-bridge.js" defer></script>'
    if marker in html:
        return html
    if "</head>" in html:
        return html.replace("</head>", f"{marker}\n</head>")
    return html


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Landing page
    landing = REPO / "docs" / "index.html"
    if not landing.exists():
        print(f"ERROR: missing {landing}", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(landing, OUT / "index.html")

    # Studio page (local static UI + hosted bridge injection)
    studio_src = REPO / "src" / "asea" / "studio" / "static" / "index.html"
    if not studio_src.exists():
        print(f"ERROR: missing {studio_src}", file=sys.stderr)
        sys.exit(1)
    studio_out = OUT / "studio" / "index.html"
    studio_out.parent.mkdir(parents=True, exist_ok=True)
    studio_html = studio_src.read_text(encoding="utf-8")
    studio_out.write_text(inject_bridge(studio_html), encoding="utf-8")

    # Bridge script
    bridge = REPO / "web" / "studio-bridge.js"
    if not bridge.exists():
        print(f"ERROR: missing {bridge}", file=sys.stderr)
        sys.exit(1)
    bridge_out = OUT / "web" / "studio-bridge.js"
    bridge_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bridge, bridge_out)

    # Assets
    static_dir = REPO / "src" / "asea" / "studio" / "static"
    for asset in ("logo.svg", "favicon.svg"):
        src = static_dir / asset
        if src.exists():
            shutil.copy2(src, OUT / asset)

    # Static docs (README links are adapted for raw-served HTML contexts;
    # PATENT.md has no repo-relative links and copies verbatim)
    for name in ("PATENT.md", "README.md"):
        src = REPO / name
        if src.exists():
            content = src.read_text(encoding="utf-8")
            if name == "README.md":
                content = rewrite_readme_for_serving(content)
            write(OUT / name, content)

    # Sitemap + robots (shared by dist-vercel/ and docs/; /studio is the
    # canonical clean URL -- trailingSlash:false 308s /studio/)
    sitemap_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>{CANONICAL}/</loc>
    <lastmod>2026-09-18</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>{CANONICAL}/studio</loc>
    <lastmod>2026-09-18</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>
'''
    robots_txt = f"""User-agent: *
Allow: /
Sitemap: {CANONICAL}/sitemap.xml
"""
    write(OUT / "sitemap.xml", sitemap_xml)
    write(OUT / "robots.txt", robots_txt)

    # Sync the deployable extras into docs/ (the directory Vercel serves).
    # vercel.json has no build command, so anything outside docs/ never
    # reaches production: sitemap.xml and robots.txt were 404 until this
    # sync existed. Root PATENT.md/README.md stay the source of truth.
    DOCS = REPO / "docs"
    write(DOCS / "sitemap.xml", sitemap_xml)
    write(DOCS / "robots.txt", robots_txt)
    shutil.copy2(REPO / "src" / "asea" / "studio" / "static" / "favicon.svg", DOCS / "favicon.svg")
    patent_src = REPO / "PATENT.md"
    if patent_src.exists():
        shutil.copy2(patent_src, DOCS / "PATENT.md")
    readme_src = REPO / "README.md"
    if readme_src.exists():
        # The served copy gets site-relative/GitHub-absolute links; the root
        # README.md stays the canonical GitHub version.
        write(DOCS / "README.md", rewrite_readme_for_serving(readme_src.read_text(encoding="utf-8")))

    print(f"Built SILT public site at {OUT}")
    print(f"Synced deployable extras into {DOCS} (sitemap.xml, robots.txt, favicon.svg, PATENT.md, README.md)")


if __name__ == "__main__":
    build()
