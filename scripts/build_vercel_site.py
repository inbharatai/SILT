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
  - docs/sitemap.xml, docs/robots.txt   (generated here)
  - docs/PATENT.md, docs/README.md      (copied from the repo root,
                                         which stays the single source)

The script is intentionally hermetic: no network, no secrets, no build-time
parameter injection. Canonical host is https://silt.inbharat.ai.
Run it whenever PATENT.md, README.md or the route set changes so the
docs/ copies stay in sync.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "dist-vercel"
CANONICAL = "https://silt.inbharat.ai"


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

    # Static docs
    for name in ("PATENT.md", "README.md"):
        src = REPO / name
        if src.exists():
            shutil.copy2(src, OUT / name)

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
    for name in ("PATENT.md", "README.md"):
        src = REPO / name
        if src.exists():
            shutil.copy2(src, DOCS / name)

    print(f"Built SILT public site at {OUT}")
    print(f"Synced deployable extras into {DOCS} (sitemap.xml, robots.txt, PATENT.md, README.md)")


if __name__ == "__main__":
    build()
