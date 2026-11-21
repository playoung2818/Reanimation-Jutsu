#!/usr/bin/env python3
"""Extract three works from the local Zhang Ailing collection (EPUB with NCX)."""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import zipfile
from pathlib import Path
from urllib.parse import unquote, urldefrag
from xml.etree import ElementTree as ET

WORKS = ("小团圆", "倾城之恋", "金锁记")
NCX = {"n": "http://www.daisy.org/z3986/2005/ncx/"}
CHAPTERS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def member_path(base: str, href: str) -> str:
    return posixpath.normpath(posixpath.join(posixpath.dirname(base), unquote(urldefrag(href)[0])))


def body_paragraphs(data: bytes) -> list[str]:
    """Keep body paragraphs, not headings, navigation, scripts, or images."""
    root = ET.fromstring(data)
    body = next((e for e in root.iter() if local_name(e.tag) == "body"), None)
    if body is None:
        raise ValueError("XHTML document has no body")
    paragraphs = []
    for element in body.iter():
        if local_name(element.tag) == "p" and any(c.startswith("bodytext") for c in element.attrib.get("class", "").split()):
            text = "".join(element.itertext()).strip()
            if text:
                paragraphs.append(text)
    if not paragraphs:
        raise ValueError("No bodytext paragraphs found; this extractor requires the collection's paragraph classes")
    return paragraphs


def extract_collection(epub: Path) -> tuple[dict[str, str], dict]:
    with zipfile.ZipFile(epub) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        opf = next(e.attrib["full-path"] for e in container.iter() if local_name(e.tag) == "rootfile")
        package = ET.fromstring(archive.read(opf))
        items = {e.attrib["id"]: e.attrib for e in package.iter() if local_name(e.tag) == "item"}
        spine_element = next(e for e in package.iter() if local_name(e.tag) == "spine")
        spine = [member_path(opf, items[e.attrib["idref"]]["href"]) for e in spine_element]
        toc_item = items.get(spine_element.attrib.get("toc", ""))
        if toc_item is None:
            toc_item = next(e for e in items.values() if e.get("media-type") == "application/x-dtbncx+xml")
        toc_path = member_path(opf, toc_item["href"])
        toc = ET.fromstring(archive.read(toc_path))
        points = toc.findall(".//n:navPoint", NCX)
        leaves = [point for point in points if not point.findall("n:navPoint", NCX)]

        def label(point):
            return point.find("n:navLabel/n:text", NCX).text.strip()

        def location(point):
            return member_path(toc_path, point.find("n:content", NCX).attrib["src"])

        selections = {}
        for work in WORKS:
            if work == "小团圆":
                groups = [p for p in points if label(p) == work and p.findall("n:navPoint", NCX)]
                if len(groups) != 1:
                    raise ValueError("Expected exactly one 小团圆 section")
                chapters = [p for p in groups[0].findall("n:navPoint", NCX) if label(p) in CHAPTERS]
                if tuple(label(p) for p in chapters) != CHAPTERS:
                    raise ValueError("Expected all twelve 小团圆 chapters in order")
                selections[work] = chapters
            else:
                matches = [p for p in leaves if label(p) == work]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one story titled {work}")
                selections[work] = matches

        texts, records = {}, {}
        for work, selected in selections.items():
            sections, members = [], []
            for point in selected:
                start = spine.index(location(point))
                following = leaves[leaves.index(point) + 1:]
                end = spine.index(location(following[0])) if following else len(spine)
                if end <= start:
                    raise ValueError("Multiple TOC sections share a file; refusing ambiguous extraction")
                for member in spine[start:end]:
                    sections.extend(body_paragraphs(archive.read(member)))
                    members.append(member)
            text = "\n\n".join(sections) + "\n"
            texts[work] = text
            records[work] = {
                "file": f"{work}.txt", "characters": len(text), "paragraphs": len(sections),
                "epub_members": members, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
    manifest = {
        "epub_filename": epub.name,
        "epub_sha256": hashlib.sha256(epub.read_bytes()).hexdigest(),
        "method": "NCX story/chapter boundaries in spine order; bodytext paragraphs only; UTF-8, blank line between paragraphs",
        "works": records,
    }
    return texts, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epub", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("source_data/zhang_ailing"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destinations = [args.out_dir / f"{work}.txt" for work in WORKS] + [args.out_dir / "extraction_manifest.json"]
    if not args.force and any(p.exists() for p in destinations):
        parser.error("Output already exists; use --force only to replace it")
    try:
        texts, manifest = extract_collection(args.epub)
    except (ValueError, KeyError, StopIteration, ET.ParseError, zipfile.BadZipFile) as exc:
        parser.error(f"Unsupported or invalid collection: {exc}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for work, text in texts.items():
        path = args.out_dir / f"{work}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"{path}: {len(text)} characters")
    destinations[-1].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
