#!/usr/bin/env python3
# =============================================================================
#  build_catalog.py — turn raw prompt datasets into a small, browser-ready,
#  structured catalog for the AI Prompt Builder (GitHub Pages, 100% static).
#
#  - Downloads each enabled source from scripts/sources.yml at BUILD time.
#  - Normalizes every prompt into the Prompt Builder field schema
#    (role / task / context / constraints / output_format / tone / examples).
#  - Lightly splits a "Role / Persona" line out of the prompt body and turns a
#    trailing "My first ... is \"...\"" instruction into a {{variable}}.
#  - Categorizes by keyword, de-duplicates, and curates a balanced subset.
#  - Writes data/prompts.catalog.json (pretty), .min.json (browser), .txt.
#
#  Dependencies: Python 3.9+ standard library only. PyYAML is used if present;
#  otherwise an embedded copy of the source config is used. No network access
#  is ever required from the browser — only from this build step.
# =============================================================================

import csv
import io
import json
import re
import sys
import urllib.request
from collections import OrderedDict, defaultdict
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ROOT = Path(__file__).resolve().parent.parent          # the prompt-builder/ folder
DATA = ROOT / "data"
SOURCES_FILE = ROOT / "scripts" / "sources.yml"

# Fallback config used when PyYAML is unavailable or sources.yml is missing.
DEFAULT_CONFIG = {
    "max_prompts": 500,
    "sources": [
        {
            "id": "prompts-chat",
            "name": "fka/awesome-chatgpt-prompts (prompts.chat)",
            "enabled": True,
            "format": "csv",
            "url": "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv",
            "fallback_url": "https://huggingface.co/datasets/fka/awesome-chatgpt-prompts/resolve/main/prompts.csv",
            "source_url": "https://github.com/f/awesome-chatgpt-prompts",
            "license": "CC0-1.0",
            "columns": {"title": "act", "prompt": "prompt", "for_devs": "for_devs",
                        "type": "type", "contributor": "contributor"},
        }
    ],
}

MIN_LEN = 40       # drop trivially short prompts
MAX_LEN = 4000     # drop pathological / leaked mega-prompts


# ----------------------------------------------------------------------------- config
def load_config():
    try:
        import yaml  # type: ignore
        if SOURCES_FILE.exists():
            with open(SOURCES_FILE, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            if cfg and cfg.get("sources"):
                return cfg
    except Exception as exc:  # noqa: BLE001
        print(f"  (PyYAML unavailable or parse failed: {exc}; using embedded defaults)")
    return DEFAULT_CONFIG


# ----------------------------------------------------------------------------- helpers
def slugify(text, maxlen=70):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:maxlen] or "item"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "rmv-prompt-builder/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted, build-time)
        return resp.read().decode("utf-8", errors="replace")


def truthy(v):
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


# Ordered: first matching category wins. Keep keywords lowercase.
CATEGORY_RULES = [
    ("Programming", ["developer", "programmer", "code", "javascript", "typescript", "python",
                     "java ", "c++", "c#", "golang", " rust", "php", "react", "node", "html",
                     "css", "sql", "regex", "terminal", "linux", "shell", "git", "docker",
                     "kubernetes", "devops", "api", "compiler", "debug", "stackoverflow",
                     "software", "backend", "frontend", "algorithm"]),
    ("Data & Analytics", ["data scientist", "data analyst", "machine learning", "statistic",
                          "excel", "spreadsheet", "pandas", "dataset", "database", "csv",
                          "json", "etl", "dashboard", "analytics", "power bi", "tableau"]),
    ("Writing & Editing", ["writer", "author", "essay", "story", "novel", "blog", "copywrit",
                          "editor", "proofread", "grammar", "screenwriter", "poet", "poem",
                          "journalist", "content", "ghostwriter", "biographer"]),
    ("Marketing & SEO", ["marketing", "seo", "advertis", "ad copy", "social media", "brand",
                        "growth", "campaign", "influencer", "salesperson", "sales"]),
    ("Business & Product", ["ceo", "founder", "startup", "product manager", "business",
                           "consultant", "strategy", "entrepreneur", "project manager",
                           "scrum", "investor pitch"]),
    ("Career & Interview", ["interview", "recruiter", "resume", "cv ", "career", "cover letter",
                          "hiring", "human resources", " hr "]),
    ("Education & Tutoring", ["teacher", "tutor", "professor", "instructor", "lecturer",
                            "math ", "mathematic", "history", "study", "exam", "education",
                            "student", "school"]),
    ("Language & Translation", ["translator", "language", "english", "spanish", "french",
                              "german", "pronunciation", "ielts", "toefl", "linguist",
                              "etymolog", "synonym"]),
    ("Design & Creative", ["designer", "ux", "ui ", "logo", "artist", "illustrat", "midjourney",
                         "stable diffusion", "photograph", "image prompt", "interior design",
                         "fashion", "drawing"]),
    ("Science & Research", ["scientist", "research", "physic", "chemist", "biolog",
                          "mathematician", "astronom", "academic", "scientific", "engineer"]),
    ("Finance", ["financ", "invest", "account", "stock", "crypto", "bitcoin", "budget",
                "tax ", "trading", "economist", "economic"]),
    ("Health & Lifestyle", ["doctor", "health", "fitness", "personal trainer", "nutrition",
                          "diet", "mental health", "therapist", "life coach", "yoga",
                          "medical", "wellness", "psycholog", "dentist"]),
    ("Legal", ["lawyer", "legal", "attorney", "contract", " law ", "gdpr", "compliance",
              "patent"]),
    ("Roleplay & Persona", ["act as a character", "roleplay", "role-play", "pretend",
                          "fantasy", "game master", "dungeon", "character from", "rapper",
                          "comedian", "storyteller", "santa", "wizard", "philosopher"]),
    ("Productivity", ["assistant", "planner", "schedule", "to-do", "productivity", "note-taking",
                     "organize", "time management", "email", "meeting"]),
]


def categorize(title, prompt, for_devs):
    hay = f" {title.lower()} {prompt.lower()} "
    for cat, kws in CATEGORY_RULES:
        if any(kw in hay for kw in kws):
            return cat
    if for_devs:
        return "Programming"
    return "General"


ROLE_PATTERNS = [
    re.compile(r"^i want you to act as\s+(.+?)\.\s+", re.I | re.S),
    re.compile(r"^act as\s+(.+?)\.\s+", re.I | re.S),
    re.compile(r"^(?:i want you to\s+)?(?:imagine|pretend)(?:\s+that)?\s+you(?:'re| are)\s+(.+?)\.\s+", re.I | re.S),
    re.compile(r"^you are\s+(.+?)\.\s+", re.I | re.S),
]


def split_role(prompt):
    """Lift a leading 'act as ...' sentence into a Role/Persona line."""
    for pat in ROLE_PATTERNS:
        m = pat.match(prompt)
        if m:
            subject = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".")
            # keep it to a single clause
            subject = re.split(r"\.\s", subject)[0].strip()
            if 2 <= len(subject) <= 120:
                role = f"You are {subject}."
                task = prompt[m.end():].strip()
                if len(task) >= 20:
                    return role, task
    return "", prompt.strip()


VAR_PATTERN = re.compile(
    r"(my first\b[^.\"“]*?\b(?:is|will be|could be)\b[^.\"“]*?)"
    r"(\"[^\"]+\"|“[^”]+”|'[^']+')",
    re.I,
)


def inject_variable(task):
    """Turn a trailing example such as: My first command is "pwd"  →  {{input}}."""
    m = VAR_PATTERN.search(task)
    if m:
        return task[: m.start(2)] + "{{input}}" + task[m.end(2):]
    return task


def normalize_placeholders(text):
    """Map common template placeholder syntaxes onto the builder's {{var}} form."""
    # ${name} / ${ name }  ->  {{name}}
    text = re.sub(r"\$\{\s*([^}]+?)\s*\}", lambda m: "{{" + re.sub(r"\s+", "_", m.group(1).strip()) + "}}", text)
    # collapse accidental triple+ braces
    text = re.sub(r"\{\{\{+", "{{", text).replace("}}}", "}}")
    return text


# ----------------------------------------------------------------------------- normalize
def normalize(title, prompt, *, source, source_url, license_name,
              for_devs=False, kind="", contributor=""):
    prompt = (prompt or "").strip()
    title = re.sub(r"\s+", " ", (title or "").strip()) or "Untitled Prompt"
    role, task = split_role(prompt)
    task = normalize_placeholders(inject_variable(task))
    role = normalize_placeholders(role)
    category = categorize(title, prompt, for_devs)

    tags = []
    if for_devs:
        tags.append("developers")
    if kind and kind.upper() != "TEXT":
        tags.append(kind.lower())
    tags.append(category.split(" & ")[0].split(" ")[0].lower())

    return {
        "id": f"{slugify(source)}-{slugify(title)}",
        "title": title,
        "category": category,
        "role": role,
        "task": task,
        "context": "",
        "constraints": "",
        "output_format": "",
        "tone": "",
        "examples": [],
        "source": source,
        "source_url": source_url,
        "license": license_name,
        "contributor": (contributor or "").strip(),
        "rating": None,
        "tags": sorted(set(t for t in tags if t)),
    }


def parse_csv_source(text, src):
    cols = src.get("columns", {})
    c_title = cols.get("title", "act")
    c_prompt = cols.get("prompt", "prompt")
    c_dev = cols.get("for_devs", "for_devs")
    c_type = cols.get("type", "type")
    c_contrib = cols.get("contributor", "contributor")
    out = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        prompt = (row.get(c_prompt) or "").strip()
        if not (MIN_LEN <= len(prompt) <= MAX_LEN):
            continue
        out.append(normalize(
            row.get(c_title, ""), prompt,
            source=src["name"], source_url=src.get("source_url", ""),
            license_name=src.get("license", ""),
            for_devs=truthy(row.get(c_dev, "")),
            kind=row.get(c_type, ""), contributor=row.get(c_contrib, ""),
        ))
    return out


# ----------------------------------------------------------------------------- curate
def dedupe(items):
    seen, out = set(), []
    for it in items:
        key = re.sub(r"\s+", " ", it["task"].lower()).strip()[:160]
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def balance(items, limit):
    """Round-robin across categories so the catalog stays diverse under the cap."""
    if len(items) <= limit:
        return items
    by_cat = OrderedDict()
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    out, exhausted = [], False
    while len(out) < limit and not exhausted:
        exhausted = True
        for cat, bucket in by_cat.items():
            if bucket:
                out.append(bucket.pop(0))
                exhausted = False
                if len(out) >= limit:
                    break
    return out


# ----------------------------------------------------------------------------- write
def write_outputs(catalog, sources_meta):
    DATA.mkdir(parents=True, exist_ok=True)
    by_cat = defaultdict(int)
    for it in catalog:
        by_cat[it["category"]] += 1

    payload = {
        "schema": "rmv-prompt-builder/catalog@1",
        "count": len(catalog),
        "categories": OrderedDict(sorted(by_cat.items(), key=lambda kv: (-kv[1], kv[0]))),
        "sources": sources_meta,
        "prompts": catalog,
    }

    (DATA / "prompts.catalog.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "prompts.catalog.min.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    with open(DATA / "prompts.catalog.txt", "w", encoding="utf-8") as fh:
        fh.write(f"AI PROMPT BUILDER — CATALOG ({len(catalog)} prompts)\n")
        fh.write("Generated by scripts/build_catalog.py · prompt text licensed per-source.\n")
        fh.write("=" * 72 + "\n\n")
        for it in catalog:
            fh.write(f"# {it['title']}  [{it['category']}]\n")
            fh.write(f"Source: {it['source']} · License: {it['license']}\n\n")
            if it["role"]:
                fh.write(f"Role: {it['role']}\n\n")
            fh.write(it["task"].strip() + "\n\n")
            fh.write("-" * 72 + "\n\n")

    return payload


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    limit = int(cfg.get("max_prompts", 500))
    all_items, sources_meta = [], []

    for src in cfg.get("sources", []):
        if not src.get("enabled", True):
            print(f"• skip (disabled): {src.get('name')}")
            continue
        if src.get("format") != "csv":
            print(f"• skip (unsupported format '{src.get('format')}'): {src.get('name')}")
            continue
        print(f"• fetching: {src.get('name')}")
        text = None
        for url in (src.get("url"), src.get("fallback_url")):
            if not url:
                continue
            try:
                text = fetch(url)
                print(f"    ↳ {len(text):,} bytes from {url.split('//')[1].split('/')[0]}")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"    ! failed {url}: {exc}")
        if not text:
            print("    ! no data; skipping source")
            continue
        items = parse_csv_source(text, src)
        print(f"    ↳ normalized {len(items)} prompts")
        all_items.extend(items)
        sources_meta.append({
            "name": src.get("name"), "url": src.get("source_url"),
            "license": src.get("license"),
        })

    print(f"\nmerged: {len(all_items)} prompts")
    all_items = dedupe(all_items)
    print(f"deduped: {len(all_items)} prompts")
    all_items.sort(key=lambda it: (it["category"], it["title"].lower()))
    catalog = balance(all_items, limit)
    print(f"curated: {len(catalog)} prompts (cap {limit})")

    payload = write_outputs(catalog, sources_meta)
    print("\nwrote:")
    for name in ("prompts.catalog.json", "prompts.catalog.min.json", "prompts.catalog.txt"):
        p = DATA / name
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.0f} KB)")
    print("\ncategories:")
    for cat, n in payload["categories"].items():
        print(f"  {n:4d}  {cat}")


if __name__ == "__main__":
    main()
