# AI Prompt Builder

A 100% client-side prompt composer for any LLM, backed by a **pre-built, static
prompt catalog** of ~500 curated, structured templates. Part of the
[ruslanmv.com tools](https://ruslanmv.com/assets/tools/) collection.

Live: <https://ruslanmv.com/assets/tools/prompt-builder/>

## Two-page workflow

The tool is split into two focused pages so that **selecting** a template never
competes with **editing** one:

1. **`index.html` — Choose a Template.** Search, a category filter, and
   Built-in / Catalog / Recent tabs over calm list rows. Selecting a row opens a
   preview (description, included fields, category, use case, best-for). _Use
   template_ saves the choice to `localStorage` and opens the builder.
2. **`builder.html` — Populate Prompt.** Labelled fields on the left, the
   assembled prompt on the right, plus a _Change template_ link back to page 1.

Data is handed between the pages via `localStorage`:

| Key | Written by | Read by | Holds |
|---|---|---|---|
| `rmv-prompt-selected-template` | `index.html` | `builder.html` | the chosen template (fields + `_ts`) |
| `rmv-prompt-builder-state` | `builder.html` | `builder.html` | autosaved field/variable state |
| `rmv-prompt-recent` | `index.html` | `index.html` | recently used templates (max 8) |

The builder applies a freshly chosen template when its `_ts` differs from the
saved state; otherwise it restores your in-progress edits, so reloading keeps
your work.

## What it does

- Assembles a high-quality prompt from labelled fields — Role, Task, Context,
  Constraints, Output format, Tone, and few-shot examples.
- **Prompt catalog**: search / filter ~500 structured templates and load any one
  into the builder with a click. Each prompt shows its source and license.
- `{{variables}}` are auto-detected and filled from a table (reset / remove /
  add custom variables).
- System / User message split, approximate token + character count.
- Copy, download as `.txt`, and autosave to the browser (`localStorage`).
- No backend, no build step in the browser, no data ever leaves the device. If
  the catalog can't load, the six built-in templates still work.

## Layout

```
prompt-builder/
├─ index.html                  # Page 1 — Choose Template (selection only)
├─ builder.html                # Page 2 — Populate Prompt (editing only)
├─ original-index.html         # backup of the previous single-page version
├─ data/
│  ├─ prompts.catalog.json      # full catalog (pretty) + metadata
│  ├─ prompts.catalog.min.json  # what the browser fetch()es
│  └─ prompts.catalog.txt       # plain-text catalog (downloadable)
├─ scripts/
│  ├─ build_catalog.py          # downloads + normalizes datasets → data/
│  └─ sources.yml               # dataset sources + curation settings
└─ .github/workflows/
   └─ update-catalog.yml        # portable refresh workflow (standalone-repo use)
```

## Data strategy

The browser **never** downloads large datasets. Instead, `build_catalog.py` runs
at build time, fetches each source, normalizes it into the app's field schema,
de-duplicates, categorizes by keyword, curates a balanced subset (default 500),
and writes the three `data/` files. The app only `fetch()`es the small local
`prompts.catalog.min.json`.

### Catalog schema

```json
{
  "schema": "rmv-prompt-builder/catalog@1",
  "count": 500,
  "categories": { "Programming": 43, "Writing & Editing": 43, "...": 0 },
  "sources": [{ "name": "...", "url": "...", "license": "CC0-1.0" }],
  "prompts": [
    {
      "id": "fka-...-code-review",
      "title": "Code Review",
      "category": "Programming",
      "role": "You are a senior software engineer.",
      "task": "Review the provided code… My first request is {{input}}",
      "context": "", "constraints": "", "output_format": "", "tone": "",
      "examples": [],
      "source": "fka/awesome-chatgpt-prompts (prompts.chat)",
      "source_url": "https://github.com/f/awesome-chatgpt-prompts",
      "license": "CC0-1.0",
      "contributor": "", "rating": null,
      "tags": ["programming", "developers"]
    }
  ]
}
```

These fields map 1:1 onto the builder: `role → Role`, `task → Task`,
`context → Context`, `constraints → Constraints`, `output_format → Output
Format`, `tone → Tone`, `examples → Few-shot Examples`.

## Sources

| Priority | Dataset | Use | License |
|---|---|---|---|
| 1 (shipped) | [`fka/awesome-chatgpt-prompts`](https://github.com/f/awesome-chatgpt-prompts) (prompts.chat) | Main general-purpose template catalog | CC0-1.0 |
| 2 (optional, v2) | [`data-is-better-together/10k_prompts_ranked`](https://huggingface.co/datasets/data-is-better-together/10k_prompts_ranked) | Quality ranking / filtering | Apache-2.0 |

Only permissively-licensed prompt text ships in the public catalog. Every prompt
records its `source` and `license`.

## Rebuilding the catalog

Requires **Python 3.9+ only** (PyYAML is used if available, otherwise an embedded
copy of `sources.yml` is used). No other dependencies, no API keys.

```bash
cd assets/tools/prompt-builder
python3 scripts/build_catalog.py
```

Edit `scripts/sources.yml` to change `max_prompts`, enable/disable sources, or add
new CSV sources.

## Automatic refresh

`.github/workflows/update-catalog.yml` (here) is a **portable** workflow for when
this folder is published as its own repository. Within the ruslanmv.com site, the
functional weekly refresh lives at the repo root:
`.github/workflows/update-prompt-catalog.yml`. Both run `build_catalog.py` and
commit the regenerated `data/` files.

## Local preview

Open `index.html` directly (`file://`) and the Built-in tab works, but the
Catalog tab's `fetch()` is blocked by the browser. To preview the full catalog:

```bash
cd assets/tools/prompt-builder
python3 -m http.server 8000
# open http://localhost:8000/  (Choose Template → Use template → Populate Prompt)
```
