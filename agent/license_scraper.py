"""
LLM-backed license scraper for NVIDIA NGC model cards.

Extracts license information by:
  1. Fetching the model card page from build.nvidia.com
  2. Extracting termsOfUse JSON fields + GOVERNING TERMS blocks
  3. Asking an LLM to pick the correct model license
  4. Falling back to regex extraction if LLM is unavailable

Usage:
    from license_scraper import find_license

    # With LLM (recommended, needs NGC_API_KEY):
    license = find_license("deepseek-v3_1", "DeepSeek AI", use_llm=True)

    # From a catalog resource dict:
    license = find_license_for_resource(resource_dict, use_llm=True)
"""

import json
import os
import re
from typing import Optional

import httpx
from openai import OpenAI
from json_repair import repair_json

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_LLM_MODEL = "meta/llama-3.1-8b-instruct"

SESSION = httpx.Client(
    headers={
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "Mozilla/5.0 (compatible; nim-license-scraper/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    },
    follow_redirects=True,
)

_LLM_CLIENT: OpenAI | None = None


def _get_llm_client(api_key: str) -> OpenAI:
    """Return a cached OpenAI client, creating it on first call."""
    global _LLM_CLIENT
    if _LLM_CLIENT is None or _LLM_CLIENT.api_key != api_key:
        _LLM_CLIENT = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)
    return _LLM_CLIENT

# ---------------------------------------------------------------------------
# License registry  (specific -> generic, order matters)
# ---------------------------------------------------------------------------

LICENSE_REGISTRY: list[tuple[str, str]] = [
    (r"apache[\s\-]*(?:license[\s,\-]*)?(?:version[\s,\-]*)?2(?:\.0)?", "Apache 2.0"),
    (r"\bmit\b", "MIT"),
    (r"bsd[\s\-]*3[\s\-]*clause", "BSD-3-Clause"),
    (r"cc[\s\-]*by[\s\-]*sa[\s\-]*3\.0", "CC BY-SA 3.0"),
    (r"cc[\s\-]*by[\s\-]*(?:nc[\s\-]*)?4\.0", "CC BY 4.0"),
    (r"agpl[\s\-]*3\.0", "AGPL-3.0"),
    (r"gpl[\s\-]*3\.0", "GPL 3.0"),
    (r"gpl[\s\-]*2\.0", "GPL 2.0"),
    (r"llama[\s\-]*4[\s\-]*community", "Llama 4 Community Model License"),
    (r"llama[\s\-]*3\.2", "Llama 3.2"),
    (r"llama[\s\-]*3\.1", "Llama 3.1"),
    (r"llama[\s\-]*3(?:\.\d+)?(?:[\s\-]*community)?", "Llama 3"),
    (r"llama[\s\-]*2", "Llama 2"),
    (r"nvidia[\s\-]*nemotron[\s\-]*open[\s\-]*model[\s\-]*license", "NVIDIA Nemotron Open Model License"),
    (r"nvidia[\s\-]*nemo[\s\-]*foundational[\s\-]*models?[\s\-]*evaluation[\s\-]*license", "NVIDIA NeMo Foundational Models Evaluation License"),
    (r"nvidia[\s\-]*software[\s\-]*and[\s\-]*model[\s\-]*evaluation[\s\-]*license", "NVIDIA Software and Model Evaluation License"),
    (r"nvidia[\s\-]*evaluation[\s\-]*license", "NVIDIA Evaluation License Agreement"),
    (r"(?:nvidia[\s\-]*)?ai[\s\-]*foundation[\s\-]*models?[\s\-]*community[\s\-]*license(?:[\s\-]*agreement)?", "NVIDIA AI Foundation Models Community License"),
    (r"nvidia[\s\-]*community[\s\-]*(?:model[\s\-]*)?license", "NVIDIA Community Model License"),
    (r"nvidia[\s\-]*open[\s\-]*(?:model[\s\-]*)?license(?:[\s\-]*agreement)?", "NVIDIA Open Model License"),
    (r"\beula\b", "EULA"),
    (r"gemma[\s\-]*terms[\s\-]*of[\s\-]*use", "Gemma Terms of Use"),
    (r"hive[\s\-]*terms[\s\-]*of[\s\-]*use", "Hive Terms of Use"),
    (r"bigcode[\s\-]*openrail", "BigCode OpenRAIL-M v1 License Agreement"),
    (r"jamba[\s\-]*open[\s\-]*license", "Jamba Open License Agreement"),
    (r"falcon[\s\-]*3[\s\-]*tii", "Falcon 3 TII Falcon License"),
    (r"license[\s\-]*agreement[\s\-]*for[\s\-]*colosseum", "License agreement for Colosseum"),
    (r"license[\s\-]*agreement[\s\-]*for[\s\-]*italia", "License agreement for Italia"),
    (r"nvidia[\s\-]*technology[\s\-]*access[\s\-]*terms", "NVIDIA Technology Access Terms of Use"),
    (r"mistral[\s\-]+\w*[\s\-]*license", "Mistral License"),
    (r"deepseek[\s\-]+license", "DeepSeek License"),
    (r"qwen[\s\-]+license", "Qwen License"),
    (r"community[\s\-]*license[\s\-]*for[\s\-]*baichuan", "Baichuan License"),
    (r"baichuan[\s\-]+license", "Baichuan License"),
]

LICENSE_PATTERNS = [pat for pat, _ in LICENSE_REGISTRY]
DATALOOP_LICENSES = list(dict.fromkeys(name for _, name in LICENSE_REGISTRY))

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_html(text: str) -> str:
    out = re.sub(r"<a[^>]*>([^<]*)</a>", r"\1", text)
    out = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", out)
    out = re.sub(r"<[^>]+>", " ", out)
    return _normalize_space(out)


def _normalize_license(raw: str) -> str:
    """Normalize a raw license string to a canonical Dataloop name."""
    result = "Other"

    # 1. Regex match from registry
    for pattern, canonical in LICENSE_REGISTRY:
        if re.search(pattern, raw, re.IGNORECASE):
            result = canonical
            break

    # 2. Fuzzy substring match against known licenses
    if result == "Other":
        normalized = raw.strip().lower()
        for dl_license in DATALOOP_LICENSES:
            if dl_license.lower() in normalized or normalized in dl_license.lower():
                result = dl_license
                break

    return result


def _extract_license_from_text(text: str) -> Optional[str]:
    """Regex-based license extraction from page text."""
    result = None

    # Pattern 1 (highest priority): "model is governed by <license>"
    # Covers: "use of this model is governed by", "The model is governed by",
    #         "Your use of the model is governed by", etc.
    m = re.search(
        r"(?:use\s+of\s+(?:this|the)\s+)?model\s+is\s+governed\s+by\s+(?:the\s+)?(.+?)"
        r"(?:[;.](?:\s|$)|Additional|\s*$)",
        text, re.IGNORECASE,
    )
    if m:
        candidate = _normalize_space(m.group(1)).strip(" .,;")
        candidate = re.split(r"\s+and\s+(?:the\s+)?", candidate, maxsplit=1)[0].strip(" .,;")
        if candidate:
            result = candidate

    # Pattern 2: "released under the <license>"
    if result is None:
        m = re.search(
            r"released\s+under\s+(?:the\s+)?(.+?)(?:\s+license)?(?:[;.](?:\s|$)|\s*$)",
            text, re.IGNORECASE,
        )
        if m:
            candidate = _normalize_space(m.group(0)).strip(" .,;")
            result = candidate

    # Pattern 3: "License: <name>"
    if result is None:
        m = re.search(r"(?:^|\s)License\s*:?\s+([A-Z][\w\s.\-]+)", text, re.IGNORECASE)
        if m:
            result = _normalize_space(m.group(1)).strip(" .")

    # Pattern 4: scan for known license patterns
    if result is None:
        for pattern in LICENSE_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                result = _normalize_space(m.group(0))
                break

    return result

# ---------------------------------------------------------------------------
# Model card fetcher
# ---------------------------------------------------------------------------

def _fetch_modelcard_sections(model_name: str, publisher: str) -> tuple[str, str]:
    """
    Fetch build.nvidia.com modelcard page and extract license sections.

    Tries both dot and underscore URL variants since build.nvidia.com is
    inconsistent (some models use dots, others underscores in the slug).

    Returns (sections_text, url).
    """
    slug = model_name.split("/")[-1]
    publisher_slug = publisher.lower().replace(" ", "-")

    base = [slug]
    if "." in slug:
        base.append(slug.replace(".", "_"))
        base.append(slug.replace(".", ""))
    elif "_" in slug:
        base.append(slug.replace("_", "."))
    v_stripped = re.sub(r"v0[._](\d)", r"v\1", slug)
    if v_stripped != slug:
        base.append(v_stripped)

    candidates = list(base)
    for v in base:
        no_version = re.sub(r"-v\d+[._]?\d*$", "", v)
        if no_version != v:
            candidates.append(no_version)
        if v.endswith("-instruct"):
            candidates.append(v[:-len("-instruct")])
    candidates = list(dict.fromkeys(candidates))

    html = ""
    url = ""
    for s in candidates:
        url = f"https://build.nvidia.com/{publisher_slug}/{s}/modelcard"
        try:
            resp = SESSION.get(url, timeout=15)
            resp.raise_for_status()
            html = resp.text
            if "termsOfUse" in html or len(html) > 100_000:
                break
        except Exception:
            continue

    result = ""
    if html:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decoded = html.encode().decode("unicode_escape", errors="ignore")
        decoded = decoded.replace("\\n", "\n").replace("\\t", " ")
        decoded = re.sub(r'"\]\)\s*self\.__next_f\.push\(\[\d+,"', " ", decoded)
        cleaned = _clean_html(decoded)

        sections: list[str] = []
        slug_lower = slug.lower()

        # Strategy 1: termsOfUse JSON fields
        seen_terms: set[str] = set()
        for m in re.finditer(r'"termsOfUse"\s*:\s*"(.*?)(?<!\\)"\s*[,}]', decoded, re.DOTALL):
            val_clean = _clean_html(m.group(1)).strip()
            if not val_clean or len(val_clean) < 20:
                continue
            context = decoded[max(0, m.start() - 100): m.end() + 500]
            artifact_m = re.search(r'"artifactName"\s*:\s*"([^"]+)"', context)
            if artifact_m and artifact_m.group(1).lower() != slug_lower:
                continue
            if val_clean not in seen_terms:
                seen_terms.add(val_clean)
                sections.append(f"[termsOfUse field]\n{val_clean}")

        # Strategy 2: GOVERNING TERMS blocks
        for m in re.finditer(r'GOVERNING\s+TERMS.*?(?=GOVERNING\s+TERMS|#{1,3}\s+\w|$)',
                             cleaned, re.IGNORECASE | re.DOTALL):
            block = m.group(0).strip()
            if len(block) < 30:
                continue
            foreign = re.findall(r'"artifactName"\s*:\s*"([^"]+)"', block)
            if not foreign or any(f.lower() == slug_lower for f in foreign):
                sections.append(f"[governing block]\n{block[:600]}")

        # Strategy 3: markdown ### License section
        md_match = re.search(
            r'#{1,4}\s+(?:License|Terms)[^\n]*\n(.*?)(?=#{1,4}\s|\Z)',
            cleaned, re.DOTALL | re.IGNORECASE,
        )
        if md_match:
            sections.append(f"[license section]\n{md_match.group(1)[:800].strip()}")

        if sections:
            result = "\n\n---\n\n".join(sections)
        else:
            for kw in ("governed by", "License/Terms", "Terms of Use", "License"):
                idx = cleaned.lower().find(kw.lower())
                if idx != -1:
                    result = cleaned[max(0, idx - 100): idx + 1000].strip()
                    break
            if not result:
                result = cleaned[-2000:].strip()
    else:
        print(f"  [license] fetch error: no valid page for {model_name}")

    return result, url

# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are a license extraction assistant for NVIDIA NGC AI model cards.
Your sole task: return the PRIMARY license that governs USE OF THE MODEL WEIGHTS/PARAMETERS.

## Input format
You will receive one or more labeled sections from the model card:
- [termsOfUse field]  -- structured JSON field embedded in the page
- [governing block]   -- a GOVERNING TERMS paragraph from the rendered page
- [license section]   -- a markdown ### License/Terms section

When multiple sections are present they may CONFLICT. Follow these rules.

## Priority rules (highest to lowest)

1. IGNORE any sentence mentioning "trial service", "API Trial", "NIM container", or
   "API Catalog Terms of Service" -- those govern the hosting endpoint, not the model.

2. FIND the sentence: "Use of this model is governed by X" or
   "use of the model is governed by X" or "Your use of this model is governed by X"
   -> X is the model license. Return it.

3. [termsOfUse field] can be STALE for some models (e.g. Llama Nemotron family).
   If a [license section] or [governing block] from the rendered page DISAGREES
   with the [termsOfUse field], PREFER the [license section] / [governing block].

4. IGNORE "ADDITIONAL INFORMATION" -- it is always the code/SDK license, not the model.

5. Multiple licenses joined by "and": "NVIDIA X License and Apache 2.0"
   -> Return only the FIRST one. The rest are always code licenses.

6. Multiple governing blocks (e.g. "Cloud API" vs "Download"):
   -> Pick the one that says "Use of this model is governed by" (Cloud API terms).

## Canonical output names
"Apache 2.0", "MIT", "NVIDIA Open Model License", "NVIDIA Community Model License",
"NVIDIA AI Foundation Models Community License", "NVIDIA Nemotron Open Model License",
"NVIDIA Evaluation License Agreement", "Llama 3 Community License",
"Llama 3.1 Community License", "Llama 3.2", "Llama 3.3 Community License",
"Gemma Terms of Use", "DeepSeek License", "Qwen License", "Baichuan License"

If truly unknown: "UNKNOWN".

Respond ONLY with valid JSON, no markdown, no extra text:
{"primary_license": "...", "reasoning": "one sentence"}"""


def _llm_extract_license(model_name: str, page_section: str, api_key: str) -> Optional[str]:
    """Ask NIM LLM to extract the license from the model card sections."""
    result = None

    if not page_section.strip():
        raise ValueError("empty page section")

    user_msg = f'Model: "{model_name}"\n\nModel card license sections:\n{page_section[:2500]}'

    try:
        client = _get_llm_client(api_key)
        response = client.chat.completions.create(
            model=NIM_LLM_MODEL,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        parsed = json.loads(repair_json(raw))
        lic = parsed.get("primary_license", "").strip()
        if lic and lic != "UNKNOWN":
            result = lic
    except ValueError:
        raise
    except Exception as e:
        print(f"  [license-llm] error: {e}")

    return result

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_license(
    model_name: str,
    publisher: str,
    use_llm: bool = True,
    api_key: str = None,
) -> Optional[str]:
    """
    Find the canonical license for a model by scraping its NVIDIA model card.

    Args:
        model_name: NGC catalog model name (e.g. "deepseek-v3_1")
        publisher:  Publisher label from catalog (e.g. "DeepSeek AI")
        use_llm:    Use LLM extraction (recommended). Falls back to regex.
        api_key:    NGC API key. Defaults to NGC_API_KEY env var.

    Returns:
        Canonical license string (e.g. "NVIDIA Community Model License") or None.
    """
    result = None

    if not publisher:
        print(f"  [license] skipping {model_name!r} — no publisher")
        return result

    page_section, url = _fetch_modelcard_sections(model_name, publisher)

    if page_section:
        if api_key is None:
            api_key = os.environ.get("NGC_API_KEY")

        # LLM extraction (primary)
        if use_llm and api_key:
            try:
                llm_raw = _llm_extract_license(model_name, page_section, api_key)
            except ValueError:
                llm_raw = None
            if llm_raw:
                canonical = _normalize_license(llm_raw)
                if canonical != "Other":
                    result = canonical

        # Regex fallback
        if result is None:
            raw = _extract_license_from_text(page_section)
            if raw:
                canonical = _normalize_license(raw)
                if canonical != "Other":
                    result = canonical

    return result


def find_license_for_resource(resource: dict, use_llm: bool = True, api_key: str = None) -> Optional[str]:
    """
    Find the license for a catalog resource dict (as returned by NGC search API).

    Args:
        resource: NGC catalog resource dict with "name" and "labels"
        use_llm:  Use LLM extraction
        api_key:  NGC API key

    Returns:
        Canonical license string or None.
    """
    model_name = resource.get("name", "")
    if not model_name:
        raise ValueError("resource dict missing 'name' key")

    publisher = resource.get("publisher", "")
    if not publisher:
        for label in resource.get("labels", []):
            if label.get("key") == "publisher":
                publisher = (label.get("values") or [""])[0]
                break

    return find_license(model_name, publisher, use_llm=use_llm, api_key=api_key)
