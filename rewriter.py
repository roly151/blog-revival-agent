import json
import re

from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are an SEO content auditor. Analyze this blog post and return ONLY a valid JSON object with no extra text, no markdown fences.

Return exactly this structure:
{{
  "recommended_title": "a stronger SEO title for this post",
  "thin_sections": ["list of H2 heading texts where the section content is under 150 words"],
  "outdated_claims": ["exact sentences or phrases that appear outdated, reference old stats, or use vague 'recently' language"],
  "missing_internal_links": ["topics or keywords mentioned in the post that match available site pages"],
  "missing_external_links": ["specific factual claims that need a citation - quote the claim"],
  "overall_word_count": {word_count},
  "verdict": "thin"
}}

recommended_title rules:
- Keep the exact topic and intent of the original post (no bait-and-switch)
- Make it more specific and compelling (clear promise, fewer generic words)
- Aim for ~45-65 characters (good for SERP display)
- Match the author's tone (casual vs formal)
- Do not add a year unless it is truly central to the post

Verdict must be one of: "thin" (under 800 words or sparse), "average" (800-1200 words), "good" (1200+ words with depth).

BLOG POST TITLE: {title}
CURRENT WORD COUNT: {word_count}

CURRENT POST CONTENT:
{body_text}

AVAILABLE SITE PAGES (for internal link matching):
{site_pages}

Return only the JSON object."""


REWRITE_PROMPT = """You are an expert SEO content writer. Rewrite the blog post below following every rule in this brief exactly.

TITLE (important):
- Use the recommended title below as the H1 (# ...) if it is provided
- If it's blank or unusable, keep the original title
- Do not change the topic, audience, or promise of the article

TONE & VOICE (critical — do this first):
- Read the original post carefully and identify the author's tone: casual or formal, technical or plain-English, use of "you" or third-person, sentence length, vocabulary level, use of humour or directness
- The rewrite MUST sound like the same person wrote it. Do not homogenise or make it generic
- If the original is conversational, keep it conversational. If it's technical, keep it technical
- Preserve any recurring phrases, stylistic quirks, or structural habits the author uses
- NEVER use em dashes (—) anywhere in the post. Replace them with a colon, comma, parentheses, or split into two sentences instead

CAPSULE CONTENT STRUCTURE (apply to 60% of H2 sections):
- H2 heading must be phrased as a question
- Immediately after the H2, write a 30-60 word direct answer in **bold** - self-contained enough to be a Google featured snippet
- Follow with 2-3 supporting paragraphs (depth, examples, data)

CONTENT REQUIREMENTS:
- Update any outdated facts with accurate 2024-2025 information
- Minimum 1,200 words total
- Fix all issues listed in the audit below

AUSTRALIAN LOCALISATION (mandatory):
- Use Australian English spelling and terminology (e.g., "colour", "organise", "optimise", "centre")
- Use metric/SI units only where possible: kg/g, cm/m/km, C, km/h, L/ml, m2/ha
- Convert any imperial/US units to metric and keep one unit system (do not show both unless clarity requires it)
- Use Australian date style (e.g., 26 February 2026) and avoid US month/day formatting
- Prefer Australia-relevant context, regulations, and examples where location-specific references are needed
- If currency is used in an Australian context, use A$ or AUD clearly

EXTERNAL LINKS (important):
- Back up every factual claim with a real, working URL from a credible source
- Use well-known sources: government sites (.gov), major publications (e.g. Search Engine Journal, Moz, Ahrefs blog, Google Search Central, HubSpot, Semrush blog), Wikipedia for general facts, academic/research sources
- Format: [descriptive anchor text](https://real-url.com)
- Only use a placeholder like (SOURCE: domain.com) if you are genuinely uncertain of the exact URL — prefer real links

INTERNAL LINKS (important):
- Add at least 3-5 internal links using the site pages listed below
- Use the FULL URL provided for each page — do not shorten to a relative path
- Format: [contextual anchor text](https://full-url-from-sitemap)
- Anchor text must be natural and contextual, never "click here"

FAQ SECTION (required, append at the very end of the post):
- Add a ## Frequently Asked Questions section as the final section
- Include 3 to 5 questions that readers of this topic are likely to ask
- The questions MUST be different from any question already answered or addressed in the post body — look at the H2 headings and content and avoid duplicating those topics
- Format each as: ### Question text? followed by a concise 2-4 sentence answer
- Questions should be genuinely useful and reflect real search intent around the topic

OUTPUT FORMAT:
- Clean markdown only — start directly with the # title, no preamble or commentary
- Do not add any text before or after the post content

ORIGINAL TITLE: {title}

RECOMMENDED TITLE (use this as the H1 if provided):
{recommended_title}

AUDIT FINDINGS:
{audit}

AVAILABLE SITE PAGES FOR INTERNAL LINKS (use these full URLs):
{site_pages}

ORIGINAL POST TO REWRITE:
{body_text}"""

AU_LOCALISATION_PROMPT = """You are an expert Australian copy editor.

Localise the markdown blog post below for an Australian audience while preserving meaning, SEO intent, structure, links, and headings.

Rules:
- Keep markdown structure intact.
- Use Australian English spelling and phrasing.
- Use metric/SI units only where practical (kg/g, cm/m/km, C, km/h, L/ml, m2/ha).
- Convert imperial/US units to metric with accurate conversions.
- Use Australian date style (e.g., 26 February 2026), not US month/day.
- Where currency is local to Australia, format as A$ or AUD.
- Keep all links valid and unchanged.
- Return markdown only with no preamble.

MARKDOWN TO LOCALISE:
{body_text}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_site_pages(site_pages: list, domain: str = "") -> str:
    """
    Format site pages for prompt injection.
    Uses full URLs so the selected model can place them directly into links.
    Caps at 100 pages to avoid token overflow.
    """
    if not site_pages:
        return "No sitemap pages available."

    # Normalise domain for building full URLs
    base = domain.rstrip("/") if domain else ""

    lines = []
    for page in site_pages[:100]:
        full_url = page.get("url", "")
        slug = page.get("slug", "")
        title = page.get("title", "")

        # Build full URL if we only have a slug
        if not full_url and slug and base:
            full_url = base + ("" if slug.startswith("/") else "/") + slug.lstrip("/")

        display_url = full_url or slug or "(no url)"
        lines.append(f"- {display_url} | {title}")

    return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from a string."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_message_text(response) -> str:
    """Extract assistant text from OpenAI-compatible chat completion responses."""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(getattr(part, "text", ""))
        return "".join(parts)
    return ""


def _usage_cost(response) -> dict:
    """Convert an API response usage object to a cost breakdown dict."""
    usage = getattr(response, "usage", None)
    inp = getattr(usage, "prompt_tokens", 0) if usage else 0
    out = getattr(usage, "completion_tokens", 0) if usage else 0

    cost = 0.0
    for candidate in (getattr(usage, "cost", None), getattr(response, "cost", None)):
        if candidate is None:
            continue
        try:
            cost = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cost_usd": cost,
    }


def _combine_usage(*usages: dict) -> dict:
    """Combine usage dicts from multiple API calls."""
    total_in = 0
    total_out = 0
    total_cost = 0.0
    for usage in usages:
        total_in += int(usage.get("input_tokens", 0))
        total_out += int(usage.get("output_tokens", 0))
        total_cost += float(usage.get("cost_usd", 0.0))
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": total_cost,
    }


_AU_LOCALISATION_CHECKS = [
    r"\b(?:lbs?|pounds?|oz|ounces?|inches?|inch|ft|feet|foot|yards?|miles?)\b",
    r"\b(?:mph|fahrenheit|°f)\b",
    r"\b(?:color|organize|optimize|center|meter|liter|analyze)\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
]


def _needs_au_localisation(text: str) -> bool:
    """Detect likely non-Australian conventions in the generated draft."""
    body = (text or "").lower()
    return any(re.search(pattern, body) for pattern in _AU_LOCALISATION_CHECKS)


def _localise_for_australia(draft_markdown: str, client: OpenAI, model: str) -> tuple[str, dict]:
    """Run a cleanup pass to enforce Australian localisation conventions."""
    prompt = AU_LOCALISATION_PROMPT.format(body_text=draft_markdown[:12000])
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    localised = _extract_message_text(response).strip()
    usage = _usage_cost(response)
    return (localised or draft_markdown), usage


def analyze_post(
    post: dict,
    site_pages: list,
    client: OpenAI,
    model: str,
    domain: str = "",
) -> tuple[dict, dict]:
    """
    First pass: analyze the post and return (audit_dict, usage_dict).
    audit_dict falls back gracefully if JSON parsing fails.
    usage_dict contains input_tokens, output_tokens, cost_usd.
    """
    prompt = ANALYSIS_PROMPT.format(
        title=post.get("title", "Untitled"),
        word_count=post.get("word_count", 0),
        body_text=post.get("body_text", "")[:8000],
        site_pages=_format_site_pages(site_pages, domain),
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    usage = _usage_cost(response)
    raw = _strip_code_fences(_extract_message_text(response))

    try:
        return json.loads(raw), usage
    except json.JSONDecodeError:
        audit = {
            "recommended_title": post.get("title", "Untitled"),
            "thin_sections": [],
            "outdated_claims": [],
            "missing_internal_links": [],
            "missing_external_links": [],
            "overall_word_count": post.get("word_count", 0),
            "verdict": "average",
            "_parse_error": raw[:500],
        }
        return audit, usage


def rewrite_post(
    post: dict,
    audit: dict,
    site_pages: list,
    client: OpenAI,
    model: str,
    domain: str = "",
) -> tuple[str, dict]:
    """
    Second pass: produce a fully rewritten post as markdown.
    Returns (markdown_string, usage_dict).
    """
    recommended_title = str(audit.get("recommended_title", "")).strip()
    prompt = REWRITE_PROMPT.format(
        title=post.get("title", "Untitled"),
        recommended_title=recommended_title,
        audit=json.dumps(audit, indent=2),
        site_pages=_format_site_pages(site_pages, domain),
        body_text=post.get("body_text", "")[:8000],
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    draft = _extract_message_text(response).strip()
    usage_primary = _usage_cost(response)

    if not draft:
        return draft, usage_primary

    if not _needs_au_localisation(draft):
        return draft, usage_primary

    localised, usage_localise = _localise_for_australia(draft, client, model)
    return localised, _combine_usage(usage_primary, usage_localise)
