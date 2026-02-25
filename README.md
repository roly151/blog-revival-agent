# Blog Content Revival Agent

**Built by [AI Ranking](https://skool.com/ai-ranking) — free to use and modify.**

This tool rewrites blog posts flagged as "Crawled - currently not indexed" in Google Search Console. It fetches your post, audits what's wrong, then rewrites it using the capsule content technique — structured for Google Featured Snippets and AI answer engines.

Each rewrite includes:
- Matched tone of voice to your original content
- Capsule content structure (question H2s with bold direct answers)
- Real external citations linked to credible sources
- Internal links pulled from your sitemap
- Minimum 1,200 words

---

## Option A: Use it instantly (no install)

👉 **[Open the tool here](https://your-app.streamlit.app)** ← *(replace with your Streamlit Cloud URL)*

You just need an OpenRouter API key and model name. The tool runs in your browser — nothing is installed on your computer.

---

## Option B: Run it locally

Running it locally means your rewrites stay on your machine, you can modify the prompts, and it works offline once set up.

### Step 1: Get Python

Check if you have it:
```
python3 --version
```

If you get an error, download Python from [python.org/downloads](https://python.org/downloads) and install it. Any version 3.10 or newer works.

### Step 2: Download this tool

Click the green **Code** button at the top of this page, then **Download ZIP**. Unzip it somewhere easy to find (like your Desktop).

Or if you use Git:
```
git clone https://github.com/YOUR-USERNAME/blog-revival-agent.git
```

### Step 3: Install dependencies

Open Terminal (Mac) or Command Prompt (Windows), navigate to the folder, and run:

```
cd blog-revival-agent
pip install -r requirements.txt
```

This installs everything the tool needs. You only do this once.

### Step 4: Launch the tool

```
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Getting your OpenRouter API key

The tool uses OpenRouter so you can pick any supported model (OpenAI, Anthropic, Google, etc). You pay based on the model you choose.

1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up or log in
3. Open [openrouter.ai/keys](https://openrouter.ai/keys)
4. Click **Create Key**, give it a name, copy it
5. Paste it into the tool, then set your model (example: `openai/gpt-4o-mini`)

Your key is only stored in your browser session — it's never saved to disk or sent anywhere except OpenRouter's API.

---

## How to use it

### Exporting from Google Search Console

1. Open [Google Search Console](https://search.google.com/search-console)
2. Click **Indexing** → **Pages** in the left sidebar
3. Click on **"Crawled - currently not indexed"** in the list
4. Click the **Export** button (top right) → **Download CSV**

That CSV is what you drop into the tool.

### Running a rewrite

1. Enter your website domain (e.g. `https://myblog.com`)
2. Upload your GSC CSV export, or paste URLs manually
3. Paste your OpenRouter API key
4. Enter the OpenRouter model you want to use
5. Click **Run Analysis & Rewrite**

The tool will:
- Fetch each post
- Run a 2-pass OpenRouter analysis (audit then rewrite)
- Show you a live cost counter as it runs
- Display the rewritten posts with a summary table
- Let you download everything as a ZIP

---

## Customising the tool

The prompts that control how the model rewrites your content are in `rewriter.py`. Open it in any text editor (or VS Code) and look for `REWRITE_PROMPT` and `ANALYSIS_PROMPT`.

Common things people change:
- **Minimum word count** — find `Minimum 1,200 words` in the rewrite prompt and change the number
- **Tone instructions** — the tone section at the top of `REWRITE_PROMPT` can be adjusted
- **Number of internal links** — find `at least 3-5 internal links` and change the range
- **Target year for updates** — find `2024-2025 information` and update as needed

Save the file and reload the browser — changes take effect immediately.

---

## File structure

```
blog-revival-agent/
├── app.py                  ← Streamlit UI (layout, inputs, results display)
├── content_fetcher.py      ← Fetches and parses your blog post HTML
├── sitemap_crawler.py      ← Discovers internal pages from your sitemap
├── rewriter.py             ← Rewrite prompts and OpenRouter API calls
├── requirements.txt        ← Python dependencies
└── output/                 ← Rewritten posts saved here as .md files
```

---

## Troubleshooting

**"No sitemap found"**
Your sitemap needs to be at `/sitemap.xml` or declared in `robots.txt`. WordPress sites usually have one at `/wp-sitemap.xml`. If yours is somewhere else, paste the sitemap URL directly into the domain field.

**"Failed to fetch post"**
Some sites block automated requests. Try with a different post first to confirm the tool is working. Cloudflare-protected sites may block fetches.

**API key error**
Double-check the key at [openrouter.ai/keys](https://openrouter.ai/keys), ensure it is active, and verify your selected model is available to your account.

**Port already in use**
If `localhost:8501` throws an error, another Streamlit app is running. Either close it or run on a different port:
```
streamlit run app.py --server.port 8502
```

---

## Questions?

Post in the [AI Ranking Skool community](https://skool.com/ai-ranking) — someone will help.
