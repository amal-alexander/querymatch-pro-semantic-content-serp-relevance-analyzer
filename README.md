# QueryMatch Pro — Semantic Content & SERP Relevance Analyzer

Compare your page and competitor pages against one target search query. See which content sits closest to the query using cosine similarity and Euclidean distance, complete with automated performance diagnostics, SERP-vs-semantic correlation, and actionable SEO recommendations.

**Built by Amal Alexander** · [amalalex95@gmail.com](mailto:amalalex95@gmail.com)

## What it does

- Has an **Analyzer** tab and a **Models** tab. The Models tab lists every model with its type, languages, size, speed and what it is best for
- Takes one target query, your page, and 1–20 competitor pages
- Lets you paste page content, or crawl a URL and fill the content box automatically
- Turns the query and every page into vectors with the model you choose
- Shows a query-to-page summary table, a cosine similarity matrix and a Euclidean distance matrix
- Delivers a **Dynamic Results Interpretation & Website Performance Analysis** section detailing which website is performing best and why, SERP rank vs. semantic alignment, and custom SEO action plans
- Exports the results as CSV (`querymatch_pro_similarity_results.csv`)
- Keeps the Google rank (1–10, Not in Page 1, Unknown) as context only. It never changes the vectors

## Models

| Model | Type | Languages | Size | Notes |
|---|---|---|---|---|
| BERT Base Uncased | Plain BERT | English | ~440 MB | Not trained for similarity. Scores sit close together |
| BERT Base Multilingual Uncased | Plain BERT | 100+ | ~670 MB | Same limitation as above |
| MiniLM L6 v2 (`all-MiniLM-L6-v2`) | Sentence model | English | ~90 MB | Fast. Good default |
| MiniLM L12 v2 (`all-MiniLM-L12-v2`) | Sentence model | English | ~130 MB | A bit better, a bit slower |
| Multilingual MiniLM L12 (`paraphrase-multilingual-MiniLM-L12-v2`) | Sentence model | 50+ | ~470 MB | Small multilingual option |
| MPNet Base v2 (`all-mpnet-base-v2`) | Sentence model | English | ~420 MB | Higher quality, heavier |
| BGE Small EN v1.5 | Retrieval model | English | ~130 MB | CLS pooling, query instruction added |
| BGE Base EN v1.5 | Retrieval model | English | ~440 MB | Stronger than small |
| Multilingual E5 Small | Retrieval model | 100 | ~470 MB | Uses `query:` / `passage:` prefixes |
| GTE Small | Embedding model | English | ~130 MB | Small general-purpose model |
| BM25 (Lexical) | Term weights | Any | none | No download. Keyword overlap only |

There is no official "MiniLM v3". The current MiniLM releases are the L6 v2 and L12 v2 checkpoints above.

Only the selected model is loaded. The app keeps at most two in memory. The first run of each model downloads it from Hugging Face. No API key is needed.

### Which one should I pick?

- Quick check: **MiniLM L6 v2**
- Best English quality: **BGE Base** or **MPNet Base v2**
- Non-English pages: **Multilingual E5 Small** or **Multilingual MiniLM**
- Keyword overlap only: **BM25**

## How vectors are built

**Transformer models**

1. Tokenize the query and each page.
2. Split long pages into token chunks that fit the model limit, so text past the limit is not dropped.
3. Run each chunk through the model.
4. Pool the output: mean pooling for most models, CLS pooling for BGE.
5. Add the model's prefix where it needs one (E5 and the BGE query instruction).
6. Combine the chunk vectors with a token-count-weighted average.
7. L2-normalise the vector for sentence and retrieval models, so Euclidean distance stays between 0 and 2.

**BM25**

BM25 normally scores a query against a document. Here the app builds a vocabulary from the query and pages, applies BM25 term weights, and compares the resulting vectors with cosine and Euclidean measures.

## Crawling a URL (Markdown mode)

1. Enter a URL for your page or a competitor.
2. Pick a **Crawl mode**: **Markdown (MD)** keeps headings, lists and tables. **Plain text** returns clean text only.
3. Click **Crawl URL**. The page content fills the "Paste content" box.
4. Review or edit the text, then run the analysis.

What the crawler does:

- Removes scripts, styles, navigation, footers and sidebars
- Uses the `<main>` or `<article>` area when there is one
- Drops link URLs and images, and keeps the link text
- Follows up to 5 redirects
- Blocks localhost and private network addresses

Limits:

- It does not run JavaScript. Pages that render content client-side may come back empty. Paste the content instead
- Some sites block automated requests
- Pages over 5 MB are cut off

## Project structure

```text
querymatch-pro/ (content-query-similarity-analyzer/)
├── app.py                  # Streamlit UI
├── analysis_engine.py      # Model registry, BM25, embeddings, matrices
├── crawler.py              # URL -> Markdown / plain text
├── visualizations.py       # Plotly heatmaps
├── requirements.txt
├── requirements-dev.txt    # adds pytest
├── pytest.ini
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── tests/
    ├── test_analysis_engine.py
    └── test_crawler.py
```

## Run locally

Python 3.10+ is required.

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

One test uses a fake model to check chunking, pooling and prefixes. It needs `torch` and is skipped if torch is not installed. No test downloads a model or needs internet.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload all files.
2. Go to Streamlit Community Cloud and select the repository.
3. Set the main file to `app.py`.
4. Deploy.

Community Cloud has limited RAM. Use the small models (MiniLM, BGE Small, GTE Small) there. The base-size models may run out of memory.

## Reading the output

- **Cosine similarity:** higher means closer to the target query
- **Euclidean distance:** lower means closer. Compare values within one run only
- **Google rank:** context only. A page can rank #1 and still score lower here, because Google also uses links, intent, page quality, freshness, site signals and SERP features
- Use the results to find content gaps. They are not a ranking score

## Ideas for next

- Best-passage similarity alongside whole-page similarity
- Query vs heading / body / FAQ sections
- JavaScript-rendered crawling (headless browser)
- Keyword and entity overlap
- Scatter plot of Google rank vs query similarity
- PNG export of the heatmaps

## Author

Built by **Amal Alexander**
Email: [amalalex95@gmail.com](mailto:amalalex95@gmail.com)
Site: [amal-alexander.in](https://amal-alexander.in)
