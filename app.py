from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from analysis_engine import (
    BM25_LABEL,
    MODEL_SPECS,
    build_bert_vectors,
    build_bm25_vectors,
    compute_matrices,
    generate_dynamic_analysis,
    load_bert,
)
from crawler import CrawlError, crawl_page
from visualizations import euclidean_heatmap, similarity_heatmap

st.set_page_config(
    page_title="QueryMatch Pro — Semantic Content & SERP Relevance Analyzer",
    page_icon="🎯",
    layout="wide",
)

RANK_OPTIONS = [str(i) for i in range(1, 11)] + ["Not in Page 1", "Unknown"]
MODEL_OPTIONS = [*MODEL_SPECS.keys(), BM25_LABEL]
CRAWL_MODES = ["Markdown (MD)", "Plain text"]

BM25_INFO = {
    "Model": BM25_LABEL,
    "Type": "Lexical (keywords)",
    "Languages": "Any",
    "Size": "None",
    "Speed": "Very fast",
    "Best for": "Exact keyword overlap. Does not understand meaning or synonyms.",
    "Hugging Face ID": "-",
}

AUTHOR_NAME = "Amal Alexander"
AUTHOR_EMAIL = "amalalex95@gmail.com"


# max_entries keeps memory in check when people try several models in one session.
@st.cache_resource(show_spinner=False, max_entries=2)
def get_cached_model(model_id: str):
    return load_bert(model_id)


def crawl_into_state(url_key: str, mode_key: str, content_key: str, status_key: str):
    """Button callback: crawl the URL and write the result into the content box.

    Runs before the script reruns, so it is allowed to set the text_area's session state.
    """
    url = st.session_state.get(url_key, "")
    mode = "markdown" if st.session_state.get(mode_key, CRAWL_MODES[0]).startswith("Markdown") else "text"
    try:
        result = crawl_page(url, mode=mode)
    except CrawlError as exc:
        st.session_state[status_key] = ("error", str(exc))
        return
    st.session_state[content_key] = result.content
    title = f" \u2013 {result.title}" if result.title else ""
    st.session_state[status_key] = (
        "success",
        f"Crawled {result.word_count:,} words{title}. Review the text below before analyzing.",
    )


def page_inputs(prefix: str, url_label: str, url_placeholder: str, rank_label: str,
                rank_index: int, content_label: str, content_placeholder: str, height: int):
    """Render URL, rank, crawl controls and the content box for one page."""
    url_key, mode_key = f"{prefix}_url", f"{prefix}_crawl_mode"
    content_key, status_key = f"{prefix}_content", f"{prefix}_crawl_status"

    left, right = st.columns([3, 1])
    url = left.text_input(url_label, placeholder=url_placeholder, key=url_key)
    rank = right.selectbox(rank_label, RANK_OPTIONS, index=rank_index, key=f"{prefix}_rank")

    mode_col, button_col, _ = st.columns([1.2, 1, 2])
    mode_col.selectbox("Crawl mode", CRAWL_MODES, key=mode_key)
    button_col.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    button_col.button(
        "Crawl URL",
        key=f"{prefix}_crawl",
        on_click=crawl_into_state,
        args=(url_key, mode_key, content_key, status_key),
        width="stretch",
    )

    status = st.session_state.get(status_key)
    if status:
        (st.success if status[0] == "success" else st.error)(status[1])

    content = st.text_area(content_label, height=height, placeholder=content_placeholder, key=content_key)
    return url.strip(), rank, content.strip()


def short_domain(url: str, fallback: str) -> str:
    if not url:
        return fallback
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.netloc.replace("www.", "")
    return host or fallback


def matrix_label(role: str, url: str, rank: str) -> str:
    domain = short_domain(url, role)
    return f"{role}: {domain} | G:{rank}"


def collect_pages(competitor_count: int):
    pages = []

    st.subheader("Your page")
    with st.container(border=True):
        url, rank, content = page_inputs(
            prefix="your",
            url_label="Your URL",
            url_placeholder="https://example.com/your-page",
            rank_label="Google rank",
            rank_index=len(RANK_OPTIONS) - 1,
            content_label="Paste your page content",
            content_placeholder="Paste the main text content from your page here, or crawl the URL above...",
            height=240,
        )
        pages.append({"role": "Me", "url": url, "rank": rank, "content": content})

    st.subheader("Competitors")
    for i in range(1, competitor_count + 1):
        with st.expander(f"Competitor {i}", expanded=i <= 3):
            url, rank, content = page_inputs(
                prefix=f"competitor_{i}",
                url_label=f"Competitor {i} URL",
                url_placeholder="https://competitor.com/page",
                rank_label=f"Competitor {i} Google rank",
                rank_index=min(i - 1, 9),
                content_label=f"Paste Competitor {i} content",
                content_placeholder="Paste the competitor page text here, or crawl the URL above...",
                height=220,
            )
            pages.append({"role": f"C{i}", "url": url, "rank": rank, "content": content})

    return pages


def models_tab():
    st.header("Models")
    st.write(
        "Pick the model in the Analyzer tab. Each model turns text into numbers in its own way, "
        "so scores are only comparable within one run and one model."
    )

    rows = [
        {
            "Model": label,
            "Type": spec.kind,
            "Languages": spec.languages,
            "Size": spec.size,
            "Speed": spec.speed,
            "Best for": spec.best_for,
            "Hugging Face ID": spec.model_id,
        }
        for label, spec in MODEL_SPECS.items()
    ]
    rows.append(BM25_INFO)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Which one should I run?")
    st.markdown(
        """
- **Quick check:** MiniLM L6 v2
- **Best English quality:** BGE Base or MPNet Base v2
- **Non-English pages:** Multilingual E5 Small or Multilingual MiniLM
- **Keyword overlap only:** BM25
- **Running on Streamlit Community Cloud:** stay with the small models (MiniLM, BGE Small, GTE Small)
        """
    )

    st.subheader("Model types")
    st.markdown(
        """
- **Plain BERT:** a general language model. It was not trained to compare texts, so scores sit close together. Use it as a baseline.
- **Sentence model:** trained so similar texts get similar vectors. Good for page-vs-query comparison.
- **Retrieval model:** trained to match a search query to a passage. Closest to how search relevance works.
- **BM25:** counts matching words, weighted by how rare they are. No download and no meaning.
        """
    )
    st.caption("The first run of each model downloads it from Hugging Face. After that it loads from cache.")


tab_analyzer, tab_models = st.tabs(["Analyzer", "Models"])

with tab_models:
    models_tab()

with tab_analyzer:
    st.title("🎯 QueryMatch Pro")
    st.caption(
        "Semantic Content & SERP Relevance Analyzer — Compare your page and competitor content against one target query using BERT, sentence-embedding models or BM25-style lexical vectors."
    )
    st.caption(f"Built by {AUTHOR_NAME} \u00b7 [{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL})")

    with st.container(border=True):
        col_query, col_model, col_count = st.columns([3, 2, 1])
        target_query = col_query.text_input(
            "Target query",
            placeholder="e.g. best online casino games",
            key="target_query",
        )
        model_choice = col_model.selectbox(
            "Comparison model",
            MODEL_OPTIONS,
            key="model_choice",
        )
        competitor_count = int(
            col_count.number_input(
                "Competitors",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="competitor_count",
            )
        )

        if model_choice != BM25_LABEL:
            spec = MODEL_SPECS[model_choice]
            st.info(
                f"{spec.description} Pages longer than the model limit are split into token chunks, "
                f"each chunk is pooled ({spec.pooling}), then combined into one page vector."
            )
        else:
            st.info(
                "BM25 is normally a retrieval scoring method, not an embedding model. "
                "This app converts BM25 term weights into lexical vectors so the same cosine and Euclidean matrices can be produced."
            )

    pages = collect_pages(competitor_count)

    analyze = st.button("Analyze content", type="primary", width="stretch")

    if analyze:
        errors = []
        if not target_query.strip():
            errors.append("Enter a target query.")

        empty_roles = [page["role"] for page in pages if not page["content"]]
        if empty_roles:
            errors.append(f"Paste content for: {', '.join(empty_roles)}.")

        if errors:
            for error in errors:
                st.error(error)
            st.stop()

        documents = [page["content"] for page in pages]

        try:
            if model_choice == BM25_LABEL:
                with st.spinner("Building BM25 lexical vectors..."):
                    vectors = build_bm25_vectors(target_query.strip(), documents)
            else:
                spec = MODEL_SPECS[model_choice]
                with st.spinner("Loading the selected model (first run downloads it) and analyzing content..."):
                    tokenizer, model = get_cached_model(spec.model_id)
                    vectors = build_bert_vectors(
                        target_query.strip(),
                        documents,
                        tokenizer,
                        model,
                        spec,
                    )

            cosine_matrix, euclidean_matrix = compute_matrices(vectors)

        except Exception as exc:
            st.exception(exc)
            st.stop()

        matrix_labels = ["Target Query"] + [
            matrix_label(page["role"], page["url"], page["rank"]) for page in pages
        ]

        result_rows = []
        for idx, page in enumerate(pages, start=1):
            result_rows.append(
                {
                    "Page": "Your Page" if page["role"] == "Me" else f"Competitor {page['role'][1:]}",
                    "URL": page["url"] or "Not provided",
                    "Google Rank": page["rank"],
                    "Cosine Similarity to Query": float(cosine_matrix[0, idx]),
                    "Euclidean Distance to Query": float(euclidean_matrix[0, idx]),
                }
            )

        result_df = pd.DataFrame(result_rows)
        result_df["Cosine Similarity Order"] = (
            result_df["Cosine Similarity to Query"].rank(method="min", ascending=False).astype(int)
        )
        result_df["Euclidean Distance Order"] = (
            result_df["Euclidean Distance to Query"].rank(method="min", ascending=True).astype(int)
        )
        result_df = result_df.sort_values(
            ["Cosine Similarity to Query", "Euclidean Distance to Query"],
            ascending=[False, True],
        ).reset_index(drop=True)

        st.divider()
        st.header("Query closeness summary")

        closest = result_df.iloc[0]
        your_row = result_df[result_df["Page"] == "Your Page"].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Closest page by cosine", closest["Page"])
        col2.metric("Your cosine similarity", f"{your_row['Cosine Similarity to Query']:.4f}")
        col3.metric("Your Euclidean distance", f"{your_row['Euclidean Distance to Query']:.4f}")

        display_df = result_df.copy()
        display_df["Cosine Similarity to Query"] = display_df["Cosine Similarity to Query"].round(4)
        display_df["Euclidean Distance to Query"] = display_df["Euclidean Distance to Query"].round(4)
        st.dataframe(display_df, width="stretch", hide_index=True)

        st.download_button(
            "Download results CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="querymatch_pro_similarity_results.csv",
            mime="text/csv",
        )

        st.header("1. Cosine similarity matrix")
        st.caption("Higher is closer. Focus on the Target Query row/column to compare pages against the query.")
        st.plotly_chart(
            similarity_heatmap(cosine_matrix, matrix_labels),
            width="stretch",
        )

        st.header("2. Euclidean distance matrix")
        st.caption("Lower is closer. A value of 0 on the diagonal means an item is being compared with itself.")
        st.plotly_chart(
            euclidean_heatmap(euclidean_matrix, matrix_labels),
            width="stretch",
        )

        st.divider()
        st.header("3. Results interpretation & Website performance analysis")
        st.caption(
            "Dynamic analysis showing which website is performing best for the target query, "
            "what the similarity and distance metrics mean, and actionable SEO takeaways."
        )

        insights = generate_dynamic_analysis(
            result_df,
            cosine_matrix,
            euclidean_matrix,
            matrix_labels,
            model_choice,
            target_query.strip(),
        )

        winner = insights["winner"]
        your_page = insights["your_page"]

        with st.container(border=True):
            b_col1, b_col2, b_col3 = st.columns([1.5, 1.5, 1])
            with b_col1:
                st.markdown(f"### 🏆 Top Performer: **{winner['page']}**")
                st.caption(f"**URL:** {winner['url'] or 'Not provided'} | **Google Rank:** {winner['google_rank']}")
                st.write(
                    f"**Cosine similarity:** `{winner['cosine']:.4f}` | **Euclidean distance:** `{winner['euclidean']:.4f}`"
                )
            with b_col2:
                st.markdown(f"### 📊 Your Standing: **Rank #{your_page['semantic_rank']} of {your_page['total_pages']}**")
                st.caption(f"**Status:** {your_page['status_badge']}")
                if your_page["is_winner"]:
                    st.write("Leading all analyzed pages in topical query relevance.")
                else:
                    st.write(
                        f"**Relevance gap:** `-{your_page['cosine_gap']:.4f}` ({your_page['pct_gap']:.1f}% behind winner)"
                    )
            with b_col3:
                st.markdown("### 🎯 Model")
                st.write(f"**{model_choice}**")
                st.caption(f"Query: *\"{target_query.strip()}\"*")

            if your_page["is_winner"]:
                st.success(f"**Topical Leader:** {your_page['status_summary']}")
            elif your_page["status_category"] == "competitive":
                st.info(f"**Close Contender:** {your_page['status_summary']}")
            else:
                st.warning(f"**Action Required:** {your_page['status_summary']}")

        st.subheader("Which website is performing best and why?")
        st.markdown(
            f"Based on vector representation, **{winner['page']}** is performing best for the target query "
            f"*\"{target_query.strip()}\"*. It achieved the highest cosine similarity (`{winner['cosine']:.4f}`) and lowest Euclidean distance "
            f"(`{winner['euclidean']:.4f}`). In search terms, this means its content vocabulary, semantic entities, and topical structure "
            f"directly answer the query's underlying intent with minimal off-topic drift."
        )

        for p in insights["page_diagnostics"]:
            with st.container(border=True):
                c_col1, c_col2 = st.columns([1, 2])
                with c_col1:
                    st.markdown(f"**{p['page']}** — `{p['tier']}`")
                    st.caption(f"**URL:** {p['url'] or 'Not provided'} | **Google Rank:** {p['google_rank']}")
                    st.write(f"Cosine: `{p['cosine']:.4f}` \u00b7 Euclidean: `{p['euclidean']:.4f}` \u00b7 Semantic Rank: `#{p['semantic_rank']}`")
                with c_col2:
                    st.write(f"**Why this score:** {p['reason']}")

        st.subheader("Google SERP rank vs. Semantic relevance")
        serp = insights["serp_analysis"]
        st.markdown(f"#### {serp['verdict']}")
        st.info(serp["narrative"])
        st.markdown(serp["your_standing_insight"])

        if insights["pairwise_overlap"]["best_pair"]:
            st.subheader("Competitor content overlap & pattern matching")
            pair = insights["pairwise_overlap"]["best_pair"]
            sim = insights["pairwise_overlap"]["similarity"]
            overlap_desc = (
                f"The two pages with the closest mutual content structure are **{pair[0]}** and **{pair[1]}** "
                f"with a **{sim * 100:.1f}% mutual similarity**. "
            )
            if sim >= 0.85:
                overlap_desc += (
                    "This high overlap indicates they follow nearly identical content outlines, headings, and keyword themes. "
                    "To beat them, look for topical differentiation and unique value rather than replicating the same outline."
                )
            else:
                overlap_desc += (
                    "This reflects moderate topical overlap, suggesting each competitor approaches the query with distinct sections or angles."
                )
            st.markdown(overlap_desc)

        with st.expander("📖 What do Cosine Similarity and Euclidean Distance mean here?"):
            spread = insights["score_spread"]
            st.markdown(
                f"""
- **Cosine Similarity (Topical Focus & Direction):**
  - Measures the *angular direction* of the page's content relative to the target query in high-dimensional vector space.
  - Scale: **0.0 (completely orthogonal/unrelated) to 1.0 (identical direction)**. Higher is better.
  - **In this run:** Scores range from `{spread['min_cosine']:.4f}` to `{spread['max_cosine']:.4f}`.
  - *Why it matters:* Cosine similarity is length-invariant. A 3,000-word comprehensive guide and a concise 400-word guide can both score high if their language stays laser-focused on the query.
- **Euclidean Distance (Straight-Line Proximity):**
  - Measures the *physical distance* between the query point and the page point in embedding space.
  - Scale: Lower is closer. **0.0 means identical**.
  - **In this run:** Distances range from `{spread['min_euclidean']:.4f}` (closest) to `{spread['max_euclidean']:.4f}` (furthest).
  - *Why it matters:* Confirms geometric proximity. When embeddings are normalized to unit length, Euclidean distance directly mirrors cosine similarity via $D = \\sqrt{{2(1 - \\cos)}}$.
- **Why Google Rank differs from Content Similarity:**
  - Google does **not** rank pages solely by textual similarity. Google's algorithm heavily weights **domain authority, high-quality backlinks, user engagement (CTR, dwell time), page experience (Core Web Vitals), and search intent formats (e.g. tools, reviews, video)**.
  - A competitor can rank #1 on Google with lower similarity if their off-page authority or brand trust compensates for lower textual specificity.
  - Use these matrices to detect **on-page content and topical gaps**, rather than treating them as an exact replica of Google's entire ranking system.
                """
            )

        st.subheader("💡 Dynamic SEO recommendations for your page")
        for rec in insights["recommendations"]:
            st.markdown(f"- {rec}")


st.divider()
st.caption(f"Built by {AUTHOR_NAME} \u00b7 [{AUTHOR_EMAIL}](mailto:{AUTHOR_EMAIL})")
