from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

BM25_LABEL = "BM25 (Lexical)"


@dataclass(frozen=True)
class ModelSpec:
    """How to turn text into a vector for one Hugging Face model."""

    model_id: str
    description: str
    pooling: str = "mean"  # "mean" or "cls"
    query_prefix: str = ""
    document_prefix: str = ""
    chunk_size: int = 480  # content tokens per chunk (before special tokens)
    normalize: bool = False  # L2-normalise final vectors (makes Euclidean meaningful)
    kind: str = "Sentence model"  # shown in the Models tab
    languages: str = "English"
    size: str = ""
    speed: str = "Medium"
    best_for: str = ""


# Order here is the order shown in the app dropdown.
MODEL_SPECS: Dict[str, ModelSpec] = {
    "BERT Base Uncased (English)": ModelSpec(
        model_id="google-bert/bert-base-uncased",
        kind="Plain BERT",
        languages="English",
        size="~440 MB",
        speed="Medium",
        best_for="Baseline only. Not trained for similarity, scores sit close together.",
        description="General BERT, English. Not trained for similarity, so scores are rough.",
    ),
    "BERT Base Multilingual Uncased": ModelSpec(
        model_id="google-bert/bert-base-multilingual-uncased",
        kind="Plain BERT",
        languages="100+",
        size="~670 MB",
        speed="Slow",
        best_for="Baseline for non-English. Same limitation as plain BERT.",
        description="General BERT, 100+ languages. Not trained for similarity, so scores are rough.",
    ),
    "MiniLM L6 v2 (all-MiniLM-L6-v2, fast)": ModelSpec(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        kind="Sentence model",
        languages="English",
        size="~90 MB",
        speed="Fast",
        best_for="Quick checks and first runs. Best default.",
        description="Small and fast sentence model, English. Best first choice for quick checks (~90 MB).",
        chunk_size=254,
        normalize=True,
    ),
    "MiniLM L12 v2 (all-MiniLM-L12-v2)": ModelSpec(
        model_id="sentence-transformers/all-MiniLM-L12-v2",
        kind="Sentence model",
        languages="English",
        size="~130 MB",
        speed="Fast",
        best_for="A bit better than L6 when you can wait a little longer.",
        description="Slightly deeper MiniLM, English. A bit better than L6, a bit slower (~130 MB).",
        chunk_size=254,
        normalize=True,
    ),
    "Multilingual MiniLM L12 (paraphrase-multilingual-MiniLM-L12-v2)": ModelSpec(
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        kind="Sentence model",
        languages="50+",
        size="~470 MB",
        speed="Medium",
        best_for="Small multilingual option. Reads only 128 tokens per chunk.",
        description="Small multilingual sentence model, 50+ languages (~470 MB).",
        chunk_size=126,
        normalize=True,
    ),
    "MPNet Base v2 (all-mpnet-base-v2)": ModelSpec(
        model_id="sentence-transformers/all-mpnet-base-v2",
        kind="Sentence model",
        languages="English",
        size="~420 MB",
        speed="Slow",
        best_for="Higher quality English comparison.",
        description="Higher-quality English sentence model, heavier (~420 MB).",
        chunk_size=382,
        normalize=True,
    ),
    "BGE Small EN v1.5": ModelSpec(
        model_id="BAAI/bge-small-en-v1.5",
        kind="Retrieval model",
        languages="English",
        size="~130 MB",
        speed="Fast",
        best_for="Query-vs-page relevance in English. Good quality for its size.",
        description="Retrieval-tuned English model, small (~130 MB). Uses the BGE query instruction.",
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
        normalize=True,
    ),
    "BGE Base EN v1.5": ModelSpec(
        model_id="BAAI/bge-base-en-v1.5",
        kind="Retrieval model",
        languages="English",
        size="~440 MB",
        speed="Slow",
        best_for="Best English relevance check in this list.",
        description="Retrieval-tuned English model, stronger than small (~440 MB).",
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
        normalize=True,
    ),
    "Multilingual E5 Small": ModelSpec(
        model_id="intfloat/multilingual-e5-small",
        kind="Retrieval model",
        languages="100",
        size="~470 MB",
        speed="Medium",
        best_for="Non-English queries and pages.",
        description="Retrieval-tuned multilingual model (~470 MB). Uses 'query:' / 'passage:' prefixes.",
        query_prefix="query: ",
        document_prefix="passage: ",
        normalize=True,
    ),
    "GTE Small": ModelSpec(
        model_id="thenlper/gte-small",
        kind="Embedding model",
        languages="English",
        size="~130 MB",
        speed="Fast",
        best_for="Small general-purpose alternative to MiniLM.",
        description="Small general-purpose English embedding model (~130 MB).",
        normalize=True,
    ),
}

# Kept for backwards compatibility: label -> Hugging Face model id.
MODEL_MAP: Dict[str, str] = {label: spec.model_id for label, spec in MODEL_SPECS.items()}

TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def lexical_tokenize(text: str) -> List[str]:
    """Simple Unicode-aware tokenizer for BM25-style lexical vectors."""
    return TOKEN_PATTERN.findall((text or "").lower())


def _bm25_idf(document_tokens: Sequence[Sequence[str]], vocabulary: Sequence[str]) -> Dict[str, float]:
    n_docs = max(len(document_tokens), 1)
    document_frequency: Counter[str] = Counter()

    for tokens in document_tokens:
        document_frequency.update(set(tokens))

    idf: Dict[str, float] = {}
    for term in vocabulary:
        df = document_frequency.get(term, 0)
        # Robertson/Sparck Jones style IDF with +1 inside log for stability.
        idf[term] = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
    return idf


def _bm25_weighted_vector(
    tokens: Sequence[str],
    vocabulary: Sequence[str],
    idf: Dict[str, float],
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    counts = Counter(tokens)
    doc_len = max(len(tokens), 1)
    avg_doc_len = max(avg_doc_len, 1.0)

    vector = np.zeros(len(vocabulary), dtype=np.float32)
    for idx, term in enumerate(vocabulary):
        tf = counts.get(term, 0)
        if tf == 0:
            continue
        denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
        vector[idx] = idf[term] * ((tf * (k1 + 1.0)) / denominator)
    return vector


def build_bm25_vectors(query: str, documents: Sequence[str]) -> np.ndarray:
    """
    Build comparable lexical vectors using BM25 term weighting.

    Row 0 is the target query. Remaining rows are page-content vectors.
    IDF statistics are calculated from the page documents, not from the query.
    This is intentionally a vectorized BM25 representation so cosine similarity
    and Euclidean distance can be calculated for the same set of inputs.
    """
    document_tokens = [lexical_tokenize(doc) for doc in documents]
    query_tokens = lexical_tokenize(query)

    vocab_set = set(query_tokens)
    for tokens in document_tokens:
        vocab_set.update(tokens)
    vocabulary = sorted(vocab_set)

    if not vocabulary:
        raise ValueError("No usable terms were found in the query or page content.")

    avg_doc_len = float(np.mean([len(tokens) for tokens in document_tokens])) if document_tokens else 1.0
    idf = _bm25_idf(document_tokens, vocabulary)

    all_tokens: List[Sequence[str]] = [query_tokens, *document_tokens]
    matrix = np.vstack(
        [
            _bm25_weighted_vector(tokens, vocabulary, idf, avg_doc_len)
            for tokens in all_tokens
        ]
    )
    return matrix


def load_bert(model_id: str):
    """Load tokenizer + encoder for a Hugging Face BERT model."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _prepare_chunk_inputs(tokenizer, token_ids: Sequence[int]):
    """Prepare an already-tokenized chunk across Hugging Face tokenizer implementations.

    Uses `prepare_for_model` when available to correctly add model-specific special
    tokens and token_type_ids. Avoids passing `return_special_tokens_mask=True` to
    `prepare_for_model` because fast tokenizers (Rust-backed PreTrainedTokenizerFast)
    assert `already_has_special_tokens=True`. Instead, the special tokens mask is
    retrieved after tokens are added.
    """
    import torch

    ids = list(token_ids)

    prepare_for_model = getattr(tokenizer, "prepare_for_model", None)
    if callable(prepare_for_model):
        try:
            encoded = prepare_for_model(
                ids,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=True,
            )

            # Ensure tensors have batch dimension of 1: shape (1, seq_len)
            input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            att_mask_val = encoded.get("attention_mask")
            if att_mask_val is not None:
                attention_mask = torch.as_tensor(att_mask_val, dtype=torch.long)
                if attention_mask.ndim == 1:
                    attention_mask = attention_mask.unsqueeze(0)
            else:
                attention_mask = torch.ones_like(input_ids)

            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            token_type_val = encoded.get("token_type_ids")
            if token_type_val is not None:
                tt_tensor = torch.as_tensor(token_type_val, dtype=torch.long)
                if tt_tensor.ndim == 1:
                    tt_tensor = tt_tensor.unsqueeze(0)
                model_inputs["token_type_ids"] = tt_tensor

            # Determine special tokens mask for mean-pooling exclusion
            get_special_tokens_mask = getattr(tokenizer, "get_special_tokens_mask", None)
            if callable(get_special_tokens_mask):
                try:
                    raw_ids = encoded["input_ids"]
                    if hasattr(raw_ids, "tolist"):
                        raw_ids = raw_ids.tolist()
                    special_mask = get_special_tokens_mask(
                        list(raw_ids), already_has_special_tokens=True
                    )
                    sm_tensor = torch.as_tensor(special_mask, dtype=torch.long)
                    if sm_tensor.ndim == 1:
                        sm_tensor = sm_tensor.unsqueeze(0)
                    pooling_mask = attention_mask * (1 - sm_tensor)
                    if pooling_mask.sum() == 0:
                        pooling_mask = attention_mask
                except Exception:
                    pooling_mask = attention_mask
            else:
                pooling_mask = attention_mask

            return model_inputs, pooling_mask
        except Exception:
            pass  # Fall back to manual assembly if prepare_for_model fails

    # Fallback for BERT-family / standard tokenizers if prepare_for_model is unavailable.
    cls_id = getattr(tokenizer, "cls_token_id", None)
    if cls_id is None:
        cls_id = getattr(tokenizer, "bos_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if sep_id is None:
        sep_id = getattr(tokenizer, "eos_token_id", None)

    prepared_ids = []
    special_mask = []
    if cls_id is not None:
        prepared_ids.append(int(cls_id))
        special_mask.append(1)

    prepared_ids.extend(ids)
    special_mask.extend([0] * len(ids))

    if sep_id is not None:
        prepared_ids.append(int(sep_id))
        special_mask.append(1)

    input_ids = torch.tensor([prepared_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    pooling_mask = attention_mask * (1 - torch.tensor([special_mask], dtype=torch.long))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }, pooling_mask


def _embed_token_chunk(tokenizer, model, token_ids: Sequence[int], pooling: str = "mean") -> np.ndarray:
    import torch

    model_inputs, pooling_mask = _prepare_chunk_inputs(tokenizer, token_ids)

    with torch.inference_mode():
        outputs = model(**model_inputs)
        if pooling == "cls":
            # First token ([CLS] / <s>) is the sentence vector for CLS-pooled models such as BGE.
            pooled = outputs.last_hidden_state[:, 0]
        else:
            pooled = _mean_pool(outputs.last_hidden_state, pooling_mask)

    return pooled.squeeze(0).cpu().numpy().astype(np.float32)


def embed_long_text(
    text: str,
    tokenizer,
    model,
    chunk_size: int = 480,
    pooling: str = "mean",
    prefix: str = "",
    normalize: bool = False,
) -> np.ndarray:
    """
    Embed text with a Hugging Face encoder. Long pages are token-chunked and then
    combined using a token-count-weighted average so content beyond the model
    limit is not silently discarded.

    pooling:   "mean" (mean of token vectors) or "cls" (first-token vector).
    prefix:    text prepended to every chunk (e.g. "query: " for E5, BGE instruction).
    normalize: L2-normalise the final vector.
    """
    if pooling not in ("mean", "cls"):
        raise ValueError("pooling must be 'mean' or 'cls'")

    token_ids = tokenizer.encode(text or "", add_special_tokens=False)
    if not token_ids:
        # Encode a harmless placeholder instead of returning a wrong-dimensional zero vector.
        token_ids = tokenizer.encode("empty", add_special_tokens=False)

    max_positions = getattr(model.config, "max_position_embeddings", 512)
    try:
        special_token_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    except (AttributeError, TypeError, ValueError):
        special_token_count = 2

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    safe_chunk_size = max(8, min(chunk_size, max_positions - special_token_count) - len(prefix_ids))

    chunk_vectors: List[np.ndarray] = []
    chunk_weights: List[int] = []

    for start in range(0, len(token_ids), safe_chunk_size):
        chunk = token_ids[start : start + safe_chunk_size]
        chunk_vectors.append(_embed_token_chunk(tokenizer, model, [*prefix_ids, *chunk], pooling))
        chunk_weights.append(len(chunk))

    stacked = np.vstack(chunk_vectors)
    weights = np.asarray(chunk_weights, dtype=np.float32)
    vector = np.average(stacked, axis=0, weights=weights).astype(np.float32)

    if normalize:
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = (vector / norm).astype(np.float32)
    return vector


def build_bert_vectors(
    query: str,
    documents: Sequence[str],
    tokenizer,
    model,
    spec: Optional[ModelSpec] = None,
) -> np.ndarray:
    """Build the query vector (row 0) and one vector per page, using the model's settings."""
    spec = spec or ModelSpec(model_id="", description="")
    options = dict(chunk_size=spec.chunk_size, pooling=spec.pooling, normalize=spec.normalize)

    vectors = [embed_long_text(query, tokenizer, model, prefix=spec.query_prefix, **options)]
    vectors.extend(
        embed_long_text(doc, tokenizer, model, prefix=spec.document_prefix, **options)
        for doc in documents
    )
    return np.vstack(vectors)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / norms
    matrix = normalized @ normalized.T
    return np.clip(matrix, -1.0, 1.0)


def euclidean_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    differences = vectors[:, None, :] - vectors[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2))


def compute_matrices(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return cosine_similarity_matrix(vectors), euclidean_distance_matrix(vectors)


def generate_dynamic_analysis(
    result_df: Any,
    cosine_matrix: np.ndarray,
    euclidean_matrix: np.ndarray,
    matrix_labels: Sequence[str],
    model_choice: str,
    target_query: str,
) -> Dict[str, Any]:
    """Generate dynamic, data-driven insights and diagnostic takeaways based on similarity results."""
    df = result_df.sort_values(
        ["Cosine Similarity to Query", "Euclidean Distance to Query"],
        ascending=[False, True],
    ).reset_index(drop=True)

    winner = df.iloc[0]
    your_rows = df[df["Page"] == "Your Page"]
    your_row = your_rows.iloc[0] if not your_rows.empty else winner

    winner_page = str(winner["Page"])
    winner_url = str(winner["URL"])
    winner_cos = float(winner["Cosine Similarity to Query"])
    winner_euc = float(winner["Euclidean Distance to Query"])
    winner_google = str(winner["Google Rank"])

    your_cos = float(your_row["Cosine Similarity to Query"])
    your_euc = float(your_row["Euclidean Distance to Query"])
    your_google = str(your_row["Google Rank"])
    your_semantic_rank = int(your_row["Cosine Similarity Order"])
    total_pages = len(df)
    is_your_page_winner = (your_row["Page"] == winner["Page"])

    cos_gap = float(winner_cos - your_cos)
    pct_gap = float((cos_gap / max(abs(winner_cos), 1e-6)) * 100.0)

    # Determine Your Page standing and status
    if is_your_page_winner:
        status_category = "leader"
        status_badge = "🏆 Top Performer"
        status_summary = (
            f"Your page has the highest topical alignment to '{target_query}' "
            f"among all {total_pages} analyzed pages, outperforming every competitor."
        )
    elif your_semantic_rank == 2 and cos_gap <= 0.03:
        status_category = "competitive"
        status_badge = "⚡ Strong Contender (Near-Tie)"
        status_summary = (
            f"Your page is closely trailing the top performer ({winner_page}) by only "
            f"{cos_gap:.4f} ({pct_gap:.1f}%). Minor topical adjustments could put your content in first place."
        )
    elif your_semantic_rank <= math.ceil(total_pages / 2):
        status_category = "moderate_gap"
        status_badge = "⚠️ Moderate Content Gap"
        status_summary = (
            f"Your page ranks #{your_semantic_rank} of {total_pages}. While your content shares relevance with "
            f"'{target_query}', {winner_page} has a clearer topical focus by a margin of {cos_gap:.4f}."
        )
    else:
        status_category = "large_gap"
        status_badge = "🚨 Significant Content Deficit"
        status_summary = (
            f"Your page ranks #{your_semantic_rank} of {total_pages}, trailing {winner_page} by {cos_gap:.4f} "
            f"({pct_gap:.1f}%). Competitors cover the target query far more comprehensively."
        )

    # Diagnostic for each page
    page_diagnostics = []
    for _, row in df.iterrows():
        p_name = str(row["Page"])
        p_url = str(row["URL"])
        p_rank = int(row["Cosine Similarity Order"])
        p_cos = float(row["Cosine Similarity to Query"])
        p_euc = float(row["Euclidean Distance to Query"])
        p_google = str(row["Google Rank"])

        if p_rank == 1:
            tier = "🏆 Best Performer"
            reason = (
                f"Highest semantic alignment (cosine: {p_cos:.4f}). Its vocabulary and topical structure "
                f"point closest to '{target_query}' with minimal off-topic drift."
            )
        elif p_cos >= 0.70:
            tier = "🟢 Strong Focus"
            reason = (
                f"Solid topical relevance (cosine: {p_cos:.4f}). Substantial coverage of '{target_query}', "
                f"closely following the leader."
            )
        elif p_cos >= 0.45:
            tier = "🟡 Moderate Focus"
            reason = (
                f"Partial topical coverage (cosine: {p_cos:.4f}). Likely covers broader themes or only "
                f"touches on '{target_query}' as a secondary section."
            )
        else:
            tier = "🔴 Diluted Focus"
            reason = (
                f"Low topical similarity (cosine: {p_cos:.4f}). Content diverges substantially from "
                f"'{target_query}', with high proportion of off-topic or generic material."
            )

        page_diagnostics.append({
            "page": p_name,
            "url": p_url,
            "semantic_rank": p_rank,
            "cosine": p_cos,
            "euclidean": p_euc,
            "google_rank": p_google,
            "tier": tier,
            "reason": reason,
        })

    # SERP vs Semantic Correlation
    numeric_ranks = []
    for _, row in df.iterrows():
        gr = str(row["Google Rank"]).strip()
        if gr.isdigit():
            numeric_ranks.append((int(gr), str(row["Page"]), float(row["Cosine Similarity to Query"]), int(row["Cosine Similarity Order"])))

    serp_analysis = {}
    if numeric_ranks:
        numeric_ranks.sort(key=lambda x: x[0])
        best_google_rank, best_google_page, best_google_cos, best_google_sem_rank = numeric_ranks[0]
        
        is_serp_winner_semantic_winner = (best_google_sem_rank == 1)
        serp_analysis["best_google_page"] = best_google_page
        serp_analysis["best_google_rank"] = best_google_rank
        serp_analysis["is_aligned"] = is_serp_winner_semantic_winner

        if is_serp_winner_semantic_winner:
            serp_analysis["verdict"] = "Direct Content-SERP Correlation"
            serp_analysis["narrative"] = (
                f"The page holding the best Google ranking ({best_google_page}, Rank #{best_google_rank}) "
                f"is ALSO the #1 page in semantic similarity ({winner_cos:.4f}). For '{target_query}', "
                f"Google search results strongly reward direct, comprehensive topical depth."
            )
        else:
            serp_analysis["verdict"] = "Authority / Search Intent Discrepancy"
            serp_analysis["narrative"] = (
                f"The highest-ranking Google page ({best_google_page}, Rank #{best_google_rank}) is NOT "
                f"the most semantically similar page (it ranks #{best_google_sem_rank} with cosine {best_google_cos:.4f}, "
                f"behind {winner_page} with {winner_cos:.4f}). This indicates that off-page authority "
                f"(backlink profile, domain age, brand recognition) or SERP intent features (e.g. interactive tools, product catalogs) "
                f"play a dominant role in Google's ranking decisions for this query."
            )

        # Your page specific SERP correlation
        your_gr_clean = str(your_google).strip()
        if your_gr_clean.isdigit():
            your_gr_num = int(your_gr_clean)
            if your_semantic_rank < your_gr_num:
                serp_analysis["your_standing_insight"] = (
                    f"⭐ **Content Advantage:** Your semantic relevance (Rank #{your_semantic_rank}) is stronger "
                    f"than your Google position (Rank #{your_gr_num}). Your content is already primed for higher rankings; "
                    f"your primary growth levers are off-page SEO, backlink acquisition, and user engagement signals."
                )
            elif your_semantic_rank > your_gr_num:
                serp_analysis["your_standing_insight"] = (
                    f"⚠️ **Topical Vulnerability:** Your current Google ranking (Rank #{your_gr_num}) outpaces your "
                    f"content similarity rank (#{your_semantic_rank}). Competitors have tighter topical alignment to "
                    f"'{target_query}'. Improving your content depth now will protect your rank against future Google core updates."
                )
            else:
                serp_analysis["your_standing_insight"] = (
                    f"⚖️ **Balanced Position:** Your semantic closeness (Rank #{your_semantic_rank}) matches your Google rank "
                    f"(Rank #{your_gr_num}). Strengthening your content's semantic alignment will directly support ranking higher."
                )
        else:
            serp_analysis["your_standing_insight"] = (
                f"Your page is ranked #{your_semantic_rank} in content similarity among the analyzed set."
            )
    else:
        serp_analysis["verdict"] = "Informational SERP Comparison"
        serp_analysis["narrative"] = "Google ranks were not specified as numbers; compare semantic orders directly."
        serp_analysis["your_standing_insight"] = (
            f"Your page ranks #{your_semantic_rank} in topical closeness out of {total_pages} analyzed pages."
        )

    # Cross-document pairwise analysis (competitor overlap)
    n_docs = cosine_matrix.shape[0]
    best_doc_pair = None
    best_doc_sim = -1.0
    for i in range(1, n_docs):
        for j in range(i + 1, n_docs):
            sim_val = float(cosine_matrix[i, j])
            if sim_val > best_doc_sim:
                best_doc_sim = sim_val
                label_i = matrix_labels[i] if i < len(matrix_labels) else f"Doc {i}"
                label_j = matrix_labels[j] if j < len(matrix_labels) else f"Doc {j}"
                best_doc_pair = (label_i, label_j)

    # Dynamic SEO recommendations
    recommendations = []
    if is_your_page_winner:
        recommendations.append(
            "**Maintain topical dominance:** Your page already leads in semantic relevance. "
            "Focus on satisfying search intent (clear UX, fast load times, structured data/FAQ schema) to convert this content advantage into SERP dominance."
        )
        recommendations.append(
            "**Defend against competitor expansion:** Audit second-place content periodically to ensure competitors "
            "do not add new subtopics or emerging query angles."
        )
    else:
        recommendations.append(
            f"**Benchmark against {winner_page}:** {winner_page} achieved a cosine similarity of {winner_cos:.4f}. "
            f"Inspect their headings, key terms, and FAQs to identify topical sub-themes missing from your page."
        )
        if cos_gap <= 0.05:
            recommendations.append(
                "**Close the near-tie gap:** Because you are within striking distance, small on-page refinements "
                "(e.g., adding explicit answers to query-related questions and tightening introductory paragraphs) "
                "can push your content ahead."
            )
        else:
            recommendations.append(
                "**Trim off-topic sections & enrich core entities:** Your score suggests either topical dilution "
                "(too much tangential text) or an incomplete discussion of the core search intent. "
                "Refocus your page sections tightly around the target query."
            )

    if model_choice == BM25_LABEL:
        recommendations.append(
            "**Note on BM25 Lexical Model:** BM25 strictly counts exact word matches. Try switching to a neural model "
            "like MiniLM L6 v2 or BGE Small to evaluate semantic meaning, synonyms, and natural language intent."
        )
    else:
        recommendations.append(
            f"**Model Context ({model_choice}):** Neural embedding models measure true semantic meaning and conceptual alignment, "
            f"not just keyword frequency. A higher score here means your text directly satisfies the query's underlying intent."
        )

    return {
        "winner": {
            "page": winner_page,
            "url": winner_url,
            "cosine": winner_cos,
            "euclidean": winner_euc,
            "google_rank": winner_google,
            "is_your_page": is_your_page_winner,
        },
        "your_page": {
            "cosine": your_cos,
            "euclidean": your_euc,
            "google_rank": your_google,
            "semantic_rank": your_semantic_rank,
            "total_pages": total_pages,
            "is_winner": is_your_page_winner,
            "cosine_gap": cos_gap,
            "pct_gap": pct_gap,
            "status_badge": status_badge,
            "status_summary": status_summary,
            "status_category": status_category,
        },
        "page_diagnostics": page_diagnostics,
        "serp_analysis": serp_analysis,
        "pairwise_overlap": {
            "best_pair": best_doc_pair,
            "similarity": best_doc_sim,
        },
        "recommendations": recommendations,
        "score_spread": {
            "min_cosine": float(df["Cosine Similarity to Query"].min()),
            "max_cosine": winner_cos,
            "min_euclidean": winner_euc,
            "max_euclidean": float(df["Euclidean Distance to Query"].max()),
        },
    }
