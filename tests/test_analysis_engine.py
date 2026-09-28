import numpy as np

from analysis_engine import build_bm25_vectors, compute_matrices


def test_bm25_vector_shape_and_query_similarity():
    query = "best casino games"
    docs = [
        "best casino games include roulette blackjack and slots",
        "football scores fixtures teams league tables",
    ]
    vectors = build_bm25_vectors(query, docs)
    cosine, euclidean = compute_matrices(vectors)

    assert vectors.shape[0] == 3
    assert cosine.shape == (3, 3)
    assert euclidean.shape == (3, 3)
    assert np.allclose(np.diag(cosine), 1.0)
    assert np.allclose(np.diag(euclidean), 0.0)
    assert cosine[0, 1] > cosine[0, 2]
    assert euclidean[0, 1] < euclidean[0, 2]


def test_model_registry_is_complete():
    from analysis_engine import BM25_LABEL, MODEL_MAP, MODEL_SPECS

    assert len(MODEL_SPECS) >= 10
    assert BM25_LABEL not in MODEL_SPECS
    assert set(MODEL_MAP) == set(MODEL_SPECS)
    for label, spec in MODEL_SPECS.items():
        assert spec.model_id, label
        assert spec.pooling in ("mean", "cls"), label
        assert spec.chunk_size >= 8, label


def test_prefixes_and_pooling_for_special_models():
    from analysis_engine import MODEL_SPECS

    e5 = MODEL_SPECS["Multilingual E5 Small"]
    assert e5.query_prefix == "query: " and e5.document_prefix == "passage: "
    assert MODEL_SPECS["BGE Small EN v1.5"].pooling == "cls"


def test_embed_long_text_with_fake_model():
    torch = __import__("pytest").importorskip("torch")
    from types import SimpleNamespace

    from analysis_engine import ModelSpec, build_bert_vectors, embed_long_text

    class FakeTokenizer:
        cls_token_id, sep_token_id = 1, 2

        def encode(self, text, add_special_tokens=False):
            return [3 + (sum(map(ord, w)) % 90) for w in text.split()]

        def num_special_tokens_to_add(self, pair=False):
            return 2

    class FakeModel:
        config = SimpleNamespace(max_position_embeddings=64)

        def __init__(self):
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(200, 8)

        def __call__(self, input_ids, attention_mask, token_type_ids=None):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    tok, model = FakeTokenizer(), FakeModel()
    long_text = " ".join(f"word{i}" for i in range(200))  # forces several chunks

    mean_vec = embed_long_text(long_text, tok, model, chunk_size=30)
    cls_vec = embed_long_text(long_text, tok, model, chunk_size=30, pooling="cls")
    norm_vec = embed_long_text(long_text, tok, model, chunk_size=30, normalize=True)
    prefixed = embed_long_text("hello world", tok, model, prefix="query: ")
    plain = embed_long_text("hello world", tok, model)

    assert mean_vec.shape == (8,)
    assert not np.allclose(mean_vec, cls_vec)
    assert np.isclose(np.linalg.norm(norm_vec), 1.0, atol=1e-5)
    assert not np.allclose(prefixed, plain)

    spec = ModelSpec(model_id="fake", description="", pooling="cls", normalize=True, query_prefix="q ")
    vectors = build_bert_vectors("hello", ["doc one", "doc two"], tok, model, spec)
    assert vectors.shape == (3, 8)


def test_every_model_has_info_for_models_tab():
    from analysis_engine import MODEL_SPECS

    for label, spec in MODEL_SPECS.items():
        assert spec.kind and spec.languages and spec.size and spec.speed and spec.best_for, label


def test_generate_dynamic_analysis():
    import pandas as pd
    from analysis_engine import generate_dynamic_analysis

    df = pd.DataFrame(
        [
            {
                "Page": "Your Page",
                "URL": "https://mysite.com",
                "Google Rank": "3",
                "Cosine Similarity to Query": 0.8123,
                "Euclidean Distance to Query": 0.6127,
                "Cosine Similarity Order": 2,
                "Euclidean Distance Order": 2,
            },
            {
                "Page": "Competitor 1",
                "URL": "https://comp1.com",
                "Google Rank": "1",
                "Cosine Similarity to Query": 0.8845,
                "Euclidean Distance to Query": 0.4806,
                "Cosine Similarity Order": 1,
                "Euclidean Distance Order": 1,
            },
            {
                "Page": "Competitor 2",
                "URL": "https://comp2.com",
                "Google Rank": "2",
                "Cosine Similarity to Query": 0.5401,
                "Euclidean Distance to Query": 0.9591,
                "Cosine Similarity Order": 3,
                "Euclidean Distance Order": 3,
            },
        ]
    )
    cos = np.array(
        [
            [1.0, 0.8123, 0.8845, 0.5401],
            [0.8123, 1.0, 0.8200, 0.5100],
            [0.8845, 0.8200, 1.0, 0.5900],
            [0.5401, 0.5100, 0.5900, 1.0],
        ]
    )
    euc = np.array(
        [
            [0.0, 0.6127, 0.4806, 0.9591],
            [0.6127, 0.0, 0.6000, 0.9899],
            [0.4806, 0.6000, 0.0, 0.9055],
            [0.9591, 0.9899, 0.9055, 0.0],
        ]
    )
    labels = ["Target Query", "Me: mysite.com | G:3", "C1: comp1.com | G:1", "C2: comp2.com | G:2"]

    res = generate_dynamic_analysis(df, cos, euc, labels, "MiniLM L6 v2", "best shoes")
    assert res["winner"]["page"] == "Competitor 1"
    assert not res["winner"]["is_your_page"]
    assert res["your_page"]["semantic_rank"] == 2
    assert len(res["page_diagnostics"]) == 3
    assert len(res["recommendations"]) >= 2
    assert "verdict" in res["serp_analysis"]

