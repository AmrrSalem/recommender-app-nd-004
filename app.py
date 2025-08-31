from typing import List, Dict, Any, Tuple
import os
import random
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# FastHTML import with fallback
try:
    from fasthtml.common import FastHTML, Div, H1, H2, P, Form, Input, Button, Ul, Li, Style, Raw
except Exception:
    class _El:
        def __init__(self, tag: str, content: str = "", **attrs: Any) -> None:
            self.tag = tag
            self.content = content
            self.attrs = attrs

        def __call__(self, *children: Any, **attrs: Any) -> "_El":
            if attrs:
                self.attrs.update(attrs)
            self.content += "".join(c if isinstance(c, str) else c.render() for c in children)
            return self

        def render(self) -> str:
            pairs = []
            for k, v in self.attrs.items():
                if k == "cls":
                    k = "class"
                elif k.endswith("_"):
                    k = k[:-1]
                pairs.append(f'{k}="{v}"')
            attrs = (" " + " ".join(pairs)) if pairs else ""
            return f"<{self.tag}{attrs}>{self.content}</{self.tag}>"


    class FastHTML:
        def __call__(self, *children: Any) -> str:
            return "".join(c if isinstance(c, str) else c.render() for c in children)


    Div = lambda **a: _El("div", **a)
    H1 = lambda s, **a: _El("h1", s, **a)
    H2 = lambda s, **a: _El("h2", s, **a)
    P = lambda s, **a: _El("p", s, **a)
    Ul = lambda **a: _El("ul", **a)
    Li = lambda s="", **a: _El("li", s, **a)


    def Form(**a: Any) -> _El:
        a.setdefault("method", "get")
        return _El("form", **a)


    def Input(**a: Any) -> _El:
        a.setdefault("type", "text")
        return _El("input", **a)


    def Button(s: str, **a: Any) -> _El:
        a.setdefault("type", "submit")
        return _El("button", s, **a)


    def Style(css: str) -> _El:
        return _El("style", css)


    def Raw(html: str) -> str:
        return html

# Configuration
DATA_DIR = "data"
INTERACTIONS_PATH = f"{DATA_DIR}/interactions.csv"
SVD_COMPONENTS = int(os.getenv("SVD_COMPONENTS", "32"))
TFIDF_MAX_FEATS = int(os.getenv("TFIDF_MAX_FEATS", "6000"))
K_CHOICES = tuple(int(x) for x in os.getenv("K_CHOICES", "8,12").split(",") if x.strip())


# Data loading
def load_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("Unnamed: 0", "index"):
        if col in df.columns:
            df = df.drop(columns=col)
    if "user_id" not in df.columns:
        if "email" in df.columns:
            df = df.rename(columns={"email": "user_id"})
        else:
            raise KeyError("CSV must have 'user_id' or 'email'.")
    if "article_id" not in df.columns:
        aliases = ("content_id", "item_id", "aid", "doc_id")
        for a in aliases:
            if a in df.columns:
                df = df.rename(columns={a: "article_id"})
                break
        if "article_id" not in df.columns:
            raise KeyError("CSV must include an item column (e.g., 'article_id').")
    if "title" not in df.columns:
        df["title"] = ""
    if df["user_id"].dtype.kind not in "iu":
        df["user_id"] = pd.factorize(df["user_id"].astype(str))[0]
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["article_id"] = pd.to_numeric(df["article_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["user_id", "article_id"]).astype({"user_id": int, "article_id": int})
    return df[["user_id", "article_id", "title"]]


def build_articles_from_interactions(inter_df: pd.DataFrame) -> pd.DataFrame:
    def pick_title(series: pd.Series) -> str:
        s = series.dropna().astype(str)
        s = s[s.str.strip() != ""]
        if s.empty:
            return ""
        mode = s.mode()
        return mode.iat[0] if not mode.empty else s.iloc[0]

    titles = inter_df.groupby("article_id")["title"].apply(pick_title).reset_index()
    titles["title"] = titles.apply(
        lambda r: r["title"] if str(r["title"]).strip() != "" else f"Article {int(r['article_id'])}",
        axis=1,
    )
    titles["text"] = titles["title"]
    return titles.astype({"article_id": int})[["article_id", "title", "text"]]


def compute_metrics_overview(inter_df: pd.DataFrame, art_df: pd.DataFrame) -> Dict[str, int]:
    user_article_interactions = len(inter_df)
    views_per_user = inter_df.groupby("user_id")["article_id"].nunique()
    views_per_item = inter_df.groupby("article_id")["user_id"].count()
    return {
        "median_val": int(views_per_user.median()),
        "user_article_interactions": int(user_article_interactions),
        "max_views_by_user": int(views_per_user.max()),
        "max_views": int(views_per_item.max()),
        "most_viewed_article_id": int(views_per_item.idxmax()),
        "unique_articles": int(inter_df["article_id"].nunique()),
        "unique_users": int(inter_df["user_id"].nunique()),
        "total_articles": int(art_df["article_id"].nunique()),
    }


def get_top_article_ids(inter_df: pd.DataFrame, n: int = 10) -> List[int]:
    counts = inter_df.groupby("article_id")["user_id"].count().sort_values(ascending=False).head(n).index
    return [int(i) for i in counts]


def get_top_article_names(art_df: pd.DataFrame, inter_df: pd.DataFrame, n: int = 10) -> List[str]:
    ids = get_top_article_ids(inter_df, n)
    mapping = art_df.set_index("article_id")
    return [str(mapping.loc[i]["title"]) if i in mapping.index else f"Article {i}" for i in ids]


# Helpers
def _to1d(x: Any) -> np.ndarray:
    if hasattr(x, "A"):
        return np.asarray(x.A).ravel()
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


# Recommenders
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


class CFRecommender:
    def __init__(self, interactions: pd.DataFrame) -> None:
        self.interactions = interactions
        self.user_ids = np.array(sorted(interactions["user_id"].unique()))
        self.item_ids = np.array(sorted(interactions["article_id"].unique()))
        self.u_index = {u: i for i, u in enumerate(self.user_ids)}
        self.i_index = {i: j for j, i in enumerate(self.item_ids)}
        self._ui: csr_matrix | None = None
        self.popular = interactions["article_id"].value_counts()

    def _build_ui(self, interactions: pd.DataFrame) -> None:
        rows = interactions["user_id"].map(self.u_index).to_numpy()
        cols = interactions["article_id"].map(self.i_index).to_numpy()
        data = np.ones(len(interactions), dtype=np.float32)
        self._ui = csr_matrix((data, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids))).astype(bool)

    def recommend_for_user(self, user_id: int, n: int) -> List[int]:
        if self._ui is None:
            self._build_ui(self.interactions)
        if user_id not in self.u_index:
            return []
        u = self.u_index[user_id]
        ui = self._ui.tocsr()
        urow = ui[u].astype(np.float32)
        sims = cosine_similarity(urow, ui).ravel()
        sims[u] = 0.0
        neigh_idx = np.argsort(-sims)[:20]
        cand = _to1d(ui[neigh_idx].sum(axis=0)) > 0
        seen = _to1d(ui[u]).astype(bool)
        mask = cand & (~seen)
        scores = mask.astype(np.float32)
        pop = np.zeros_like(scores)
        for aid, cnt in self.popular.items():
            j = self.i_index.get(aid)
            if j is not None:
                pop[j] = cnt
        final = scores * 1.0 + (pop / (pop.max() or 1.0)) * 0.1
        topj = np.argsort(-final)[:n]
        return [int(self.item_ids[j]) for j in topj if final[j] > 0][:n]


class ContentRecommender:
    def __init__(self, articles: pd.DataFrame) -> None:
        self.articles = articles.reset_index(drop=True)
        self.id2row = {int(r.article_id): i for i, r in self.articles.iterrows()}
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=TFIDF_MAX_FEATS)
        self.X = None
        self.kmeans: KMeans | None = None

    def _fit_if_needed(self) -> None:
        if self.X is not None:
            return
        self.X = self.vec.fit_transform(self.articles["text"].fillna(""))
        best: tuple[float, KMeans] | None = None
        for k in K_CHOICES:
            km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(self.X)
            if best is None or km.inertia_ < best[0]:
                best = (km.inertia_, km)
        self.kmeans = best[1] if best else None

    def similar(self, article_id: int, n: int) -> List[int]:
        self._fit_if_needed()
        if article_id not in self.id2row:
            return []
        i = self.id2row[article_id]
        sims = cosine_similarity(self.X[i], self.X).ravel()
        sims[i] = 0.0
        idx = np.argsort(-sims)[:n]
        return [int(self.articles.loc[j, "article_id"]) for j in idx]


class SVDRecommender:
    def __init__(self, interactions: pd.DataFrame, n_components: int = SVD_COMPONENTS) -> None:
        self.interactions = interactions
        self.n_components = n_components
        self.user_ids = np.array(sorted(interactions["user_id"].unique()))
        self.item_ids = np.array(sorted(interactions["article_id"].unique()))
        self.u_index = {u: i for i, u in enumerate(self.user_ids)}
        self.i_index = {i: j for j, i in enumerate(self.item_ids)}
        self._ui: csr_matrix | None = None
        self._emb: np.ndarray | None = None
        self._svd = TruncatedSVD(n_components=n_components, random_state=0)

    def _fit_if_needed(self) -> None:
        if self._emb is not None:
            return
        rows = self.interactions["user_id"].map(self.u_index).to_numpy()
        cols = self.interactions["article_id"].map(self.i_index).to_numpy()
        data = np.ones(len(self.interactions), dtype=np.float32)
        self._ui = csr_matrix((data, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids)))
        self._svd.fit(self._ui)
        self._emb = self._svd.components_.T

    def similar(self, article_id: int, n: int) -> List[int]:
        self._fit_if_needed()
        j = self.i_index.get(article_id)
        if j is None:
            return []
        sims = cosine_similarity(self._emb[j: j + 1], self._emb).ravel()
        sims[j] = 0.0
        idx = np.argsort(-sims)[:n]
        return [int(self.item_ids[k]) for k in idx]


# App setup
app = FastAPI(title="Minimal Recommender")
html = FastHTML()
INTERACTIONS = load_interactions(INTERACTIONS_PATH)
ARTICLES = build_articles_from_interactions(INTERACTIONS)
USER_IDS = set(int(u) for u in INTERACTIONS["user_id"].unique())
USER_MIN = int(INTERACTIONS["user_id"].min())
USER_MAX = int(INTERACTIONS["user_id"].max())


def _titles_for(ids: List[int]) -> List[str]:
    mapping = ARTICLES.drop_duplicates("article_id").set_index("article_id")
    return [str(mapping.at[i, "title"]) if i in mapping.index else f"Article {i}" for i in ids]


# Evaluation utilities
def precision_recall_f1_at_k(recs: List[int], truth: List[int], k: int) -> Tuple[float, float, float]:
    if k <= 0:
        return 0.0, 0.0, 0.0
    truth_set = set(truth)
    recs_k = recs[:k]
    tp = sum(1 for r in recs_k if r in truth_set)
    precision = tp / max(1, k)
    recall = tp / max(1, len(truth_set))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def map_at_k(recs: List[int], truth: List[int], k: int) -> float:
    truth_set = set(truth)
    score = 0.0
    hits = 0
    for i, r in enumerate(recs[:k], start=1):
        if r in truth_set:
            hits += 1
            score += hits / i
    denom = min(len(truth_set), k)
    return 0.0 if denom == 0 else score / denom


def ndcg_at_k(recs: List[int], truth: List[int], k: int) -> float:
    truth_set = set(truth)
    dcg = 0.0
    for i, r in enumerate(recs[:k], start=1):
        rel = 1.0 if r in truth_set else 0.0
        if rel:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(truth_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def item_coverage(recs_lists: List[List[int]], all_items: List[int]) -> float:
    catalog = set(all_items)
    recommended = set(i for lst in recs_lists for i in lst)
    return 0.0 if not catalog else len(recommended & catalog) / len(catalog)


# UI styling
APP_CSS = """
:root{
  --bg:#fff; --fg:#111827; --muted:#6b7280; --card:#fff; --border:#e5e7eb;
  --brand:#2563eb; --brand-600:#1d4ed8;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font-family:ui-sans-serif,sssystem-ui,Segoe UI,Roboto,Arial}
.header{background:linear-gradient(135deg,var(--brand),var(--brand-600));
        color:#fff;padding:26px 18px}
.header h1{margin:0;font-size:32px}
.container{max-width:1100px;margin:24px auto;padding:0 16px}
.grid{display:grid;grid-template-columns:1fr;gap:18px}
@media(min-width:920px){.grid{grid-template-columns:1fr 1fr}}
.card{background:var(--card);border:1px solid var(--border);
      border-radius:12px;padding:16px 16px 10px}
.form-row{display:flex;gap:10px;margin:10px 0 8px}
.input{width:100%;padding:10px 12px;border:1px solid var(--border);
       border-radius:10px;background:transparent;color:var(--fg)}
.btn{padding:10px 14px;border-radius:10px;border:1px solid transparent;
     cursor:pointer;font-weight:700;background:var(--brand);color:#fff}
.btn:hover{background:var(--brand-600)}
.result-title{margin:10px 0 6px;font-weight:700}
.result-list{margin:6px 0 0;padding-left:18px;line-height:1.5}
.tip{color:var(--muted);margin:16px 0 42px}
.footer{color:var(--muted);font-size:12px;margin:28px 0 20px;text-align:center}
"""


# Routes
@app.get("/", response_class=HTMLResponse)
def home() -> str:
    overview = compute_metrics_overview(INTERACTIONS, ARTICLES)
    metric_items = Ul()(
        Li(f"Median interactions per user: {overview['median_val']}"),
        Li(f"Total user–article interactions: {overview['user_article_interactions']}"),
        Li(f"Max views by a user: {overview['max_views_by_user']}"),
        Li(f"Max views for an article: {overview['max_views']}"),
        Li(f"Most viewed article ID: {overview['most_viewed_article_id']}"),
        Li(f"Unique articles: {overview['unique_articles']}"),
        Li(f"Unique users: {overview['unique_users']}"),
        Li(f"Total articles: {overview['total_articles']}"),
    )
    user_form = Form(action="/recommend/user", method="get", id="userForm")(
        P("Enter user ID for recommendations:"),
        Div(cls="form-row")(
            Input(name="user_id", placeholder=f"user_id ({USER_MIN}–{USER_MAX})", cls="input"),
            Input(name="n", placeholder="e.g., 10", cls="input"),
            Button("Recommend", cls="btn"),
        ),
        Div(cls="result-title")("Recommendations"),
        Div(id="userPretty")("No results yet."),
    )
    art_form = Form(action="/recommend/article", method="get", id="artForm")(
        P("Find similar articles by ID:"),
        Div(cls="form-row")(
            Input(name="article_id", placeholder="e.g., 1430", cls="input"),
            Input(name="n", placeholder="e.g., 10", cls="input"),
            Button("Find Similar", cls="btn"),
        ),
        Div(cls="result-title")("Similar Articles"),
        Div(id="artPretty")("No results yet."),
    )
    eval_form = Form(action="/eval", method="get", id="evalForm")(
        P("Run offline evaluation:"),
        Div(cls="form-row")(
            Input(name="k", placeholder="k (e.g., 10)", cls="input"),
            Input(name="sample", placeholder="users to sample (e.g., 500)", cls="input"),
            Button("Run Eval", cls="btn"),
        ),
        Div(cls="result-title")("Metrics"),
        Div(id="evalPretty")("Not run yet."),
    )
    script = Raw(
        """
    <script>
      function L(items){
        return '<ul class="result-list">' +
               items.map(x=>'<li>'+x+'</li>').join('') +
               '</ul>';
      }
      function load(el){ el.textContent='Loading…'; }
      function renderUser(id, d){
        const el = document.getElementById(id); if(!el) return;
        if(d && d.error){ el.textContent = d.message || 'Error.'; return; }
        const names = (d.titles||[]).filter(Boolean);
        const ids = (d.article_ids||[]).map(String);
        const items = names.length ? names : ids;
        let html = '';
        if(d.strategy) html += '<div><strong>Strategy:</strong> '+d.strategy+'</div>';
        html += items.length ? L(items) : 'No recommendations found.';
        if(d.hint) html += '<div style="margin-top:8px;color:#6b7280;">'+d.hint+'</div>';
        el.innerHTML = html;
      }
      function renderArt(id, d){
        const el = document.getElementById(id); if(!el) return;
        if(d && d.error){ el.textContent = d.message || 'Error.'; return; }
        const ct = (d.content_based_titles||[]).filter(Boolean);
        const ci = ct.length ? ct : (d.content_based_ids||[]).map(String);
        const st = (d.svd_based_titles||[]).filter(Boolean);
        const si = st.length ? st : (d.svd_based_ids||[]).map(String);
        let html = '';
        html += '<div><strong>Content-based</strong>' + (ci.length?'':' — none') + '</div>' + (ci.length?L(ci):'');
        html += '<div style="margin-top:8px;"><strong>SVD-based</strong>' + (si.length?'':' — none') + '</div>' + (si.length?L(si):'');
        if(!ci.length && !si.length) html = 'No similar articles found.';
        el.innerHTML = html;
      }
      function renderEval(id, d){
        const el = document.getElementById(id); if(!el) return;
        if(d && d.error){ el.textContent = d.message || 'Error.'; return; }
        function row(name,m){
          return `<li><strong>${name}</strong> — `
                + `P@k: ${m.precision.toFixed(3)}, `
                + `R@k: ${m.recall.toFixed(3)}, `
                + `F1@k: ${m.f1.toFixed(3)}, `
                + `MAP@k: ${m.map.toFixed(3)}, `
                + `NDCG@k: ${m.ndcg.toFixed(3)}, `
                + `Coverage: ${m.coverage.toFixed(3)}</li>`;
        }
        let html = '<ul class="result-list">';
        html += row('Popularity', d.popularity);
        html += row('CF (user–user)', d.cf);
        html += row('Content (TF-IDF)', d.content);
        html += row('SVD (latent)', d.svd);
        html += '</ul>';
        html += `<div style="margin-top:8px;color:#6b7280;">Users evaluated: ${d.users_evaluated}, k=${d.k}</div>`;
        el.innerHTML = html;
      }
      async function hook(formId,outId,kind){
        const f = document.getElementById(formId),
              out = document.getElementById(outId);
        if(!f || !out) return;
        f.addEventListener('submit', async e=>{
          e.preventDefault();
          const url = f.action + '?' + new URLSearchParams(new FormData(f)).toString();
          load(out);
          try{
            const r = await fetch(url, {headers:{Accept:'application/json'}});
            const j = await r.json();
            (kind==='user'?renderUser:kind==='article'?renderArt:renderEval)(outId,j);
          }catch(err){
            out.textContent = 'Error: ' + String(err);
          }
        });
      }
      hook('userForm','userPretty','user');
      hook('artForm','artPretty','article');
      hook('evalForm','evalPretty','eval');
    </script>
    """
    )
    return html(
        Style(APP_CSS),
        Div(cls="header")(H1("Minimal Article Recommender")),
        Div(cls="container")(
            H2("Overview Metrics"),
            Div(cls="card")(metric_items),
            Div(cls="grid")(Div(cls="card")(user_form), Div(cls="card")(art_form)),
            Div(cls="card")(eval_form),
            Div(cls="tip")("Input: data/interactions.csv (articles & text derived in-memory)."),
            Div(cls="footer")("Local demo • FastAPI + FastHTML"),
        ),
        script,
    )


@app.get("/top", response_class=JSONResponse)
def api_top(n: int = Query(10, ge=1, le=50)) -> Dict[str, Any]:
    ids = get_top_article_ids(INTERACTIONS, n=n)
    names = get_top_article_names(ARTICLES, INTERACTIONS, n=n)
    return {"n": int(n), "article_ids": ids, "article_names": names}


@app.get("/recommend/user", response_class=JSONResponse)
def api_recommend_user(user_id: int = Query(..., ge=0), n: int = Query(10, ge=1, le=50)) -> Dict[str, Any]:
    try:
        if user_id not in USER_IDS:
            rec_ids = get_top_article_ids(INTERACTIONS, n=n)
            return {
                "user_id": int(user_id),
                "n": int(n),
                "strategy": "cold_start_popularity",
                "article_ids": rec_ids,
                "titles": _titles_for(rec_ids),
                "hint": f"Valid user_id range: {USER_MIN}–{USER_MAX}",
            }
        rec_ids = CFRecommender(INTERACTIONS).recommend_for_user(user_id=user_id, n=n) or []
        rec_ids = [int(x) for x in rec_ids]
        strategy = "user_user_cf" if rec_ids else "popularity_fallback"
        if not rec_ids:
            rec_ids = get_top_article_ids(INTERACTIONS, n=n)
        return {
            "user_id": int(user_id),
            "n": int(n),
            "strategy": strategy,
            "article_ids": rec_ids,
            "titles": _titles_for(rec_ids),
        }
    except Exception as exc:
        return {"error": True, "message": f"/recommend/user failed: {type(exc).__name__}: {exc}"}


@app.get("/recommend/article", response_class=JSONResponse)
def api_recommend_article(article_id: int = Query(..., ge=0), n: int = Query(10, ge=1, le=50)) -> Dict[str, Any]:
    try:
        content_ids = ContentRecommender(ARTICLES).similar(article_id=article_id, n=n)
        svd_ids = SVDRecommender(INTERACTIONS).similar(article_id=article_id, n=n)
        return {
            "article_id": int(article_id),
            "n": int(n),
            "content_based_ids": content_ids,
            "content_based_titles": _titles_for(content_ids),
            "svd_based_ids": svd_ids,
            "svd_based_titles": _titles_for(svd_ids),
        }
    except Exception as exc:
        return {"error": True, "message": f"/recommend/article failed: {type(exc).__name__}: {exc}"}


@app.get("/eval", response_class=JSONResponse)
def eval_offline(k: int = Query(10, ge=1, le=50), sample: int = Query(500, ge=10, le=5000)) -> Dict[str, Any]:
    rng = random.Random(42)
    inter = INTERACTIONS
    counts = inter.groupby("user_id")["article_id"].count()
    eligible = [int(u) for u, c in counts.items() if c >= 2]
    if not eligible:
        return {"error": True, "message": "No users with >=2 interactions."}
    if sample < len(eligible):
        eligible = rng.sample(eligible, sample)
    holdout: List[tuple[int, int]] = []
    rows_to_drop: List[int] = []
    by_user = inter.groupby("user_id")
    for u in eligible:
        items_u = list(by_user.get_group(u)["article_id"])
        ho = rng.choice(items_u)
        holdout.append((u, ho))
        idx = by_user.get_group(u).query("article_id == @ho").index[0]
        rows_to_drop.append(idx)
    train_df = inter.drop(index=rows_to_drop)
    articles_train = build_articles_from_interactions(train_df)
    pop_model_items = get_top_article_ids(train_df, n=k)
    cf = CFRecommender(train_df)
    content = ContentRecommender(articles_train)
    svd = SVDRecommender(train_df)
    all_items = list(articles_train["article_id"].unique())
    pop_lists: List[List[int]] = []
    cf_lists: List[List[int]] = []
    cont_lists: List[List[int]] = []
    svd_lists: List[List[int]] = []
    p_pop = r_pop = f_pop = map_pop = ndcg_pop = 0.0
    p_cf = r_cf = f_cf = map_cf = ndcg_cf = 0.0
    p_ct = r_ct = f_ct = map_ct = ndcg_ct = 0.0
    p_svd = r_svd = f_svd = map_svd = ndcg_svd = 0.0
    for u, ho in holdout:
        truth = [ho]
        rec_pop = pop_model_items[:k]
        pr, rc, f1 = precision_recall_f1_at_k(rec_pop, truth, k)
        p_pop += pr
        r_pop += rc
        f_pop += f1
        map_pop += map_at_k(rec_pop, truth, k)
        ndcg_pop += ndcg_at_k(rec_pop, truth, k)
        pop_lists.append(rec_pop)
        rec_cf = cf.recommend_for_user(u, k)
        pr, rc, f1 = precision_recall_f1_at_k(rec_cf, truth, k)
        p_cf += pr
        r_cf += rc
        f_cf += f1
        map_cf += map_at_k(rec_cf, truth, k)
        ndcg_cf += ndcg_at_k(rec_cf, truth, k)
        cf_lists.append(rec_cf)
        train_items_u = list(train_df[train_df.user_id == u]["article_id"].unique())
        anchor = rng.choice(train_items_u) if train_items_u else None
        if anchor is not None:
            rec_ct = content.similar(anchor, k)
            pr, rc, f1 = precision_recall_f1_at_k(rec_ct, truth, k)
            p_ct += pr
            r_ct += rc
            f_ct += f1
            map_ct += map_at_k(rec_ct, truth, k)
            ndcg_ct += ndcg_at_k(rec_ct, truth, k)
            cont_lists.append(rec_ct)
            rec_svd = svd.similar(anchor, k)
            pr, rc, f1 = precision_recall_f1_at_k(rec_svd, truth, k)
            p_svd += pr
            r_svd += rc
            f_svd += f1
            map_svd += map_at_k(rec_svd, truth, k)
            ndcg_svd += ndcg_at_k(rec_svd, truth, k)
            svd_lists.append(rec_svd)
        else:
            cont_lists.append([])
            svd_lists.append([])
    n_users = len(holdout)

    def pack(p: float, r: float, f: float, m: float, n_: float) -> Dict[str, float]:
        return {
            "precision": p / n_users,
            "recall": r / n_users,
            "f1": f / n_users,
            "map": m / n_users,
            "ndcg": n_ / n_users,
        }

    return {
        "k": int(k),
        "users_evaluated": int(n_users),
        "popularity": {**pack(p_pop, r_pop, f_pop, map_pop, ndcg_pop), "coverage": item_coverage(pop_lists, all_items)},
        "cf": {**pack(p_cf, r_cf, f_cf, map_cf, ndcg_cf), "coverage": item_coverage(cf_lists, all_items)},
        "content": {**pack(p_ct, r_ct, f_ct, map_ct, ndcg_ct), "coverage": item_coverage(cont_lists, all_items)},
        "svd": {**pack(p_svd, r_svd, f_svd, map_svd, ndcg_svd), "coverage": item_coverage(svd_lists, all_items)},
    }
