"""
src/recommender.py
------------------
Sistema de recomendação híbrido de músicas.

Abordagens implementadas
------------------------
1. Item-Based (Content-Based Filtering)
   - Vetoriza músicas usando TF-IDF sobre suas tags/gêneros
   - Calcula similaridade de cosseno entre itens
   - Recomenda músicas com tags mais parecidas

2. User-Based (Collaborative Filtering)
   - Constrói matriz usuário × música com ratings
   - Calcula similaridade entre usuários via cosseno
   - Recomenda músicas que usuários similares avaliaram bem

3. Híbrido
   - Combina os dois scores com peso configurável
   - alpha=1.0 → só Item-Based
   - alpha=0.0 → só User-Based
   - alpha=0.5 → peso igual (padrão)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from fetcher import MusicDataset

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Uma recomendação individual."""
    track_id: int
    title: str
    artist: str
    genre: str
    score: float           # score híbrido [0–1]
    item_score: float      # contribuição item-based
    user_score: float      # contribuição user-based
    popularity: float


class HybridRecommender:
    """
    Recomendador híbrido: Item-Based + User-Based.

    Uso típico
    ----------
    >>> rec = HybridRecommender(dataset)
    >>> rec.fit()

    # Por música (item-based)
    >>> recs = rec.recommend_by_track("Blinding Lights", "The Weeknd", n=10)

    # Por usuário (user-based)
    >>> recs = rec.recommend_by_user("user_001", n=10)

    # Híbrido
    >>> recs = rec.recommend_hybrid("user_001", "Blinding Lights", "The Weeknd")
    """

    def __init__(self, dataset: MusicDataset, alpha: float = 0.5) -> None:
        """
        Parâmetros
        ----------
        dataset : MusicDataset
        alpha   : float — peso do item-based (0.0–1.0)
                  0.5 = igual peso para ambos
        """
        self.dataset = dataset
        self.alpha = alpha
        self.tracks = dataset.tracks_df.copy()
        self.ratings = dataset.ratings_df.copy()
        self.tags = dataset.tags_df.copy()

        # Artefatos computados em fit()
        self._item_sim_matrix: np.ndarray | None = None
        self._user_sim_matrix: np.ndarray | None = None
        self._rating_matrix: pd.DataFrame | None = None
        self._tfidf_index: dict[int, int] = {}   # track_id → índice na matriz TF-IDF
        self._user_index: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self) -> "HybridRecommender":
        """Calcula matrizes de similaridade. Deve ser chamado antes de recommend."""
        logger.info("Treinando Item-Based (TF-IDF + cosine)...")
        self._fit_item_based()

        logger.info("Treinando User-Based (ratings matrix + cosine)...")
        self._fit_user_based()

        logger.info("Recomendador pronto.")
        return self

    def _fit_item_based(self) -> None:
        """TF-IDF sobre tags por música → similaridade de cosseno."""
        # Agrega tags por música em uma string de documento
        tag_docs = (
            self.tags.groupby("track_id")["tag"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )

        # Garante que todas as músicas estão presentes
        all_ids = self.tracks["track_id"].tolist()
        tag_docs = (
            self.tracks[["track_id"]]
            .merge(tag_docs, on="track_id", how="left")
        )
        tag_docs["tag"] = tag_docs["tag"].fillna("")

        # Adiciona genre/subgenre como fallback
        tag_docs = tag_docs.merge(
            self.tracks[["track_id", "genre", "subgenre"]],
            on="track_id", how="left",
        )
        tag_docs["doc"] = (
            tag_docs["tag"] + " " +
            tag_docs["genre"].fillna("") + " " +
            tag_docs["subgenre"].fillna("")
        ).str.strip()

        tfidf = TfidfVectorizer(token_pattern=r"[a-z0-9\-]+")
        tfidf_matrix = tfidf.fit_transform(tag_docs["doc"])

        self._item_sim_matrix = cosine_similarity(tfidf_matrix)
        self._tfidf_index = {
            tid: idx for idx, tid in enumerate(tag_docs["track_id"])
        }

    def _fit_user_based(self) -> None:
        """Matriz usuário × música com ratings → similaridade de cosseno."""
        rating_matrix = self.ratings.pivot_table(
            index="user_id",
            columns="track_id",
            values="rating",
            fill_value=0,
        )
        self._rating_matrix = rating_matrix
        self._user_index = {u: i for i, u in enumerate(rating_matrix.index)}

        sparse = csr_matrix(rating_matrix.values)
        self._user_sim_matrix = cosine_similarity(sparse)

    # ------------------------------------------------------------------
    # Item-Based
    # ------------------------------------------------------------------
    def recommend_by_track(
        self,
        title: str,
        artist: str,
        n: int = 10,
        exclude_same_artist: bool = False,
    ) -> list[Recommendation]:
        """Recomenda músicas similares a uma faixa (item-based)."""
        track_row = self.tracks[
            (self.tracks["title"].str.lower() == title.lower()) &
            (self.tracks["artist"].str.lower() == artist.lower())
        ]
        if track_row.empty:
            raise ValueError(f"Música '{title}' de '{artist}' não encontrada.")

        track_id = int(track_row["track_id"].values[0])
        idx = self._tfidf_index.get(track_id)
        if idx is None:
            raise ValueError("Música sem índice no modelo.")

        sim_scores = self._item_sim_matrix[idx]

        recs = []
        for tid, sim in zip(self._tfidf_index.keys(), sim_scores):
            if tid == track_id:
                continue
            row = self.tracks[self.tracks["track_id"] == tid]
            if row.empty:
                continue
            if exclude_same_artist and row["artist"].values[0].lower() == artist.lower():
                continue
            recs.append(Recommendation(
                track_id=tid,
                title=row["title"].values[0],
                artist=row["artist"].values[0],
                genre=row["genre"].values[0],
                score=round(float(sim), 4),
                item_score=round(float(sim), 4),
                user_score=0.0,
                popularity=float(row["popularity"].values[0]),
            ))

        recs.sort(key=lambda x: x.score, reverse=True)
        return recs[:n]

    # ------------------------------------------------------------------
    # User-Based
    # ------------------------------------------------------------------
    def recommend_by_user(
        self,
        user_id: str,
        n: int = 10,
    ) -> list[Recommendation]:
        """Recomenda músicas baseado em usuários similares."""
        if user_id not in self._user_index:
            raise ValueError(f"Usuário '{user_id}' não encontrado.")

        u_idx = self._user_index[user_id]
        sim_scores = self._user_sim_matrix[u_idx]

        # Usuários mais similares (excluindo o próprio)
        similar_users_idx = np.argsort(sim_scores)[::-1][1:11]
        similar_users = [
            list(self._user_index.keys())[i] for i in similar_users_idx
        ]
        user_weights = sim_scores[similar_users_idx]

        # Músicas já avaliadas pelo usuário
        rated_by_user = set(
            self.ratings[self.ratings["user_id"] == user_id]["track_id"]
        )

        # Score ponderado pela similaridade dos usuários similares
        scores: dict[int, float] = {}
        for sim_user, weight in zip(similar_users, user_weights):
            user_ratings = self.ratings[self.ratings["user_id"] == sim_user]
            for _, r in user_ratings.iterrows():
                tid = int(r["track_id"])
                if tid in rated_by_user:
                    continue
                scores[tid] = scores.get(tid, 0.0) + weight * r["rating"]

        if not scores:
            return []

        # Normaliza
        max_score = max(scores.values()) or 1
        recs = []
        for tid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]:
            row = self.tracks[self.tracks["track_id"] == tid]
            if row.empty:
                continue
            norm = round(score / max_score, 4)
            recs.append(Recommendation(
                track_id=tid,
                title=row["title"].values[0],
                artist=row["artist"].values[0],
                genre=row["genre"].values[0],
                score=norm,
                item_score=0.0,
                user_score=norm,
                popularity=float(row["popularity"].values[0]),
            ))

        return recs

    # ------------------------------------------------------------------
    # Híbrido
    # ------------------------------------------------------------------
    def recommend_hybrid(
        self,
        user_id: str,
        title: str,
        artist: str,
        n: int = 10,
        alpha: float | None = None,
    ) -> list[Recommendation]:
        """
        Combina item-based e user-based com peso alpha.

        score_final = alpha * item_score + (1 - alpha) * user_score
        """
        alpha = alpha if alpha is not None else self.alpha

        # Scores item-based
        try:
            item_recs = self.recommend_by_track(title, artist, n=len(self.tracks))
        except ValueError:
            item_recs = []
        item_scores = {r.track_id: r.item_score for r in item_recs}

        # Scores user-based
        try:
            user_recs = self.recommend_by_user(user_id, n=len(self.tracks))
        except ValueError:
            user_recs = []
        user_scores = {r.track_id: r.user_score for r in user_recs}

        # União de todos os track_ids candidatos
        all_ids = set(item_scores) | set(user_scores)

        # Músicas já avaliadas pelo usuário (excluir)
        rated = set(self.ratings[self.ratings["user_id"] == user_id]["track_id"])
        # Exclui a música de entrada
        seed_row = self.tracks[
            (self.tracks["title"].str.lower() == title.lower()) &
            (self.tracks["artist"].str.lower() == artist.lower())
        ]
        if not seed_row.empty:
            rated.add(int(seed_row["track_id"].values[0]))

        recs = []
        for tid in all_ids:
            if tid in rated:
                continue
            row = self.tracks[self.tracks["track_id"] == tid]
            if row.empty:
                continue

            i_score = item_scores.get(tid, 0.0)
            u_score = user_scores.get(tid, 0.0)
            hybrid = alpha * i_score + (1 - alpha) * u_score

            recs.append(Recommendation(
                track_id=tid,
                title=row["title"].values[0],
                artist=row["artist"].values[0],
                genre=row["genre"].values[0],
                score=round(hybrid, 4),
                item_score=round(i_score, 4),
                user_score=round(u_score, 4),
                popularity=float(row["popularity"].values[0]),
            ))

        recs.sort(key=lambda x: x.score, reverse=True)
        return recs[:n]

    # ------------------------------------------------------------------
    # Helpers para o dashboard
    # ------------------------------------------------------------------
    def get_all_tracks(self) -> pd.DataFrame:
        return self.tracks.copy()

    def get_all_users(self) -> list[str]:
        return sorted(self._user_index.keys())

    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Retorna histórico de ratings de um usuário."""
        rated = self.ratings[self.ratings["user_id"] == user_id].copy()
        return rated.merge(self.tracks, on="track_id", how="left") \
                    .sort_values("rating", ascending=False)

    def get_similarity_heatmap(self, n: int = 15) -> pd.DataFrame:
        """Retorna submatriz de similaridade para os n primeiros itens."""
        sub_ids = self.tracks["track_id"].values[:n]
        labels = [
            f"{r['title'][:20]}…" if len(r['title']) > 20 else r['title']
            for _, r in self.tracks.head(n).iterrows()
        ]
        indices = [self._tfidf_index[tid] for tid in sub_ids
                   if tid in self._tfidf_index]
        matrix = self._item_sim_matrix[np.ix_(indices, indices)]
        return pd.DataFrame(matrix, index=labels, columns=labels)