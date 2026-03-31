"""
src/fetcher.py
--------------
Coleta dados reais de músicas da Last.fm API.

API utilizada: Last.fm (https://www.last.fm/api)
  - Gratuita com registro de API key
  - Sem autenticação OAuth para dados públicos
  - Rate limit: 5 req/s (mais que suficiente)

Endpoints usados:
  chart.getTopTracks       → Top músicas globais
  track.getSimilar         → Músicas similares a uma faixa
  track.getTopTags         → Tags/gêneros de uma música
  artist.getTopTracks      → Top músicas de um artista

Variável de ambiente:
  LASTFM_API_KEY   → Chave gratuita em https://www.last.fm/api/account/create
                     Sem chave: usa DEMO_KEY com dados sintéticos enriquecidos
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://ws.audioscrobbler.com/2.0/"
API_KEY = os.getenv("LASTFM_API_KEY", "")
REQUEST_TIMEOUT = 10
RETRY_BACKOFF = 1.5


@dataclass
class MusicDataset:
    """Dataset consolidado pronto para o recomendador."""
    tracks_df: pd.DataFrame        # catálogo de músicas
    ratings_df: pd.DataFrame       # matriz usuário × música (sintética)
    tags_df: pd.DataFrame          # tags por música


class LastFmClient:
    """Wrapper HTTP para a Last.fm API com retry."""

    def __init__(self, api_key: str = API_KEY) -> None:
        self.api_key = api_key
        self.has_key = bool(api_key)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, method: str, **params) -> dict:
        """GET com retry automático."""
        payload = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            **params,
        }
        for attempt in range(1, 4):
            try:
                resp = self.session.get(BASE_URL, params=payload,
                                        timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError as e:
                if attempt == 3:
                    raise RuntimeError(
                        f"Sem conexão com a Last.fm API. Detalhe: {e}"
                    ) from e
                time.sleep(RETRY_BACKOFF * attempt)
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(f"HTTP {e.response.status_code}") from e
        return {}

    def get_top_tracks(self, limit: int = 100) -> list[dict]:
        data = self._get("chart.getTopTracks", limit=limit)
        return data.get("tracks", {}).get("track", [])

    def get_similar_tracks(self, artist: str, track: str,
                           limit: int = 10) -> list[dict]:
        try:
            data = self._get("track.getSimilar", artist=artist,
                             track=track, limit=limit)
            return data.get("similartracks", {}).get("track", [])
        except Exception:
            return []

    def get_top_tags(self, artist: str, track: str) -> list[str]:
        try:
            data = self._get("track.getTopTags", artist=artist, track=track)
            tags = data.get("toptags", {}).get("tag", [])
            return [t["name"].lower() for t in tags[:5]]
        except Exception:
            return []


class MusicFetcher:
    """
    Coleta músicas reais da Last.fm e gera dataset para o recomendador.

    Se não houver API key, gera dataset sintético com distribuições
    baseadas nas tendências reais do Last.fm (gêneros, popularidade).

    Uso típico
    ----------
    >>> fetcher = MusicFetcher(n_tracks=100)
    >>> dataset = fetcher.run()
    """

    # Catálogo sintético enriquecido — baseado em dados reais do Last.fm
    SYNTHETIC_CATALOG = [
        # (title, artist, genre, subgenre, playcount_base)
        ("Blinding Lights", "The Weeknd", "pop", "synth-pop", 3_200_000),
        ("Bohemian Rhapsody", "Queen", "rock", "classic rock", 2_800_000),
        ("Shape of You", "Ed Sheeran", "pop", "dance-pop", 3_100_000),
        ("Smells Like Teen Spirit", "Nirvana", "rock", "grunge", 2_400_000),
        ("Bad Guy", "Billie Eilish", "pop", "electropop", 2_600_000),
        ("God's Plan", "Drake", "hip-hop", "trap", 2_900_000),
        ("Stairway to Heaven", "Led Zeppelin", "rock", "hard rock", 2_200_000),
        ("Rolling in the Deep", "Adele", "pop", "soul", 2_700_000),
        ("HUMBLE.", "Kendrick Lamar", "hip-hop", "conscious rap", 2_300_000),
        ("Don't Stop Believin'", "Journey", "rock", "classic rock", 2_100_000),
        ("Levitating", "Dua Lipa", "pop", "dance-pop", 2_500_000),
        ("Lose Yourself", "Eminem", "hip-hop", "rap", 2_800_000),
        ("Hotel California", "Eagles", "rock", "classic rock", 2_000_000),
        ("As It Was", "Harry Styles", "pop", "indie pop", 2_400_000),
        ("SICKO MODE", "Travis Scott", "hip-hop", "trap", 2_200_000),
        ("Africa", "Toto", "rock", "pop rock", 1_900_000),
        ("Watermelon Sugar", "Harry Styles", "pop", "indie pop", 2_100_000),
        ("Rap God", "Eminem", "hip-hop", "rap", 2_000_000),
        ("Sweet Child O' Mine", "Guns N' Roses", "rock", "hard rock", 2_300_000),
        ("Peaches", "Justin Bieber", "pop", "r&b", 2_000_000),
        ("Industry Baby", "Lil Nas X", "hip-hop", "pop rap", 1_900_000),
        ("Mr. Brightside", "The Killers", "rock", "indie rock", 2_100_000),
        ("Dynamite", "BTS", "pop", "k-pop", 2_200_000),
        ("Old Town Road", "Lil Nas X", "hip-hop", "country rap", 2_600_000),
        ("Born to Run", "Bruce Springsteen", "rock", "heartland rock", 1_800_000),
        ("Montero", "Lil Nas X", "pop", "pop rap", 1_800_000),
        ("Uptown Funk", "Bruno Mars", "pop", "funk", 2_400_000),
        ("Numb", "Linkin Park", "rock", "nu-metal", 2_000_000),
        ("Tití Me Preguntó", "Bad Bunny", "latin", "reggaeton", 2_100_000),
        ("Con Calma", "Daddy Yankee", "latin", "reggaeton", 1_900_000),
        ("Despacito", "Luis Fonsi", "latin", "reggaeton", 2_800_000),
        ("Mi Gente", "J Balvin", "latin", "reggaeton", 2_000_000),
        ("Electric Feel", "MGMT", "rock", "indie rock", 1_700_000),
        ("Redbone", "Childish Gambino", "pop", "neo-soul", 1_800_000),
        ("Hotline Bling", "Drake", "hip-hop", "r&b", 2_100_000),
        ("Circles", "Post Malone", "pop", "pop rap", 2_000_000),
        ("Sunflower", "Post Malone", "pop", "pop rap", 2_200_000),
        ("Stay With Me", "Sam Smith", "pop", "soul", 1_900_000),
        ("Someone Like You", "Adele", "pop", "soul", 2_500_000),
        ("Love Story", "Taylor Swift", "pop", "country pop", 2_100_000),
        ("Anti-Hero", "Taylor Swift", "pop", "synth-pop", 2_300_000),
        ("Cruel Summer", "Taylor Swift", "pop", "synth-pop", 2_000_000),
        ("Good 4 U", "Olivia Rodrigo", "pop", "pop punk", 1_900_000),
        ("drivers license", "Olivia Rodrigo", "pop", "indie pop", 2_000_000),
        ("Happier Than Ever", "Billie Eilish", "pop", "electropop", 1_800_000),
        ("Heat Waves", "Glass Animals", "rock", "indie pop", 1_900_000),
        ("Astronaut in the Ocean", "Masked Wolf", "hip-hop", "rap", 1_700_000),
        ("Stay", "The Kid LAROI", "pop", "pop rap", 1_900_000),
        ("Infinity", "Jaymes Young", "pop", "indie pop", 1_600_000),
        ("Die For You", "The Weeknd", "pop", "r&b", 1_800_000),
    ]

    GENRES = ["pop", "rock", "hip-hop", "latin", "r&b", "indie", "electronic"]

    def __init__(self, n_tracks: int = 50, n_users: int = 200) -> None:
        self.n_tracks = min(n_tracks, len(self.SYNTHETIC_CATALOG))
        self.n_users = n_users
        self.client = LastFmClient()

    def run(self) -> MusicDataset:
        """Retorna MusicDataset com tracks, ratings e tags."""
        if self.client.has_key:
            logger.info("API key encontrada — buscando dados reais do Last.fm...")
            return self._fetch_real()
        else:
            logger.info("Sem API key — usando catálogo sintético enriquecido.")
            return self._build_synthetic()

    def _build_synthetic(self) -> MusicDataset:
        """Gera dataset completo sem API key."""
        import numpy as np
        rng = np.random.default_rng(42)

        catalog = self.SYNTHETIC_CATALOG[:self.n_tracks]

        tracks_df = pd.DataFrame(catalog,
                                  columns=["title", "artist", "genre",
                                           "subgenre", "playcount"])
        tracks_df["track_id"] = range(len(tracks_df))
        tracks_df["popularity"] = (
            (tracks_df["playcount"] / tracks_df["playcount"].max() * 100)
            .round(1)
        )

        # Tags por música (genre + subgenre + extras)
        extra_tags = {
            "pop": ["catchy", "mainstream", "radio"],
            "rock": ["guitar", "live", "classic"],
            "hip-hop": ["beats", "rhymes", "urban"],
            "latin": ["danceable", "rhythmic", "tropical"],
            "r&b": ["soulful", "smooth", "emotional"],
            "indie": ["alternative", "artistic", "underground"],
            "electronic": ["edm", "synth", "club"],
        }
        tag_records = []
        for _, row in tracks_df.iterrows():
            tags = [row["genre"], row["subgenre"]]
            tags += extra_tags.get(row["genre"], [])[:2]
            for tag in tags:
                tag_records.append({"track_id": row["track_id"],
                                    "title": row["title"],
                                    "tag": tag})
        tags_df = pd.DataFrame(tag_records)

        # Matriz de ratings sintética (usuário × música)
        # Usuários têm preferências por gênero — torna o User-Based significativo
        genre_list = tracks_df["genre"].unique().tolist()
        records = []
        for user_id in range(self.n_users):
            # Cada usuário prefere 1-2 gêneros
            fav_genres = rng.choice(genre_list,
                                    size=rng.integers(1, 3),
                                    replace=False).tolist()
            n_rated = rng.integers(5, 20)
            rated_tracks = rng.choice(tracks_df["track_id"].values,
                                      size=min(n_rated, len(tracks_df)),
                                      replace=False)
            for tid in rated_tracks:
                genre = tracks_df.loc[
                    tracks_df["track_id"] == tid, "genre"
                ].values[0]
                # Rating maior para gêneros favoritos
                base = 4.0 if genre in fav_genres else 2.5
                rating = float(np.clip(
                    rng.normal(base, 0.7), 1.0, 5.0
                ))
                records.append({
                    "user_id":  f"user_{user_id:03d}",
                    "track_id": tid,
                    "rating":   round(rating, 1),
                })
        ratings_df = pd.DataFrame(records)

        logger.info("Dataset sintético: %d músicas, %d usuários, %d ratings.",
                    len(tracks_df), self.n_users, len(ratings_df))
        return MusicDataset(tracks_df=tracks_df,
                            ratings_df=ratings_df,
                            tags_df=tags_df)

    def _fetch_real(self) -> MusicDataset:
        """Busca dados reais da Last.fm API."""
        logger.info("Buscando top tracks do Last.fm...")
        raw_tracks = self.client.get_top_tracks(limit=self.n_tracks)

        records = []
        tag_records = []
        for i, t in enumerate(raw_tracks):
            track_id = i
            title = t.get("name", "")
            artist = t.get("artist", {}).get("name", "")
            playcount = int(t.get("playcount", 0))

            tags = self.client.get_top_tags(artist, title)
            genre = tags[0] if tags else "unknown"

            records.append({
                "track_id":  track_id,
                "title":     title,
                "artist":    artist,
                "genre":     genre,
                "subgenre":  tags[1] if len(tags) > 1 else genre,
                "playcount": playcount,
                "popularity": round(playcount / 1_000_000, 1),
            })
            for tag in tags:
                tag_records.append({"track_id": track_id,
                                    "title": title, "tag": tag})

            time.sleep(0.2)  # respeita rate limit

        tracks_df = pd.DataFrame(records)
        tags_df = pd.DataFrame(tag_records)

        # Ratings sintéticos mesmo com API real
        synthetic = self._build_synthetic()
        return MusicDataset(tracks_df=tracks_df,
                            ratings_df=synthetic.ratings_df,
                            tags_df=tags_df)