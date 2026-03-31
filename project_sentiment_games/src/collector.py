"""
src/collector.py
----------------
Coleta comentários reais de subreddits de games via Reddit API (PRAW).

API utilizada: Reddit API — https://www.reddit.com/dev/api
  - Gratuita com conta Reddit
  - Sem autenticação: usa user-agent público (read-only, limite menor)
  - Com credenciais: até 60 req/min

Variáveis de ambiente (opcionais):
  REDDIT_CLIENT_ID      → App ID do Reddit
  REDDIT_CLIENT_SECRET  → App Secret do Reddit
  REDDIT_USER_AGENT     → Nome do seu app (ex: "games-sentiment/1.0")

Como obter credenciais gratuitas:
  1. Acesse https://www.reddit.com/prefs/apps
  2. Clique "Create App" → tipo "script"
  3. Anote client_id e client_secret

Sem credenciais: usa dataset sintético realista de comentários de games.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "games-sentiment-analyzer/1.0")

# Subreddits por tema
SUBREDDITS = {
    "fps":     ["GlobalOffensive", "valorant", "apexlegends", "Overwatch"],
    "moba":    ["leagueoflegends", "DotA2", "heroesofthestorm"],
    "general": ["gaming", "Games", "pcgaming"],
    "rpg":     ["Eldenring", "Witcher", "assassinscreed"],
}

ALL_SUBREDDITS = [s for subs in SUBREDDITS.values() for s in subs]


@dataclass
class CollectionResult:
    """Dados brutos coletados."""
    comments: list[dict]
    source: str           # "reddit_api" | "synthetic"
    subreddits: list[str]
    collected_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.comments)


class RedditCollector:
    """
    Coleta comentários do Reddit via PRAW.

    Se não houver credenciais, gera dataset sintético com distribuições
    realistas de sentimentos para o contexto de games.

    Uso típico
    ----------
    >>> collector = RedditCollector(subreddits=["gaming", "GlobalOffensive"])
    >>> result = collector.run(limit=500)
    """

    # Dataset sintético — comentários reais de games categorizados
    SYNTHETIC_COMMENTS = {
        "positive": [
            "This game is absolutely incredible, best purchase of the year!",
            "Finally got my first win, so satisfying after all those hours!",
            "The new update is amazing, they really listened to the community.",
            "This map design is genius, every corner tells a story.",
            "Just hit Diamond rank, the grind was totally worth it!",
            "The graphics on this are stunning, runs perfectly on my rig.",
            "This patch fixed so many issues, the devs are doing great work.",
            "Best competitive game I've played in years, highly recommend.",
            "The storyline in this game is absolutely breathtaking.",
            "New season content is fire, can't stop playing.",
            "Finally a game that respects my time and doesn't feel grindy.",
            "The gunplay feels so satisfying, every shot matters.",
            "Community in this game is actually wholesome, rare these days.",
            "Just watched the esports finals, what a match! Insane plays.",
            "The devs dropped a free DLC, this studio deserves all support.",
            "My teammates carried me to victory, best squad ever!",
            "This game keeps getting better with every update.",
            "Spent 200 hours and still finding new things to explore.",
            "The anti-cheat update actually worked, matches feel fair now.",
            "New character kit is so well designed, love the creativity.",
        ],
        "negative": [
            "This game is completely broken after the latest patch.",
            "Getting matched against cheaters every single game, unplayable.",
            "The netcode is terrible, dying behind walls constantly.",
            "Devs only care about cosmetics and monetization, sad.",
            "Worst matchmaking I've ever seen, skill gap is ridiculous.",
            "Server issues again? This happens every weekend.",
            "The ranked system is rigged, been hardstuck for 2 months.",
            "New update broke more things than it fixed, classic.",
            "This battle pass is way overpriced for what you get.",
            "My team just trolls every game, ranked is a nightmare.",
            "Performance has gotten worse with every patch, uninstalling.",
            "The hit registration is completely inconsistent.",
            "Toxic players ruin every lobby, report system does nothing.",
            "Balance team has no idea what they're doing.",
            "Game crashes every 30 minutes since last update.",
            "The new monetization model is predatory and shameless.",
            "Ranked demotion system is so demoralizing, no fun.",
            "This map is so poorly designed, unbalanced for one side.",
            "Devs banned streamers for criticizing the game, shameful.",
            "Game feels abandoned, no communication from developers.",
        ],
        "neutral": [
            "Anyone know when the new season starts?",
            "What's the best loadout for ranked right now?",
            "How many hours do you guys have on this game?",
            "Is it worth buying on sale this weekend?",
            "Which character do you recommend for beginners?",
            "Does anyone use controller on PC for this?",
            "What rank are you currently in this season?",
            "Anyone else noticed the new sound effects?",
            "What server do you guys play on?",
            "Is cross-play enabled by default?",
            "Looking for teammates, anyone want to party up?",
            "What do you think about the new map rotation?",
            "Has anyone tried the new game mode yet?",
            "What's your opinion on the current meta?",
            "Do you think they'll add more content this month?",
            "How does this compare to last season?",
            "Anyone else grinding the battle pass?",
            "What settings do you use for competitive?",
            "Is the game free to play now or still paid?",
            "What GPU are you running this on?",
        ],
    }

    GAMES = [
        "CS2", "Valorant", "Apex Legends", "League of Legends",
        "Dota 2", "Elden Ring", "Fortnite", "Overwatch 2",
        "The Witcher 3", "Cyberpunk 2077",
    ]

    def __init__(
        self,
        subreddits: Optional[list[str]] = None,
        use_credentials: bool = True,
    ) -> None:
        self.subreddits = subreddits or ["gaming", "GlobalOffensive", "valorant"]
        self.use_credentials = use_credentials
        self._has_creds = bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)

    def run(self, limit: int = 300) -> CollectionResult:
        """Coleta comentários. Usa API real se tiver credenciais, senão sintético."""
        if self._has_creds and self.use_credentials:
            return self._collect_real(limit)
        return self._collect_synthetic(limit)

    def _collect_real(self, limit: int) -> CollectionResult:
        """Coleta via PRAW."""
        try:
            import praw
        except ImportError:
            logger.warning("praw não instalado. Usando dados sintéticos.")
            return self._collect_synthetic(limit)

        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            read_only=True,
        )

        comments = []
        per_sub = max(1, limit // len(self.subreddits))

        for sub_name in self.subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                for submission in subreddit.hot(limit=10):
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[:per_sub // 10]:
                        if len(comment.body) < 20 or comment.body == "[deleted]":
                            continue
                        comments.append({
                            "text":       comment.body[:500],
                            "subreddit":  sub_name,
                            "score":      comment.score,
                            "created_at": datetime.fromtimestamp(
                                comment.created_utc
                            ).isoformat(),
                            "game": self._guess_game(comment.body),
                        })
                logger.info("r/%s: %d comentários coletados.", sub_name, len(comments))
            except Exception as e:
                logger.error("Erro em r/%s: %s", sub_name, e)

        return CollectionResult(
            comments=comments, source="reddit_api",
            subreddits=self.subreddits,
        )

    def _collect_synthetic(self, limit: int) -> CollectionResult:
        """Gera dataset sintético realista."""
        rng = random.Random(42)
        comments = []

        # Distribuição realista: 40% positivo, 35% negativo, 25% neutro
        dist = (
            [("positive", t) for t in self.SYNTHETIC_COMMENTS["positive"]] * 8 +
            [("negative", t) for t in self.SYNTHETIC_COMMENTS["negative"]] * 7 +
            [("neutral",  t) for t in self.SYNTHETIC_COMMENTS["neutral"]]  * 5
        )
        rng.shuffle(dist)

        for i in range(min(limit, len(dist))):
            sentiment_label, base_text = dist[i]
            game = rng.choice(self.GAMES)
            sub = rng.choice(self.subreddits)

            # Varia ligeiramente o texto para parecer orgânico
            text = base_text
            if rng.random() > 0.6:
                text = f"{game}: {text}"

            created = datetime.now() - timedelta(
                hours=rng.randint(0, 72),
                minutes=rng.randint(0, 59),
            )

            comments.append({
                "text":             text,
                "subreddit":        sub,
                "score":            rng.randint(-5, 1500),
                "created_at":       created.isoformat(timespec="seconds"),
                "game":             game,
                "_true_sentiment":  sentiment_label,  # usado para avaliação
            })

        logger.info("Dataset sintético: %d comentários gerados.", len(comments))
        return CollectionResult(
            comments=comments, source="synthetic",
            subreddits=self.subreddits,
        )

    @staticmethod
    def _guess_game(text: str) -> str:
        """Tenta identificar o jogo mencionado no texto."""
        keywords = {
            "CS2": ["cs2", "csgo", "counter-strike", "ct side", "t side"],
            "Valorant": ["valorant", "valo", "radiant", "jett", "sage"],
            "Apex Legends": ["apex", "legend", "wraith", "bloodhound"],
            "League of Legends": ["league", "lol", "rift", "champion", "jungle"],
            "Dota 2": ["dota", "dota2", "ancient", "roshan"],
            "Fortnite": ["fortnite", "battle royale", "building"],
            "Overwatch": ["overwatch", "ow2", "pharah", "tracer"],
        }
        text_lower = text.lower()
        for game, kws in keywords.items():
            if any(kw in text_lower for kw in kws):
                return game
        return "Other"