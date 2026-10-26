"""
Foundations of AI — Knowledge Graphs & LLM Fact-Checking Lab
=============================================================

This lab builds a knowledge graph from IMDb data, then compares a graph-based
fact-checker against a local LLM (via Ollama) on the same set of claims.

Setup
-----
1. Download the IMDb datasets (free, non-commercial use):
       https://datasets.imdbws.com/title.basics.tsv.gz
       https://datasets.imdbws.com/title.principals.tsv.gz
       https://datasets.imdbws.com/name.basics.tsv.gz
       https://datasets.imdbws.com/title.ratings.tsv.gz
   Place the .gz files (or unzipped .tsv files) in the same directory as this script.

2. Install dependencies:
       pip install networkx requests tqdm

3. Install Ollama from https://ollama.ai, then:
       ollama pull llama3          # ~4 GB; swap any model you prefer

Usage
-----
   python kg_factcheck_lab.py --build          # build and cache the graph
   python kg_factcheck_lab.py --benchmark      # run fixed benchmark (KG only)
   python kg_factcheck_lab.py --benchmark --llm   # benchmark + LLM comparison
   python kg_factcheck_lab.py --repl           # interactive REPL
   python kg_factcheck_lab.py --repl --llm     # REPL with live LLM comparison
"""

import argparse
import gzip
import json
from pathlib import Path

import networkx as nx
import requests


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these to match your environment)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR      = Path(".")          # where the IMDb .tsv/.gz files live
CACHE_FILE    = Path("imdb_kg.json")   # cached graph (built once, reused after)
MIN_VOTES     = 0                # popularity filter
YEAR_MIN, YEAR_MAX = 2000, 2025        # release-year filter
MAX_FILMS     = 1000000                  # cap for quick iteration; remove to use all
OLLAMA_URL    = "http://localhost:11434"
OLLAMA_MODEL  = "llama3"              # swap to any model you have pulled


# ─────────────────────────────────────────────────────────────────────────────
# PART 1  ─  BUILD THE KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def open_tsv(filename):
    """Opens a .tsv or .tsv.gz file and returns a line iterator."""
    p = DATA_DIR / filename
    if p.exists():
        return open(p, encoding="utf-8")
    gz = DATA_DIR / (filename + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", encoding="utf-8")
    raise FileNotFoundError(
        f"Neither {p} nor {gz} found.\n"
        "Download the IMDb datasets from https://datasets.imdbws.com/ and place "
        "them in the same directory as this script."
    )


def load_ratings():
    """Returns {tconst: numVotes} from title.ratings.tsv."""
    votes = {}
    try:
        fh = open_tsv("title.ratings.tsv")
    except FileNotFoundError:
        return votes          # ratings are optional; we'll skip the vote filter
    with fh:
        next(fh)              # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                votes[parts[0]] = int(parts[2]) if parts[2] != "\\N" else 0
    return votes


def load_films(votes):
    films = {}
    count = 0

    with open_tsv("title.basics.tsv") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            tconst, ttype, primary_title, _, _, start_year, end_year, _, genres = parts[:9]

            if ttype not in {"movie", "tvMovie"}:
                continue
            if start_year == "\\N" and end_year == "\\N":
                continue
            try:
                year = int(start_year)
            except ValueError:
                try :
                    year = int(end_year)
                except ValueError:
                    continue

            if not (YEAR_MIN <= year <= YEAR_MAX):
                continue
            n_votes = votes.get(tconst, 0)
            if votes and n_votes < MIN_VOTES:
                continue
            genre_list = [] if genres == "\\N" else genres.split(",")
            films[tconst] = {
                "title":  primary_title,
                "year":   year,
                "genres": genre_list,
                "votes":  n_votes,
            }
            count += 1
            if MAX_FILMS and count >= MAX_FILMS:
                break
    return films




def load_people(film_ids):
    """
    Returns:
        people  – {nconst: {"name": str, "roles": {tconst: [category]}}}
        cast    – {tconst: {nconst: [category]}}   (director, actor, actress, …)

    Only loads people linked to films in `film_ids`.
    """
    people = {}
    cast   = {}

    with open_tsv("title.principals.tsv") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            tconst, _, nconst, category = parts[0], parts[1], parts[2], parts[3]

            if tconst not in film_ids:
                continue
            if category not in {"actor", "actress", "director", "writer", "producer"}:
                continue

            cast.setdefault(tconst, {}).setdefault(nconst, []).append(category)
            people.setdefault(nconst, {"name": "", "roles": {}})
            people[nconst]["roles"].setdefault(tconst, []).append(category)

    # fill in names
    nconst_needed = set(people.keys())
    with open_tsv("name.basics.tsv") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in nconst_needed:
                people[parts[0]]["name"] = parts[1]

    return people, cast


def build_graph(films, people, cast):
    """
    Constructs a directed NetworkX graph.

    Node types
    ----------
    film   – tconst,  attrs: title, year, genres, votes
    person – nconst,  attrs: name
    genre  – genre string

    Edge types (all directed: source → target)
    ----------
    DIRECTED_BY   film  → person
    ACTED_IN_BY   film  → person
    WRITTEN_BY    film  → person
    HAS_GENRE     film  → genre
    CO_STAR       person → person  (both appeared in the same film)

    Reverses (added automatically for easy traversal)
    ----------
    DIRECTED      person → film
    ACTED_IN      person → film
    WROTE         person → film
    """
    G = nx.MultiDiGraph()

    # ── Film nodes ────────────────────────────────────────────────────────────
    for tconst, info in films.items():
        G.add_node(tconst,
                   ntype="film",
                   title=info["title"],
                   year=info["year"],
                   genres=info["genres"],
                   votes=info["votes"])

    # ── Person nodes ──────────────────────────────────────────────────────────
    for nconst, info in people.items():
        G.add_node(nconst, ntype="person", name=info["name"])

    # ── Edges ─────────────────────────────────────────────────────────────────
    for tconst, crew in cast.items():
        for nconst, categories in crew.items():
            if tconst not in G or nconst not in G:
                continue
            for cat in categories:
                if cat == "director":
                    G.add_edge(tconst, nconst, rel="DIRECTED_BY")
                    G.add_edge(nconst, tconst, rel="DIRECTED")
                elif cat in {"actor", "actress"}:
                    G.add_edge(tconst, nconst, rel="ACTED_IN_BY")
                    G.add_edge(nconst, tconst, rel="ACTED_IN")
                elif cat == "writer":
                    G.add_edge(tconst, nconst, rel="WRITTEN_BY")
                    G.add_edge(nconst, tconst, rel="WROTE")

    # ── Genre nodes ───────────────────────────────────────────────────────────
    for tconst, info in films.items():
        for genre in info["genres"]:
            if genre:
                if not G.has_node(genre):
                    G.add_node(genre, ntype="genre")
                G.add_edge(tconst, genre, rel="HAS_GENRE")
                G.add_edge(genre, tconst, rel="GENRE_INCLUDES")

    # ── Co-star edges (2-hop compression) ────────────────────────────────────
    #   For each film, add a CO_STAR edge between every pair of actors/actresses.
    #   This lets us answer "did X and Y appear together?" in a single hop.
    for tconst, crew in cast.items():
        actors = [nc for nc, cats in crew.items()
                  if any(c in cats for c in ("actor", "actress"))]
        for i, a1 in enumerate(actors):
            for a2 in actors[i+1:]:
                if G.has_node(a1) and G.has_node(a2):
                    G.add_edge(a1, a2, rel="CO_STAR", film=tconst)
                    G.add_edge(a2, a1, rel="CO_STAR", film=tconst)

    return G


# ── Serialisation helpers ─────────────────────────────────────────────────────

def save_graph(G, path):
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Graph saved to {path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    G = nx.node_link_graph(data, directed=True)
    print(f"Graph loaded from {path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G


def build_and_save():
    print("Step 1/4  Loading ratings …")
    votes  = load_ratings()
    print(f"          {len(votes):,} titles with vote counts")

    print("Step 2/4  Loading films …")
    films  = load_films(votes)
    print(f"          {len(films):,} films pass the filter")

    print("Step 3/4  Loading cast & crew …")
    people, cast = load_people(set(films.keys()))
    print(f"          {len(people):,} people linked to selected films")

    print("Step 4/4  Building graph …")
    G = build_graph(films, people, cast)
    save_graph(G, CACHE_FILE)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# PART 2  ─  GRAPH TRAVERSAL AND VERDICT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
#
# The fact-checker operates under the CLOSED-WORLD ASSUMPTION:
#   - If the graph contains the path that would verify a claim → TRUE
#   - If the graph contains a contradicting path            → FALSE
#   - If the relevant entities aren't in the graph at all    → UNKNOWN
#
# Verdicts are always accompanied by the evidence path used.

UNKNOWN = "UNKNOWN"
TRUE    = "TRUE"
FALSE   = "FALSE"


class KnowledgeGraphChecker:
    """
    Provides a .check(claim_type, **kwargs) method that traverses the graph
    to verify structured claims.

    Claim types and required kwargs
    --------------------------------
    directed_film    – person_name, film_title
    acted_in_film    – person_name, film_title
    wrote_film       – person_name, film_title
    film_year        – film_title, year (int or str)
    film_genre       – film_title, genre
    co_starred       – person1_name, person2_name
    same_director    – film1_title, film2_title
    person_directed  – person_name  (how many films?)   [count query]
    films_in_genre   – genre, year_min, year_max        [count query]
    director_of_genre – person_name, genre              [bool query]
    """

    def __init__(self, G):
        self.G = G
        # build look-up indexes (name → id)
        self._film_idx   = {}   # lower-case title → [tconst, …]
        self._person_idx = {}   # lower-case name  → [nconst, …]
        self._genre_idx  = set()

        for node, attrs in G.nodes(data=True):
            ntype = attrs.get("ntype")
            if ntype == "film":
                key = attrs.get("title", "").lower()
                self._film_idx.setdefault(key, []).append(node)
            elif ntype == "person":
                key = attrs.get("name", "").lower()
                self._person_idx.setdefault(key, []).append(node)
            elif ntype == "genre":
                self._genre_idx.add(node)

    # ── Entity look-up helpers ────────────────────────────────────────────────

    def _find_film(self, title):
        """Returns (tconst, attrs) or None.  Fuzzy-ish: strips punctuation."""
        key = title.lower().strip()
        if key in self._film_idx:
            tc = self._film_idx[key][0]
            return tc, self.G.nodes[tc]
        # try stripping "the", "a", etc.
        for stored_key, ids in self._film_idx.items():
            if stored_key.startswith(key) or key.startswith(stored_key):
                tc = ids[0]
                return tc, self.G.nodes[tc]
        return None

    def _find_person(self, name):
        """Returns (nconst, attrs) or None."""
        key = name.lower().strip()
        if key in self._person_idx:
            nc = self._person_idx[key][0]
            return nc, self.G.nodes[nc]
        return None

    def _edge_rel(self, u, v, rel):
        """Returns True if edge u→v with rel=rel exists."""
        return self.G.has_edge(u, v) and self.G[u][v].get("rel") == rel

    # ── Core claim checkers ───────────────────────────────────────────────────

    def _check_person_film(self, person_name, film_title, edge_rel, rev_rel):
        """Generic handler for directed/acted/wrote claims."""
        person = self._find_person(person_name)
        film   = self._find_film(film_title)

        if person is None:
            return UNKNOWN, f"Person '{person_name}' not found in graph."
        if film is None:
            return UNKNOWN, f"Film '{film_title}' not found in graph."

        nc, p_attrs = person
        tc, f_attrs = film

        # direct 1-hop check
        if self._edge_rel(nc, tc, rev_rel):
            return TRUE, (f"{p_attrs['name']} —[{rev_rel}]→ "
                          f"{f_attrs['title']} ({f_attrs['year']})")

        # was the film found but the person not linked at all?
        return FALSE, (f"No {rev_rel} edge from {p_attrs['name']} to "
                       f"{f_attrs['title']}. "
                       f"Actual {edge_rel} of that film: "
                       + self._who_did(tc, edge_rel))

    def _who_did(self, tconst, rel, max_names=3):
        """Returns a comma-separated list of people with rel from tconst."""
        names = []
        for _, v, data in self.G.out_edges(tconst, data=True):
            if data.get("rel") == rel:
                attrs = self.G.nodes[v]
                names.append(attrs.get("name", v))
                if len(names) >= max_names:
                    break
        return ", ".join(names) if names else "(none in graph)"

    def check(self, claim_type, **kwargs):
        """
        Returns (verdict, evidence_string) where verdict ∈ {TRUE, FALSE, UNKNOWN}.

        ── 1-hop claims ──────────────────────────────────────────────────────
        directed_film  : Did <person_name> direct <film_title>?
        acted_in_film  : Did <person_name> appear in <film_title>?
        wrote_film     : Did <person_name> write <film_title>?
        film_year      : Was <film_title> released in <year>?
        film_genre     : Is <film_title> a <genre> film?

        ── 2-hop claims ──────────────────────────────────────────────────────
        co_starred     : Did <person1_name> and <person2_name> appear together?
        same_director  : Do <film1_title> and <film2_title> share a director?
        director_of_genre : Did <person_name> ever direct a <genre> film?

        ── 3-hop claims ──────────────────────────────────────────────────────
        shared_cast_genre : Did any actor from <film1_title> also appear in
                            a <genre> film directed by <person_name>?
        """
        ct = claim_type

        # ── 1-hop ──────────────────────────────────────────────────────────
        if ct == "directed_film":
            return self._check_person_film(
                kwargs["person_name"], kwargs["film_title"],
                "DIRECTED_BY", "DIRECTED")

        if ct == "acted_in_film":
            return self._check_person_film(
                kwargs["person_name"], kwargs["film_title"],
                "ACTED_IN_BY", "ACTED_IN")

        if ct == "wrote_film":
            return self._check_person_film(
                kwargs["person_name"], kwargs["film_title"],
                "WRITTEN_BY", "WROTE")

        if ct == "film_year":
            film = self._find_film(kwargs["film_title"])
            if film is None:
                return UNKNOWN, f"Film '{kwargs['film_title']}' not found."
            tc, attrs = film
            claimed = int(kwargs["year"])
            actual  = attrs["year"]
            if actual == claimed:
                return TRUE, f"{attrs['title']} was released in {actual}."
            return FALSE, f"{attrs['title']} was released in {actual}, not {claimed}."

        if ct == "film_genre":
            film  = self._find_film(kwargs["film_title"])
            genre = kwargs["genre"].strip().title()
            if film is None:
                return UNKNOWN, f"Film '{kwargs['film_title']}' not found."
            tc, f_attrs = film
            if self.G.has_edge(tc, genre):
                return TRUE, f"{f_attrs['title']} genres: {f_attrs['genres']}"
            actual_genres = f_attrs.get("genres", [])
            return FALSE, (f"{f_attrs['title']} genres are {actual_genres}, "
                           f"not '{genre}'.")

        # ── 2-hop ──────────────────────────────────────────────────────────
        if ct == "co_starred":
            p1 = self._find_person(kwargs["person1_name"])
            p2 = self._find_person(kwargs["person2_name"])
            if p1 is None:
                return UNKNOWN, f"Person '{kwargs['person1_name']}' not found."
            if p2 is None:
                return UNKNOWN, f"Person '{kwargs['person2_name']}' not found."
            nc1, a1 = p1; nc2, a2 = p2

            # Check direct CO_STAR edge (pre-computed)
            if self.G.has_edge(nc1, nc2) and self.G[nc1][nc2].get("rel") == "CO_STAR":
                film_tc = self.G[nc1][nc2].get("film")
                film_title = (self.G.nodes[film_tc].get("title", film_tc)
                              if film_tc and film_tc in self.G else "a shared film")
                return TRUE, (f"{a1['name']} and {a2['name']} both appeared in "
                              f"'{film_title}'.")

            # Fallback: explicit 2-hop traversal (handles multi-film)
            films_1 = {v for _, v, d in self.G.out_edges(nc1, data=True)
                       if d.get("rel") == "ACTED_IN"}
            films_2 = {v for _, v, d in self.G.out_edges(nc2, data=True)
                       if d.get("rel") == "ACTED_IN"}
            shared = films_1 & films_2
            if shared:
                tc = next(iter(shared))
                t  = self.G.nodes[tc].get("title", tc)
                return TRUE, f"{a1['name']} and {a2['name']} both appeared in '{t}'."
            return FALSE, (f"No shared film found between {a1['name']} "
                           f"and {a2['name']}.")

        if ct == "same_director":
            f1 = self._find_film(kwargs["film1_title"])
            f2 = self._find_film(kwargs["film2_title"])
            if f1 is None: return UNKNOWN, f"Film '{kwargs['film1_title']}' not found."
            if f2 is None: return UNKNOWN, f"Film '{kwargs['film2_title']}' not found."
            tc1, a1 = f1; tc2, a2 = f2

            dirs1 = {v for _, v, d in self.G.out_edges(tc1, data=True)
                     if d.get("rel") == "DIRECTED_BY"}
            dirs2 = {v for _, v, d in self.G.out_edges(tc2, data=True)
                     if d.get("rel") == "DIRECTED_BY"}
            shared = dirs1 & dirs2
            if shared:
                nc = next(iter(shared))
                name = self.G.nodes[nc].get("name", nc)
                return TRUE, (f"{name} directed both '{a1['title']}' ({a1['year']}) "
                              f"and '{a2['title']}' ({a2['year']}).")
            d1 = self._who_did(tc1, "DIRECTED_BY")
            d2 = self._who_did(tc2, "DIRECTED_BY")
            return FALSE, (f"'{a1['title']}' was directed by {d1}; "
                           f"'{a2['title']}' was directed by {d2}. No overlap.")

        if ct == "director_of_genre":
            person = self._find_person(kwargs["person_name"])
            genre  = kwargs["genre"].strip().title()
            if person is None:
                return UNKNOWN, f"Person '{kwargs['person_name']}' not found."
            nc, p_attrs = person

            # films they directed
            directed = [v for _, v, d in self.G.out_edges(nc, data=True)
                        if d.get("rel") == "DIRECTED"]
            genre_hits = [tc for tc in directed
                          if self.G.has_edge(tc, genre)]
            if genre_hits:
                titles = [self.G.nodes[tc].get("title", tc) for tc in genre_hits[:3]]
                return TRUE, (f"{p_attrs['name']} directed {genre} films: "
                              + ", ".join(f"'{t}'" for t in titles))
            if directed:
                return FALSE, (f"{p_attrs['name']} directed {len(directed)} films "
                               f"in this graph, but none are {genre}.")
            return UNKNOWN, f"{p_attrs['name']} has no directed films in this graph."

        # ── 3-hop ──────────────────────────────────────────────────────────
        if ct == "shared_cast_genre":
            """
            3-hop: film1 → actors → films_in_genre directed by person_name.
            Answers: "Did any actor from <film1> also appear in a <genre>
            film directed by <person_name>?"
            """
            f1     = self._find_film(kwargs["film1_title"])
            person = self._find_person(kwargs["person_name"])
            genre  = kwargs["genre"].strip().title()

            if f1 is None:     return UNKNOWN, f"Film '{kwargs['film1_title']}' not found."
            if person is None: return UNKNOWN, f"Person '{kwargs['person_name']}' not found."
            tc1, f1_attrs = f1
            nc, p_attrs   = person

            # hop 1: actors in film1
            actors_in_f1 = {v for _, v, d in self.G.out_edges(tc1, data=True)
                            if d.get("rel") == "ACTED_IN_BY"}
            # hop 2: films directed by person in genre
            directed_genre_films = {v for _, v, d in self.G.out_edges(nc, data=True)
                                    if d.get("rel") == "DIRECTED"
                                    and self.G.has_edge(v, genre)}
            # hop 3: intersection via ACTED_IN
            for nc_actor in actors_in_f1:
                actor_films = {v for _, v, d in self.G.out_edges(nc_actor, data=True)
                               if d.get("rel") == "ACTED_IN"}
                hit = actor_films & directed_genre_films
                if hit:
                    a_name  = self.G.nodes[nc_actor].get("name", nc_actor)
                    hit_tc  = next(iter(hit))
                    hit_t   = self.G.nodes[hit_tc].get("title", hit_tc)
                    return TRUE, (
                        f"{a_name} appeared in '{f1_attrs['title']}' (hop 1) "
                        f"AND in '{hit_t}' (hop 2), which is a {genre} film "
                        f"directed by {p_attrs['name']} (hop 3).")
            return FALSE, (
                f"No actor from '{f1_attrs['title']}' appeared in a {genre} film "
                f"directed by {p_attrs['name']}.")

        return UNKNOWN, f"Unknown claim type: '{claim_type}'"


# ─────────────────────────────────────────────────────────────────────────────
# PART 3  ─  OLLAMA LLM ADAPTER
# ─────────────────────────────────────────────────────────────────────────────
#
# The adapter wraps any Ollama-compatible endpoint.  Swap OLLAMA_MODEL or
# point OLLAMA_URL at a different host to use any model Ollama supports.
# The interface is intentionally thin: one method, ask(prompt) → str.
#
# HOW TO ADD A DIFFERENT BACKEND
# ───────────────────────────────
# Subclass LLMAdapter and override .ask():
#
#   class OpenAIAdapter(LLMAdapter):
#       def ask(self, prompt):
#           import openai
#           resp = openai.ChatCompletion.create(
#               model="gpt-4o",
#               messages=[{"role": "user", "content": prompt}])
#           return resp.choices[0].message.content
#
# Then pass your adapter to compare_claim() or the REPL.

class LLMAdapter:
    """Base class — override ask() to swap models."""
    def ask(self, prompt: str) -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        return False


class OllamaAdapter(LLMAdapter):
    """
    Talks to a locally running Ollama instance.

    Install Ollama: https://ollama.ai
    Pull a model:  ollama pull llama3

    The adapter uses the /api/generate endpoint with stream=False so the
    full response is returned in one JSON blob.  Any model you have pulled
    works; set OLLAMA_MODEL at the top of this file.
    """

    def __init__(self, base_url=OLLAMA_URL, model=OLLAMA_MODEL, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout
        self._system  = (
            "You are a precise fact-checking assistant. "
            "Answer ONLY with TRUE, FALSE, or UNKNOWN on the first line, "
            "then a short explanation (≤ 2 sentences). "
            "Do not hedge with phrases like 'I believe' — commit to one verdict."
        )

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def ask(self, prompt: str) -> str:
        payload = {
            "model":  self.model,
            "prompt": f"{self._system}\n\nClaim: {prompt}",
            "stream": False,
        }
        try:
            r = requests.post(f"{self.base_url}/api/generate",
                              json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return "[Ollama not running — start with: ollama serve]"
        except Exception as e:
            return f"[LLM error: {e}]"

    def parse_verdict(self, response: str):
        """
        Extracts (verdict, explanation) from a raw LLM response.
        verdict ∈ {TRUE, FALSE, UNKNOWN, ERROR}
        """
        first_line = response.strip().split("\n")[0].upper()
        for v in (TRUE, FALSE, UNKNOWN):
            if v in first_line:
                explanation = response.strip()
                return v, explanation
        return "ERROR", response.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PART 4  ─  COMPARISON ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def natural_language_claim(claim_type, **kwargs):
    """Converts a structured claim dict into a natural-language question for the LLM."""
    ct = claim_type
    if ct == "directed_film":
        return f"Did {kwargs['person_name']} direct the film '{kwargs['film_title']}'?"
    if ct == "acted_in_film":
        return f"Did {kwargs['person_name']} appear in the film '{kwargs['film_title']}'?"
    if ct == "wrote_film":
        return f"Did {kwargs['person_name']} write the screenplay for '{kwargs['film_title']}'?"
    if ct == "film_year":
        return f"Was the film '{kwargs['film_title']}' released in {kwargs['year']}?"
    if ct == "film_genre":
        return f"Is '{kwargs['film_title']}' a {kwargs['genre']} film?"
    if ct == "co_starred":
        return (f"Did {kwargs['person1_name']} and {kwargs['person2_name']} "
                f"appear together in the same film?")
    if ct == "same_director":
        return (f"Were '{kwargs['film1_title']}' and '{kwargs['film2_title']}' "
                f"directed by the same person?")
    if ct == "director_of_genre":
        return f"Has {kwargs['person_name']} ever directed a {kwargs['genre']} film?"
    if ct == "shared_cast_genre":
        return (f"Did any actor from '{kwargs['film1_title']}' also appear in a "
                f"{kwargs['genre']} film directed by {kwargs['person_name']}?")
    return str(kwargs)


def compare_claim(checker, llm, claim_type, expected=None, **kwargs):
    """
    Runs both the KG checker and the LLM on the same claim.

    Returns a result dict with keys:
        claim_nl, kg_verdict, kg_evidence,
        llm_verdict, llm_response,
        expected, kg_correct, llm_correct
    """
    nl = natural_language_claim(claim_type, **kwargs)

    # KG check (always)
    kg_verdict, kg_evidence = checker.check(claim_type, **kwargs)

    # LLM check (if available)
    if llm and llm.is_available():
        llm_raw = llm.ask(nl)
        llm_verdict, llm_response = llm.parse_verdict(llm_raw)
    else:
        llm_verdict  = "N/A"
        llm_response = "(Ollama not available)"

    result = {
        "claim_nl":    nl,
        "claim_type":  claim_type,
        "kwargs":      kwargs,
        "kg_verdict":  kg_verdict,
        "kg_evidence": kg_evidence,
        "llm_verdict": llm_verdict,
        "llm_response": llm_response,
        "expected":    expected,
    }
    if expected is not None:
        result["kg_correct"]  = (kg_verdict  == expected)
        result["llm_correct"] = (llm_verdict == expected)
    return result


def print_result(result, idx=None):
    """Pretty-prints a single comparison result."""
    sep = "─" * 70
    label = f"  Claim {idx}" if idx is not None else ""
    print(f"\n{sep}")
    print(f"{'CLAIM':>8}{label}: {result['claim_nl']}")
    if result.get("expected"):
        print(f"{'EXPECTED':>8}: {result['expected']}")
    print(f"{'KG':>8}: {result['kg_verdict']}")
    print(f"{'':>10}{result['kg_evidence']}")
    if result["llm_verdict"] != "N/A":
        print(f"{'LLM':>8}: {result['llm_verdict']}")
        # print first 2 lines of LLM response
        lines = result["llm_response"].strip().split("\n")
        for ln in lines[:2]:
            print(f"{'':>10}{ln}")
    if "kg_correct" in result:
        kg_mark  = "✓" if result["kg_correct"]  else "✗"
        llm_mark = ("✓" if result["llm_correct"] else "✗") if result["llm_verdict"] != "N/A" else "–"
        print(f"{'SCORE':>8}: KG {kg_mark}   LLM {llm_mark}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5  ─  FIXED BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
#
# NOTE FOR STUDENTS
# ─────────────────
# These benchmark claims use films and people that appear in the IMDb dataset.
# A few are intentionally FALSE (wrong year, wrong director, fabricated pairing)
# and a few will be UNKNOWN for small subsets of the data.
# Your task: run the benchmark, inspect where KG and LLM disagree, and explain why.
#
# INSTRUCTOR NOTE
# ───────────────
# To rebuild this benchmark for your own filtered dataset, run the script once
# with --build, then call checker.find_benchmark_examples() (stub below) to
# print valid claim kwargs from the actual graph.

BENCHMARK_CLAIMS = [
    # ── 1-hop: TRUE ────────────────────────────────────────────────────────
    {
        "claim_type": "directed_film",
        "kwargs":     {"person_name": "Christopher Nolan", "film_title": "Inception"},
        "expected":   TRUE,
        "notes":      "Well-known 1-hop fact; LLM should agree.",
    },
    {
        "claim_type": "acted_in_film",
        "kwargs":     {"person_name": "Leonardo DiCaprio", "film_title": "Inception"},
        "expected":   TRUE,
        "notes":      "1-hop; both systems should agree.",
    },
    {
        "claim_type": "film_year",
        "kwargs":     {"film_title": "The Dark Knight", "year": 2008},
        "expected":   TRUE,
        "notes":      "1-hop year check.",
    },
    {
        "claim_type": "film_genre",
        "kwargs":     {"film_title": "Parasite", "genre": "Thriller"},
        "expected":   TRUE,
        "notes":      "Genre membership; test IMDb's genre tagging.",
    },
    # ── 1-hop: FALSE ───────────────────────────────────────────────────────
    {
        "claim_type": "directed_film",
        "kwargs":     {"person_name": "Steven Spielberg",
                       "film_title": "The Dark Knight"},
        "expected":   FALSE,
        "notes":      "Plausible but wrong director — LLM may hallucinate TRUE.",
    },
    {
        "claim_type": "film_year",
        "kwargs":     {"film_title": "Inception", "year": 2009},
        "expected":   FALSE,
        "notes":      "Off-by-one year; classic LLM error.",
    },
    {
        "claim_type": "film_genre",
        "kwargs":     {"film_title": "Toy Story 3", "genre": "Horror"},
        "expected":   FALSE,
        "notes":      "Obvious genre mismatch; both should answer FALSE.",
    },
    # ── 2-hop: TRUE ────────────────────────────────────────────────────────
    {
        "claim_type": "co_starred",
        "kwargs":     {"person1_name": "Brad Pitt",
                       "person2_name": "Cate Blanchett"},
        "expected":   TRUE,
        "notes":      "Both appeared in The Curious Case of Benjamin Button (2008).",
    },
    {
        "claim_type": "same_director",
        "kwargs":     {"film1_title": "The Dark Knight",
                       "film2_title": "Inception"},
        "expected":   TRUE,
        "notes":      "Both directed by Christopher Nolan; 2-hop via director node.",
    },
    {
        "claim_type": "director_of_genre",
        "kwargs":     {"person_name": "Christopher Nolan", "genre": "Sci-Fi"},
        "expected":   TRUE,
        "notes":      "Nolan directed Interstellar (Sci-Fi); KG traversal test.",
    },
    # ── 2-hop: FALSE ───────────────────────────────────────────────────────
    {
        "claim_type": "co_starred",
        "kwargs":     {"person1_name": "Meryl Streep",
                       "person2_name": "Tom Hanks"},
        "expected":   FALSE,
        "notes":      "Both famous, often confused as co-stars; test plausible falsehood.",
    },
    {
        "claim_type": "same_director",
        "kwargs":     {"film1_title": "Avatar",
                       "film2_title": "The Avengers"},
        "expected":   FALSE,
        "notes":      "Cameron vs. Whedon; LLM might hallucinate.",
    },
    # ── 3-hop ──────────────────────────────────────────────────────────────
    {
        "claim_type": "shared_cast_genre",
        "kwargs":     {"film1_title": "Inception",
                       "person_name": "Christopher Nolan",
                       "genre":       "Action"},
        "expected":   TRUE,
        "notes":      "3-hop: actor(Inception) → acted_in → action film directed by Nolan.",
    },
    # ── UNKNOWN (entities in graph but claim unverifiable) ─────────────────
    {
        "claim_type": "directed_film",
        "kwargs":     {"person_name": "Zack Snyder",
                       "film_title":  "Midsommar"},
        "expected":   UNKNOWN,
        "notes":      "Snyder is in graph; Midsommar may or may not be (Ari Aster directed it). "
                      "If both are in graph → FALSE; if film absent → UNKNOWN.",
    },
]


def run_benchmark(checker, llm=None, verbose=True):
    """
    Runs all benchmark claims.  Returns a list of result dicts.

    ── STUDENT TASKS (Part B) ───────────────────────────────────────────────
    After running the benchmark:
    1. Count how many claims each system answered correctly.
    2. Find at least two claims where KG and LLM disagree.  For each:
       - Is the KG wrong (missing data), or is the LLM hallucinating?
       - What would it take to fix the wrong answer?
    3. Identify one UNKNOWN verdict from the KG.  Why is it unknown?
       What additional data would you need to make it TRUE or FALSE?
    4. Look at the 3-hop claim.  Draw the entity-relation path the checker
       traversed.  Why can't the LLM reliably answer 3-hop questions?
    """
    results = []
    kg_correct_total  = 0
    llm_correct_total = 0
    scored = 0

    for i, claim in enumerate(BENCHMARK_CLAIMS):
        r = compare_claim(checker, llm, claim["claim_type"],
                          expected=claim["expected"],
                          **claim["kwargs"])
        r["notes"] = claim.get("notes", "")
        results.append(r)
        if verbose:
            print_result(r, idx=i + 1)

        if r.get("expected"):
            scored += 1
            if r.get("kg_correct"):  kg_correct_total  += 1
            if r.get("llm_correct"): llm_correct_total += 1

    print("\n" + "═" * 70)
    print(f"BENCHMARK SUMMARY  ({scored} scored claims)")
    print(f"  KG accuracy:   {kg_correct_total}/{scored} = {kg_correct_total/scored:.0%}")
    if llm and llm.is_available():
        print(f"  LLM accuracy:  {llm_correct_total}/{scored} = {llm_correct_total/scored:.0%}")
    else:
        print("  LLM:           not available  (start Ollama to enable)")
    print("═" * 70)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PART 6  ─  INTERACTIVE REPL
# ─────────────────────────────────────────────────────────────────────────────

REPL_HELP = """
Available commands
──────────────────
  d  <person>  |  <film>      Directed  (did person direct film?)
  a  <person>  |  <film>      Acted in  (did person appear in film?)
  w  <person>  |  <film>      Wrote     (did person write film?)
  y  <film>    |  <year>      Year check
  g  <film>    |  <genre>     Genre check
  cs <person1> | <person2>    Co-starred
  sd <film1>   | <film2>      Same director?
  dg <person>  | <genre>      Director of genre?
  sc <film>    | <person> | <genre>   Shared cast + genre (3-hop)

  help         Show this menu
  quit / exit  Exit the REPL

Separate arguments with a pipe character  |
Example:  d Christopher Nolan | Inception
          cs Brad Pitt | Cate Blanchett
          sc Inception | Christopher Nolan | Action
"""

CMD_MAP = {
    "d":  ("directed_film",   ["person_name", "film_title"]),
    "a":  ("acted_in_film",   ["person_name", "film_title"]),
    "w":  ("wrote_film",      ["person_name", "film_title"]),
    "y":  ("film_year",       ["film_title",  "year"]),
    "g":  ("film_genre",      ["film_title",  "genre"]),
    "cs": ("co_starred",      ["person1_name","person2_name"]),
    "sd": ("same_director",   ["film1_title", "film2_title"]),
    "dg": ("director_of_genre", ["person_name", "genre"]),
    "sc": ("shared_cast_genre", ["film1_title", "person_name", "genre"]),
}


def repl(checker, llm=None):
    """Interactive read-eval-print loop for ad-hoc claim checking."""
    use_llm = llm and llm.is_available()
    llm_label = f"[LLM: {llm.model}]" if use_llm else "[LLM: not available]"
    print(f"\n{'═'*70}")
    print(f"  IMDb Knowledge Graph Fact-Checker  {llm_label}")
    print(f"{'═'*70}")
    print(REPL_HELP)

    while True:
        try:
            raw = input("kg> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw.lower() in ("help", "?", "h"):
            print(REPL_HELP)
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        parts = [p.strip() for p in raw.split("|")]
        cmd   = parts[0].split()[0].lower()
        # first arg may be rest of first token
        first_rest = parts[0][len(cmd):].strip()
        args = ([first_rest] if first_rest else []) + parts[1:]

        if cmd not in CMD_MAP:
            print(f"Unknown command '{cmd}'.  Type 'help' for usage.")
            continue

        claim_type, arg_names = CMD_MAP[cmd]
        if len(args) < len(arg_names):
            print(f"Need {len(arg_names)} argument(s): {', '.join(arg_names)}")
            continue

        # coerce year to int if needed
        kwargs = {}
        for name, val in zip(arg_names, args):
            if name == "year":
                try:    kwargs[name] = int(val)
                except: kwargs[name] = val
            else:
                kwargs[name] = val

        r = compare_claim(checker, llm, claim_type, **kwargs)
        print_result(r)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY  ─  Auto-discover valid benchmark examples from the graph
# ─────────────────────────────────────────────────────────────────────────────

def find_benchmark_examples(checker, n=5):
    """
    Prints n real claim examples discovered from the graph.
    Useful for instructors building a custom benchmark after
    re-filtering the IMDb data.

    Usage:  checker.find_benchmark_examples() inside a script
    or as: python kg_factcheck_lab.py --examples
    """
    G = checker.G
    print("\nAuto-discovered benchmark examples from the loaded graph:")

    # 1-hop directed_film
    count = 0
    for nc, attrs in G.nodes(data=True):
        if attrs.get("ntype") != "person": continue
        for _, tc, d in G.out_edges(nc, data=True):
            if d.get("rel") == "DIRECTED":
                print(f'  directed_film: person_name="{attrs["name"]}", '
                      f'film_title="{G.nodes[tc].get("title")}"')
                count += 1
                if count >= n: break
        if count >= n: break

    # 1-hop co_starred via CO_STAR edge
    count = 0
    for nc, attrs in G.nodes(data=True):
        if attrs.get("ntype") != "person": continue
        for _, nc2, d in G.out_edges(nc, data=True):
            if d.get("rel") == "CO_STAR":
                a2 = G.nodes[nc2]
                film_tc = d.get("film", "")
                ft = G.nodes[film_tc].get("title", "?") if film_tc else "?"
                print(f'  co_starred: person1_name="{attrs["name"]}", '
                      f'person2_name="{a2["name"]}"  [via {ft}]')
                count += 1
                if count >= n: break
        if count >= n: break


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IMDb KG Fact-Checking Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--build",     action="store_true",
                        help="(Re)build graph from IMDb TSV files and save cache.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run the fixed benchmark claim set.")
    parser.add_argument("--repl",      action="store_true",
                        help="Launch the interactive REPL.")
    parser.add_argument("--examples",  action="store_true",
                        help="Print auto-discovered claim examples from the graph.")
    parser.add_argument("--llm",       action="store_true",
                        help="Enable LLM comparison (requires Ollama running).")
    parser.add_argument("--model",     default=OLLAMA_MODEL,
                        help=f"Ollama model name (default: {OLLAMA_MODEL})")
    parser.add_argument("--cache",     default=str(CACHE_FILE),
                        help=f"Graph cache file (default: {CACHE_FILE})")
    args = parser.parse_args()

    # ── Load or build graph ───────────────────────────────────────────────────
    cache = Path(args.cache)
    if args.build or not cache.exists():
        if not args.build and not cache.exists():
            print(f"Cache '{cache}' not found — building graph now.")
            print("(Run with --build explicitly to rebuild after updating IMDb files.)\n")
        G = build_and_save()
        # update cache path in case user changed --cache
        if str(cache) != str(CACHE_FILE):
            save_graph(G, cache)
    else:
        G = load_graph(cache)

    checker = KnowledgeGraphChecker(G)

    # ── Set up LLM adapter ────────────────────────────────────────────────────
    llm = None
    if args.llm:
        llm = OllamaAdapter(model=args.model)
        if not llm.is_available():
            print(f"⚠  Ollama not reachable at {OLLAMA_URL}.")
            print(f"   Start it with: ollama serve")
            print(f"   Pull a model:  ollama pull {args.model}")
            print(f"   Continuing with KG-only mode.\n")
            llm = None

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if not (args.benchmark or args.repl or args.examples):
        # default: print graph stats
        n_films   = sum(1 for _, a in G.nodes(data=True) if a.get("ntype") == "film")
        n_people  = sum(1 for _, a in G.nodes(data=True) if a.get("ntype") == "person")
        n_genres  = sum(1 for _, a in G.nodes(data=True) if a.get("ntype") == "genre")
        print(f"\nGraph loaded:  {n_films:,} films · {n_people:,} people · "
              f"{n_genres} genres · {G.number_of_edges():,} edges")
        print("Use --benchmark, --repl, or --examples to explore.\n")
        return

    if args.examples:
        find_benchmark_examples(checker)

    if args.benchmark:
        run_benchmark(checker, llm=llm)

    if args.repl:
        repl(checker, llm=llm)


if __name__ == "__main__":
    main()
