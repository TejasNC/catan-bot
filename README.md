# Catan AI Bot in Python: a 2-Player Settlers of Catan Agent

An open-source **Python Catan bot** and game-playing AI for 2-player **Settlers of Catan**. This
research agent is built on the [catanatron](https://github.com/bcollazo/catanatron) engine and
combines a hand-built value function, short-horizon planning, a learned opening, and self-play
tuning. It supports the standard 10-VP rules with no player trading and with bank and port trades
enabled.

The strongest agent plays within a tight CPU budget—about **1 second of thinking per game** on one
core—and wins about **4 games in 5** against catanatron's `ValueFunctionPlayer`. The repository also
includes a reproducible evaluation harness, benchmark bots, experiments, and examples for building
your own Catan game agent.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

| Agent | vs catanatron `ValueFunctionPlayer` (VF) | vs `AlphaBetaPlayer` depth 2 | Mean CPU / game |
|---|---|---|---|
| `army/v5`: value function + self-play-tuned weights | 0.79 (1000 games) | 0.88 (60 games, AB unclocked) | ~160 ms |
| **`army/comboop1`**: v5 + turn planning + regime weights + learned opening | **0.82** (1000 games) | — | ~280 ms |

*All games are played through the clocked harness in `catan_eval/`: each agent runs in its own
process under a CPU budget of 300 ms + 12.5 ms per decision (max 1.5 s per game), and sees only
what a player can see (the opponent's unplayed dev cards are resampled).*

**VF was never used as a training or tuning opponent**, only as a held-out benchmark.

---

## What this Catan AI includes

- A fast Python agent for 2-player Settlers of Catan.
- A value function that scores victory points, production, expansion, ports, resource readiness,
  development cards, army and road races, and discard risk.
- Hidden-information handling through one sampled opponent world per decision.
- Short lookahead for trades, builds, knights, Road Building, and opening settlement placement.
- Self-play SPSA tuning, reproducible clocked matches, and a bot army for benchmarking.

## How the Catan game-playing AI works

1. **One sampled world per decision.** In 2-player Catan the opponent's resource hand is derivable,
   so the only hidden information is their unplayed dev cards.
2. **Every legal action is simulated and scored** with a hand-built value function in victory-point
   units:

   $$V = \text{Score(me)} - k \cdot \text{Score(opp)}$$

   Score covers actual VPs (hidden VP cards included), robber-aware production by resource,
   diversity, expansion reach, ports, hand readiness, discard risk, dev cards, and army and road leads.
3. **Random outcomes are scored in expectation**: dev-card draws, robber steals.
4. **Short lookahead where it pays**:
   - trade, then the best build;
   - knight, then the best robber move;
   - Road Building, then the two best roads;
   - top-3 first moves played greedily to the end of the turn.
5. **Opening**: a learned model of P(win | the four opening settlements), with a snake-draft search
   that predicts the opponent's picks.
6. **Weights** were tuned by **SPSA self-play** ($\theta^+$ vs $\theta^-$ of our own bot).

The main lesson: about 26 of our ~32 points over VF came from *what the bot values*, not how far it
searches. A logistic regression that predicted game winners with 78 % accuracy won only 10 % of its
games as a player. See [`docs/STORY.md`](docs/STORY.md) and [`docs/INSIGHTS.md`](docs/INSIGHTS.md).

## Version history

| Version | Change | vs VF (≥ 400 games) |
|---|---|---|
| VF | catanatron `ValueFunctionPlayer` | 0.50 |
| v1 | value function in VP units, chance in expectation, trade→build lookahead | 0.635 |
| v2 | + knight / Road Building follow-up | 0.683 |
| v3 | + snake-draft opening | 0.726 |
| v5 | + SPSA self-play weights | 0.791 |
| combo | + regime weights + turn planning | 0.793 (52.4 % head-to-head vs v5, 3000 games) |
| comboop1 | + learned opening (op1) | 0.820 |
| comboop1_fix | + decision-audit fix (no pre-roll Monopoly) + stricter clock guard | 0.770 (400 games, fresh seeds) |

---

## Repository layout

| Path | What it is |
|---|---|
| [`army/`](army/README.md) | **Every bot**, each self-contained (an `agent.py` with an `Agent` class). The strongest is `comboop1` |
| `bot/` | Research agent: v5 defaults plus every experimental switch (regime, linear eval, robber weight, …) |
| `catan_eval/` | Match harness: clocked agent processes, hidden-information masking, reproducible seeding |
| [`tools/`](tools/README.md) | Evaluation (`par_eval`, `h2h`, `gauntlet`, `parallel_eval`), tuning (`spsa`), learning (`gen_data`, `fit`), decision `audit` |
| `experiments/` | Variant wrappers for every experiment, SPSA logs, audit data, raw outputs |
| [`docs/`](docs/) | [`STORY.md`](docs/STORY.md) (how the bot was built), [`INSIGHTS.md`](docs/INSIGHTS.md) (full results log), [`EXPLAIN_comboop1.md`](docs/EXPLAIN_comboop1.md) (how the strongest bot works), [`results/`](docs/results/) (per-experiment write-ups) |
| `examples/` | Minimal agents: catanatron's `ValueFunctionPlayer` and a random agent |

## Quick start: run the Python Catan bot

Use **Python 3.11**. The engine is a pinned catanatron commit, so do not install catanatron from PyPI.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m catan_eval.selfcheck
```

Play the strongest agent against catanatron's ValueFunctionPlayer, with the CPU clock on:

```bash
python -m catan_eval.evaluate --a army/comboop1/agent.py --b examples/valuefn_agent.py --games 200
python tools/par_eval.py --a army/comboop1/agent.py --b army/valuefn/agent.py --games 1000 --procs 4
```

Head-to-head between two bots (paired boards), or one bot against the whole army:

```bash
python tools/h2h.py army/comboop1/agent.py army/v5/agent.py 600
python tools/gauntlet.py army/comboop1/agent.py 400 4 4000000 --skip ab2
```

## Build your own Catan agent

An agent is a folder with an `agent.py` defining:

```python
class Agent:
    def decide(self, obs, playable_actions) -> int:
        """Return the index of the chosen action in playable_actions."""
```

`obs` (see `catan_eval/obs.py`) exposes the board, both players' hands, public VPs and played dev
cards, and `obs.sample_world(rng)`: a full catanatron `Game` consistent with what you can see. Use it
for lookahead. `obs.cpu_left` is the remaining CPU budget for the game.

## Authors

Uday, Avinash and Tejas.

## Licence

GPL-3.0-or-later (catanatron is GPL-3.0): see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md) for
third-party credits. Much of the code and analysis was written with Claude Code; every number in
this repository comes from a run in it.
