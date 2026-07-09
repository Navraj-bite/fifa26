# 2026 World Cup Predictor

Trained on 154 years of international football history, this predicts the outcome
of the **remaining** matches of the 2026 FIFA World Cup and Monte Carlo-simulates
the rest of the bracket. First run: **2026-07-07**, with the Round of 16 still in
progress (Argentina vs Egypt and Switzerland vs Colombia had not yet kicked off).
The tournament resolves on 2026-07-19 — this is a timestamped, checkable
prediction, not a backtest of a dead dataset.

## Method

1. **Data**: [`international-football-results-from-1872-to-2017`](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
   (Kaggle, martj42), an auto-updated dataset of ~49,500 international matches.
   As of this run it lagged real-world results by 1-2 days for six 2026 Round of
   16 matches; those six are patched in from confirmed press reporting (sources
   below). Two Round of 16 matches were genuinely unplayed at the time of this
   run — those are left alone and are exactly what the model predicts live.

2. **Elo ratings** ([`src/elo.py`](src/elo.py)): team strength isn't a column in
   the data, it's computed by replaying all ~49,500 matches in order. Standard
   [eloratings.net](https://www.eloratings.net)-style update: goal-difference
   weighting, a competition-importance K-factor (60 for World Cup matches down
   to 20 for friendlies), and a 100-point home-advantage adjustment for
   non-neutral venues.

3. **Features** ([`src/features.py`](src/features.py)): each side's pre-match
   Elo, the Elo differential (home-advantage-adjusted), neutral-venue flag,
   competition-importance weight, and each team's rolling goal difference over
   its last 10 matches.

4. **Models** ([`src/model.py`](src/model.py)): a multinomial logistic
   regression baseline and an XGBoost classifier, both predicting win/draw/loss.
   Evaluated with a **time-based split** — trained on matches before
   2024-01-01, evaluated on 2024-01-01 through 2026-06-10 (the day before this
   World Cup kicked off). Never shuffled; sports outcomes are time-ordered.

5. **Backtest** ([`src/backtest.py`](src/backtest.py)): the model is re-fit on
   *all* data through 2026-06-10 (still zero 2026 World Cup matches) and
   scored against the 94 World Cup matches actually played so far. This is the
   headline result.

6. **Forward simulation** ([`src/simulate.py`](src/simulate.py)): the
   confirmed knockout bracket (verified via live web search, cross-checked
   across ESPN, Fox Sports, Al Jazeera, Olympics.com) is simulated 20,000 times
   using the model's win probabilities, producing a championship-probability
   table. Draws in a knockout tie are split 50/50 (penalty-shootout coin flip),
   and each matchup's probability is averaged across both home/away orderings
   since knockout venues are neutral.

## Results

### Generic time-based validation (2024-01-01 to 2026-06-10, pre-tournament)

| Model | Accuracy | Log-loss | n |
|---|---|---|---|
| Logistic regression | 59.9% | 0.871 | 2,543 |
| XGBoost | 59.9% | 0.869 | 2,543 |

### 2026 World Cup backtest (94 matches played, group stage through Round of 16)

| Model | Accuracy | Log-loss |
|---|---|---|
| Logistic regression | 63.8% | 0.857 |
| XGBoost | **64.9%** | **0.846** |

For reference, always-predict-home-win scores ~49% accuracy on this dataset's
historical base rate, and uniform random 3-way guessing scores 33.3% accuracy
and 1.099 log-loss. The model is meaningfully better calibrated than either.
Full per-match predictions: [`results/wc2026_backtest_detail.csv`](results/wc2026_backtest_detail.csv).

### Live Round of 16 predictions (not yet played as of 2026-07-07)

| Match | Win probability |
|---|---|
| Argentina vs Egypt | Argentina 82.2% / Egypt 17.8% |
| Switzerland vs Colombia | Switzerland 42.0% / Colombia 58.0% |

### Confirmed quarterfinal win probabilities

| Match | Win probability |
|---|---|
| France vs Morocco (Jul 9, Boston) | France 63.6% / Morocco 36.4% |
| Spain vs Belgium (Jul 10, LA) | Spain 64.4% / Belgium 35.6% |
| England vs Norway (Jul 11, Miami) | England 61.6% / Norway 38.4% |

### Championship probability (20,000-simulation Monte Carlo, as of 2026-07-07)

![Championship probability chart](results/championship_probabilities.png)

| Team | Reach SF | Reach Final | Win it all |
|---|---|---|---|
| Spain | 64.0% | 36.7% | **21.1%** |
| Argentina | 55.0% | 34.9% | 19.7% |
| France | 63.0% | 33.6% | 19.1% |
| England | 61.4% | 30.6% | 14.2% |
| Belgium | 36.0% | 15.3% | 6.8% |
| Morocco | 37.0% | 14.4% | 6.0% |
| Norway | 38.6% | 15.4% | 5.7% |
| Colombia | 24.3% | 11.5% | 4.7% |
| Switzerland | 15.2% | 6.3% | 2.3% |
| Egypt | 5.5% | 1.4% | 0.3% |

Full table: [`results/championship_probabilities.csv`](results/championship_probabilities.csv).

## Bracket structure (confirmed 2026-07-07)

```
QF1 Boston Jul 9:   France vs Morocco       ─┐
QF2 LA     Jul 10:  Spain vs Belgium        ─┼─ SF1 Dallas Jul 14  ─┐
QF3 Miami  Jul 11:  England vs Norway       ─┐                     │
QF4 KC     Jul 11/12: (Arg/Egypt) v (Swi/Col)┼─ SF2 Atlanta Jul 15 ─┼─ Final Jul 19, East Rutherford
```

This gets re-run as each round actually completes: after the July 9-12
quarterfinals, again after the July 14-15 semifinals, and a final grading after
the July 19 final.

## Sources for patched 2026 results

- Brazil 1-2 Norway (Jul 5): [Al Jazeera](https://www.aljazeera.com/sports/liveblog/2026/7/5/brazil-vs-norway-live-fifa-world-cup-2026-last-16), [FIFA.com](https://www.fifa.com/en/match-centre/match/17/285023/289288/400021532)
- Mexico 2-3 England (Jul 5): [ESPN](https://www.espn.com/soccer/match/_/gameId/760505/england-mexico)
- Portugal 0-1 Spain (Jul 6): [ESPN](https://www.espn.com/soccer/match/_/gameId/760506/spain-portugal), [CBS Sports](https://www.cbssports.com/soccer/news/portugal-spain-live-updates-world-cup-2026-score-result/live/)
- United States 1-4 Belgium (Jul 6): [ESPN](https://www.espn.com/soccer/match/_/gameId/760507/belgium-united-states), [NPR](https://www.npr.org/2026/07/06/nx-s1-5883842/2026-world-cup-fifa-usmnt-belgium-round-of-16)
- Quarterfinal/semifinal bracket: [Olympics.com](https://www.olympics.com/en/news/fifa-world-cup-2026-bracket-quarter-finals-full-schedule-live-updates), [Fox Sports](https://www.foxsports.com/stories/soccer/world-cup-bracket-live-quarterfinals-update-standings)

## Reproducing

```
pip install -r requirements.txt
python3 src/model.py       # trains models, writes results/models.pkl + metrics
python3 src/backtest.py    # scores the 2026 World Cup matches played so far
python3 src/simulate.py    # Monte Carlo forward simulation + probability table
python3 src/chart.py       # championship probability chart
```

## Repo layout

```
data/           raw Kaggle CSVs
src/
  load_data.py  load + patch known 2026 results
  elo.py        rolling Elo rating engine
  features.py   feature engineering (Elo diff, form, neutral, importance)
  model.py      logistic regression + XGBoost, time-based split
  backtest.py   2026 World Cup-specific backtest
  simulate.py   Monte Carlo bracket simulation
  chart.py      championship probability chart
results/        trained models, metrics, backtest detail, probability table/chart
```
