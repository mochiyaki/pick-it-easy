# bots/

Controllers that can drive the arm. Pick one with `?bot=<spec>` on the viewer's address (a bare name means
`bots/<name>.json`), by dropping the file on the page, or with `--bot` on `bench/run.mjs` and `train/play.mjs`.

- `*.json` — trained networks (written by `train/clone.mjs`, `train/evolve.mjs`, `train/reinforce.mjs`, or the viewer's
  Train drawer). The file holds the weights, the feature and action names they were trained on, and how they were
  trained (`trained: { algorithm, episodes, score, … }`).
- `*.js` — hand-written policies, like `example.js`; copy it and change the rules.

| File | What it is | Reward on hard (8 fresh situations) |
|---|---|---|
| `example.js` | the closed-loop pick in 15 lines, on the observation vector alone | 111.5 (= the careful rules) |
| `clone.json` | lesson 1: imitation of the open-loop rules — never nudges | 42.4 (the rules: 38.4) |
| `clone-careful.json` | lesson 1 with the careful rules as teacher — nudges, because the teacher did | 111.4 |
| `reinforce.json` | lesson 3 from `clone.json`: found the nudges by reward alone, and is faster than the careful rules | 101.0, in two thirds of the time |
| `evolve.json` | lesson 2 from scratch, on normal: found the whole pick sequence with no teacher at all | 25.1 on hard (it was trained on normal: 60.6 there) |

See [train/README.md](../train/README.md) for how a controller sees the cell and how to train one.
