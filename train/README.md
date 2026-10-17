# Train your own controller

The controller that drives the arm is a page of hand-written rules — the open-loop pick: the camera says where the
object is, the arm goes there, descends, closes, lifts, carries, lets go. The rules sit on a small *environment* that
any policy can use, hand-written or learned. This folder trains one, three ways, in plain JavaScript with no
libraries, so every step of the maths is in a file you can read. It is the same three lessons as the Hacker Squad
game's `train/` folder, on a different world, with the same network file.

Everything a controller does fits in one loop:

```
obs    = observe(sim)        what it sees: 21 numbers (the camera's offset, the wrist sensor's, the gripper, the bin…)
action = policy(obs)         what it decides: one of 11 moves (approach, descend, grasp, nudge…)   ← the part you train
cmd    = act(sim, action)    how that becomes a command: a point for the fingertips (the inverse kinematics does the rest) and open / close
reward = …                   how good that was (a clean grasp, an object in the bin, a miss, a drop)
```

## Quick start

```sh
npm run train:clone                                   # lesson 1: copy the open-loop rules            (≈ 1 s)   → bots/clone.json
npm run train:reinforce -- --from bots/clone.json     # lesson 3: improve them by reward alone         (≈ 45 s)  → bots/reinforce.json
npm run train:evolve                                  # lesson 2: evolve from scratch (on normal)      (≈ 2 min) → bots/evolve.json
npm run play -- --bot bots/reinforce.json             # score it against the rules on the same situations (add --vs careful)
npm run bench -- --bot bots/reinforce.json --diff hard   # a whole shift, 20 situations: placed, missed, dropped, picks/min
node bench/trace.mjs --bot bots/reinforce.json --seed 7   # one episode, decision by decision, in the terminal
```

Then watch it: `npm start`, open `http://localhost:8000/?bot=bots/reinforce.json` (a bare `?bot=reinforce` works too;
so does dropping a `.json` or `.js` on the page). The viewer's **Train a controller** drawer runs the same three
lessons in a Web Worker with a live reward curve, and plays the result.

Node ≥ 22, nothing to install. The sim runs headless at about a million frames a second, so a 30-second episode
takes a millisecond or two; every script prints its progress and ends with a score against the rules — the plain
ones and the `careful` ones with the hand-written correction, the bar to reach.

## The pieces

| File | What it is |
|---|---|
| `src/bot/env.js` | The environment: `FEATURES` (what a controller sees), `ACTIONS` (what it can do), `sense` / `observe` / `act` |
| `src/bot/scripted.js` | The hand-written rules as a policy — the baseline and the teacher (`style: 'careful'` adds the correction) |
| `src/bot/nn.js` | A tiny neural network: forward pass, backpropagation, Adam, mutation, save / load (the game's file, unchanged) |
| `src/bot/bots.js` | Loading a controller: the rules, a `bots/*.json` network, or your own `bots/*.js` module |
| `train/core.js` | The reward, an episode, random situations, scoring, and the three lessons as functions (Node and browser) |
| `train/clone.mjs` | Lesson 1 — imitation learning |
| `train/evolve.mjs` | Lesson 2 — evolution |
| `train/reinforce.mjs` | Lesson 3 — reinforcement learning (policy gradient) |
| `train/play.mjs` | Score any controller against any other |
| `train/worker.js` | The lessons in a Web Worker, for the viewer's Train drawer |
| `bots/` | Trained networks and hand-written modules (`example.js`) |

### What the controller sees — `FEATURES`

Twenty-one numbers, each about −1…1, computed every frame from the sim: where the **camera** says the target object
is relative to the fingertips (x, z, distance — smoothed by a tracker, so the per-frame noise is gone but the
calibration bias is not), the fingertips' height, whether the **wrist sensor** sees the target and the offset it
reports (the truth — this is the correction signal, but only within 6 cm and below 13 cm), the gripper's aperture and
whether it is closing, holding, or holding by an edge (*slip*: a force sensor would feel it), the bin's offset and
whether we are over it, whether the tip is moving or has arrived, the target's size, how many objects are left, how
many grasps closed on air, and the time left. The list is in `env.js`; add a line there and every controller sees one
more thing (then retrain).

### What it can do — `ACTIONS`

Eleven moves: `approach` (to the hover height above where the camera says the target is), `descend` (straight down
to the grasp height, where we are), `grasp`, `lift`, `carry` (to above the bin), `release`, four `nudge`s (6 mm along
±x / ±z, at this height — *the corrections*), and `wait`. Each is a *where to*: the env turns it into a point for the
fingertips and the arm's inverse kinematics does the trigonometry, so a policy never learns joint angles — it learns
**when** to do what, and when the camera is wrong. This is the design decision that makes the whole thing trainable
in minutes: the action space is small and every action is meaningful. (A nudge is 6 mm and a policy nudges while the
sensor's offset is over 4 mm: the residual is under 4 mm per axis, inside the clean-grasp tolerance of the smallest
object, and the two never oscillate.)

### The rules — `scripted.js`

Ten lines, read top to bottom, the first rule that fits wins: holding → lift, carry, release over the bin; closing →
wait, closed on air → open again; not above the camera's spot → approach; above it → descend; down → grasp. It is a
policy like any other (`(obs, s, mem) → action`), which is why lesson 1 can copy it and lessons 2–3 can be measured
against it. The `careful` style adds one rule before the descent — *the wrist sensor sees the object off-centre:
nudge towards it* — and that one rule takes it from half the objects to all of them on hard.

### The reward — `core.js`

```js
export const REWARD = { grasp: 10, slipGrasp: 2, lift: 3, place: 20, miss: -2, drop: -5, lost: -15, bump: -1,
                        closer: 0.1, align: 0.5, toBin: 0.1, reach: 2, tick: -0.02, done: 30 };
```

A clean grasp is worth 10 (once per object), a placed object 20, all of them 30 more; a grasp that closes on air costs
2, a dropped object 5, one knocked off the table 15; every decision costs 0.02 (time is money). The rest is **reward
shaping**: a little for every centimetre of new progress towards an object (`closer`), five times as much while low
and within the wrist sensor's reach (`align` — the correction, paid as it happens), a little for every centimetre
towards the bin with an object in hand (`toBin`), and 2 for reaching the pre-grasp pose. **The reward defines what
"good" means** — change it and the controllers learn something else. The shaping terms exist for the learners that
start from nothing: without them a random network never finds the first grasp, and lesson 2 never gets off the ground.
(They are also where reward hacking lives: the first version paid every grasp, and evolution found grasp–release–grasp
on the table worth more than the bin. Now it pays once per object.)

### Episodes and situations

An **episode** is 30 seconds in the cell with 1–3 objects at random spots and a random camera bias, drawn from a
seed, played until every object is in the bin (or lost) or time is up. A set of **situations** is a list of such
seeds; a controller's score is its mean reward over them. Training runs on *hard* by default (the camera off by up to
30 mm): on *easy* the open-loop rules already pick everything and there is nothing to learn. Every script ends by
scoring its controller on eight fresh situations it never trained on, against both kinds of rules on the same eight.

## Lesson 1 — imitation learning (`clone.mjs`)

The rules play 24 episodes; every decision is written down as (observation, action) — about 10 000 of them. A network
with one hidden layer of 32 units is fitted to predict the action from the observation: plain **supervised learning**
(minibatch gradient descent on the cross-entropy loss). Ten percent of the data is held out: the accuracy on it says
whether the network learned the rules or memorised the examples.

What to watch: the loss falls, the accuracy climbs past 98 % (held out too), in a second. The student plays like the
teacher — and fails like the teacher: it has never seen a nudge (look at the action balance it prints: 0 %), so it
trusts the camera just as blindly. **Imitation copies habits, mistakes included, and can never beat the teacher by
much: it has never seen a reward.** `--teach careful` copies the rules *with* the correction, and then the student
corrects too — supervised learning learns whatever it is shown (nudges are ≈ 5 % of that teacher's decisions;
`--balance` weights rare actions more — try with and without).

## Lesson 2 — evolution (`evolve.mjs`)

No gradients, no teacher: 64 random networks each play 4 episodes; the best 8 survive; their mutated copies
(Gaussian noise on every weight, `--sigma`) fill the next generation; 250 generations. Everyone in a generation plays
the *same* situations, or luck would decide who survives. It works on anything you can score — even a controller that
is not a network — and it is slow, because each individual plays whole episodes to earn one number.

What to watch: the generation's best and mean against the rules' score on the same situations (which swing from one
generation to the next — compare against the rules' column, not the last line). For fifty generations nothing is
placed and the best network is the one that approaches and waits (it earns the `closer` shaping and little else).
Around generation 100 one individual grasps *and* lifts *and* carries *and* releases over the bin — the whole sequence,
found by mutation — and from there the population climbs fast. On **normal** (the default for this lesson) it ends
close to the rules. On **hard**, from random weights, nothing is ever placed in 250 generations: a grasp on a biased
camera pays too rarely to be selected for, and the nudges would have to be found at the same time. That is what a
sparse reward looks like — and the fixes are the ones you would guess: shape the reward more, or train on the easy
setting first and carry on from there (`--from bots/evolve.json --diff hard --sigma 0.05`: a curriculum — in our run
100 more generations on hard did not improve it; making that work is an exercise).

## Lesson 3 — reinforcement learning (`reinforce.mjs`)

REINFORCE, the simplest **policy gradient**: the network plays by *drawing* actions from its own probabilities
(exploration); for every decision, the **return** is the discounted sum of the rewards that followed it
(`--gamma 0.95`: at 15 decisions a second, about 1.3 s of consequences count — a nudge is paid by the grasp that
follows it, not by the whole shift); the **advantage** is that return against the batch's average; then every action
taken is pushed up or down in probability in proportion to its advantage. An **entropy** bonus keeps the policy from
freezing. Adam takes the step. 24 000 episodes in batches of 32; about 45 seconds.

What to watch: the mean reward per batch (noisy, and worse at first than the clone it started from — the drawn actions
are random a few percent of the time, which costs objects during training; the final score is taken greedily), and the
**nudges** column: the share of decisions that are a nudge. It is 0.0 % for the first few thousand episodes — the clone
never nudged, so the network gives nudges almost no probability, and the entropy bonus has to pry that open — then it
climbs to 5–10 %, and the reward climbs with it. The trained controller places every object, and does it in two thirds
of the careful rules' time: nobody told it to be careful, only to be paid, so it nudges only when a grasp would
otherwise fail, and it skips the hover. Start it from the imitation controller — `--from bots/clone.json` — and it
begins where lesson 1 ended; from random weights it needs many more episodes. This "imitate, then improve by reward"
recipe is how most game-playing agents (and language models) are actually trained.

## Our runs (one laptop, default settings, seed 1)

Eight fresh situations on *hard* (1–3 objects each), 30 s each. "Shift" = `bench/run.mjs`: 20 situations × 3 objects.

| Controller | Trained in | Score | Placed | Missed | Per situation | Shift on hard: placed, picks/min | Shift on normal |
|---|---|---|---|---|---|---|---|
| rules (`scripted.js`) | — | 38.4 | 8 / 16 | 9× | 8.0 s | 70 %, 9.5/min | 97 %, 22.6/min |
| rules, careful (= `bots/example.js`) | — | 111.5 | 16 / 16 | 0× | 5.5 s | 100 %, 19.7/min | 100 %, 19.8/min |
| `bots/clone.json` | 1 s | 42.4 | 8 / 16 | 5× | 11.3 s | 65 %, 7.1/min | 90 %, 22.3/min |
| `bots/clone-careful.json` | 1 s | 111.4 | 16 / 16 | 0× | 5.3 s | 100 %, 20.3/min | 100 %, 20.5/min |
| `bots/reinforce.json` (from clone) | 45 s | 101.0 | 15 / 16 | 0× | **3.5 s** | 98 %, **29.2/min** | 98 %, 30.5/min |
| `bots/evolve.json` (from scratch, on normal) | 2 min 18 s | 25.1 | 5 / 16 | 4× | 18.6 s | 45 %, 4.0/min | 85 %, 11.6/min |

Scores: `npm run play -- --bot <spec> --seed 1001` (eight fresh situations on hard, the ones every lesson ends with);
shifts: `npm run bench -- --bot <spec> --diff hard|normal` (20 situations × 3 objects). The evolved controller's last
generation scored 133.9 on its own four situations and 60.6 on eight fresh ones on normal — it fits the situations
it was selected on more than it generalises (four episodes per individual is few; `--episodes 8` is slower and
steadier), and on hard it has never seen a camera this far off.

The runs are seeded (`--seed`), so the same command gives the same numbers; the sim itself is deterministic.

## Plug in your own controller

Three levels, from the easiest:

1. **Train one** — any of the scripts above writes a `bots/<name>.json`; `?bot=<name>` plays it.
2. **Write a policy** on the environment — copy `bots/example.js`: `policy(obs, s, mem)` returns an action index; you
   get the same 21 numbers a network sees, and the raw scan `s` if you want more. Fifteen lines make a controller that
   beats the rules.
3. **Write a raw controller** — a module that exports `createBot()` returning `(sim) → { goal: {x, y, z}, grip: 'open' | 'close' | null }`:
   read anything in the sim, write the command yourself.

Every kind runs the same way: `?bot=bots/<file>` or a file dropped on the viewer; `npm run bench -- --bot <file>` for
a shift; `npm run play -- --bot <file>` for the score.

## Concepts, in the order they appear

| Term | Here |
|---|---|
| observation / feature | one of the 21 numbers in `observe()` — the controller's senses, designed by you |
| action space | the 11 `ACTIONS`; small and meaningful on purpose |
| policy | any `(obs) → action`: rules, a network, or your code |
| environment | `env.js` + the sim: it turns actions into frames and frames into observations |
| open loop / closed loop | trust the plan (the camera) / correct on feedback (the wrist sensor) — the lesson's subject |
| reward | `REWARD` in `core.js`; the one definition of "good" every learner shares |
| episode | 30 s in the cell; a sample of the task from one seed |
| supervised learning | lesson 1: inputs with the right answers attached (the teacher's actions) |
| loss (cross-entropy) | −log of the probability the network gave the teacher's action |
| generalisation / held-out set | the 10 % never trained on: the accuracy that counts |
| class imbalance | 0 % nudges in the naive teacher's data, 5 % in the careful one's: `--balance` |
| evolution: mutation, selection, elites | lesson 2; `--sigma`, `--elite` |
| common random numbers | one generation, one set of situations, so scores compare |
| sparse reward | lesson 2 on hard: the first grasp is never found by chance |
| curriculum | evolve on normal, then carry on on hard |
| exploration vs. exploitation | lesson 3 draws actions (explores) while training, takes the best (exploits) when playing |
| return and discount | the reward that followed a decision, weighted by `gamma^t` |
| advantage and baseline | the return minus the batch average: "better than usual" is positive |
| policy gradient | push up the log-probability of what worked, in proportion |
| entropy bonus | a little randomness kept in the policy, so it keeps trying things — here, what lets the nudges appear |
| pre-train, then fine-tune | `reinforce --from clone`; `evolve --from bots/evolve.json --diff hard` |
| reward shaping | `closer`, `align`, `toBin`, `reach`: a little reward on the way, or the first grasp is never found |
| reward hacking | grasp-release-grasp on the table paid more than the bin, until grasps paid once per object |
| seed / reproducibility | `--seed`; the sim itself is deterministic |

## Things to try

- **Change the reward.** Make a miss free — does the controller stop nudging? Make a drop cost 50 — does it slow down?
- **Blind the controller.** Delete `wristDx` and `wristDz` from `FEATURES` (retrain): can it still correct from
  `wristDist` alone (how far, not which way)? Delete the whole sensor: is there anything left to learn?
- **The drift.** `--diff extreme`: the camera's bias turns while the arm works. Which lesson copes?
- **Bigger brain.** `--hidden 64`, or a second hidden layer (`createNet([21, 32, 32, 11])` — `nn.js` supports any depth).
- **More data vs. more epochs.** `--episodes 60 --epochs 10` against `--episodes 6 --epochs 100`: watch the held-out accuracy.
- **Reinforce from scratch.** `train:reinforce` without `--from`: how many episodes before anything is placed?
- **A new action.** Add `'peek'` to `ACTIONS` (descend to the sensor's height and stop) with a line in `act()`, and see
  whether any learner uses it.
- **Beat the rules by hand.** Edit `bots/example.js` until `npm run play -- --bot bots/example.js --vs bots/reinforce.json` wins.
