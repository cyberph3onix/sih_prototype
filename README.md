# Quantum-Channel Digital Twin for Temporal Threat Detection

**SIH26141 — Quantum-Inspired Cyber Threat Detection for Digital Signature Security**

A judge-facing prototype. It simulates a quantum channel with Kraus operators,
generates realistic measurement outcomes, and runs a classical temporal
detector (HMM + SPRT/CUSUM) over them — then explains, on screen, exactly why
it reached the conclusion it reached.

---

## Scientific boundary — read this first

This is a **software-based quantum-channel digital twin**.

* There is **no physical quantum computer** and **no physical QDS hardware**.
* There is **no deployed quantum-digital-signature system**. The prototype
  produces a security *decision* and its supporting evidence; it does not
  perform key refresh, signature revocation, or session teardown.
* The HMM, SPRT, and CUSUM are **classical statistical algorithms**, not
  quantum algorithms. They are applied to measurement data that came from a
  simulated quantum channel.
* The "clean baseline" is a **modelled** value (a 3% error rate from a 6%
  depolarizing channel), not a figure learned from a real deployment.
* There is **no machine learning** anywhere in the detector.

Everything the dashboard displays is computed by the code in `qtwin/`. Nothing
is hard-coded, scripted, or replayed from a recording.

---

## Quick start

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
./run_demo.sh
```

That is the whole setup. On its first run the script creates a virtual
environment and installs the dependencies, then opens the dashboard at
**http://localhost:8501**. Later runs start in about two seconds. Press
`Ctrl-C` to stop it.

**Requirements:** Python 3.10 or newer (developed and tested on 3.12), and an
internet connection for that first install only. No GPU, no database, no API
keys, no accounts. Once installed, the demo runs entirely offline — it makes no
network calls at all.

**A different port:** `PORT=8600 ./run_demo.sh`

**Windows:** use WSL, or run the same three steps by hand:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

<details>
<summary>If setup fails</summary>

**`could not create a virtual environment`** — your Python is missing the
`venv` module (common on Debian/Ubuntu, and the case on the machine this was
built on). Install it, or install [`uv`](https://docs.astral.sh/uv/), then
re-run `./run_demo.sh`, which will use whichever it finds:

```bash
sudo apt install python3-venv                       # Debian / Ubuntu
curl -LsSf https://astral.sh/uv/install.sh | sh     # or uv, no sudo required
```

**`Port 8501 is already in use`** — `PORT=8600 ./run_demo.sh`

**Prefer to set it up by hand?**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

</details>

## Demoing it

Turn on **Judge Demo Mode** (top right) and press the numbered buttons **[1] →
[6]**; each one sets up a step of the story and prints a presenter note. Or
just press **START SLOW DRIFT ATTACK** — the alert fires about 9 seconds later.
**RESET TO CLEAN** always returns to a known-good state.

While playback runs, the five figures across the top stay live and the
per-section breakdowns are deferred; **press PAUSE** and every panel fills in
for the round you stopped on. The rhythm is *play a stretch, pause, explain*.

`DEMO_SCRIPT.md` has a full 2–3 minute narration, the questions judges tend to
ask, and the honest answers to them.

---

## The pipeline

```
  NORMAL QUANTUM CHANNEL
          |
  Quantum-channel digital twin            qtwin/channel.py, qtwin/scenarios.py
    Kraus operators -> CPTP map
          |
  Quantum measurement outcomes            qtwin/pipeline.py
    200 qubits/round, BB84 sifting,
    ~100 usable bits/round
          |
  Measurement statistics / fingerprint
    QBER, per-basis QBER, Z-outcome bias,
    Bell-state populations
          |
  HMM estimates hidden channel state      qtwin/detectors.py: ChannelStateHMM
    P(Clean) / P(Under Attack)
          |
  SPRT / CUSUM accumulate evidence        qtwin/detectors.py: SPRT, CUSUM
          |
  Security decision                       qtwin/explain.py
          |
  Dashboard alert                         app.py
```

### The channel model

Three elementary CPTP maps, composed per round:

| map | Kraus set | meaning |
|---|---|---|
| Depolarizing `p` | `sqrt(1-3p/4)I, sqrt(p/4){X,Y,Z}` | ordinary fibre/detector noise; gives QBER `p/2` exactly |
| Amplitude damping `gamma` | `diag(1, sqrt(1-g))`, `sqrt(g) * \|0><1\|` | physical link degradation; **asymmetric** between bases |
| Intercept-and-resend `mu` | `sqrt(1-mu)I, sqrt(mu/2){P0,P1,P+,P-}` | eavesdropper measuring in a random BB84 basis; gives QBER `mu/4` |

The intercept-resend set is exact, not an approximation: measure-and-reprepare
in a basis *is* complete dephasing in that basis, so the attack has a genuine
Kraus representation. `tests/test_channel.py` verifies the completeness
relation and both textbook error-rate identities.

### The detector

Every round yields `n` sifted bits of which `e` disagree. That pair is the only
input to every detector, so the static-vs-temporal comparison is like-for-like.

* **Log-likelihood ratio.** `LLR = e·log(p1/p0) + (n-e)·log((1-p1)/(1-p0))`,
  with `p0 = 3.0%` (H0, clean) and `p1 = 4.5%` (H1, attacked). Disagreeing bits
  add evidence; agreeing bits subtract. Zero average drift occurs at an error
  rate of **3.70%** — the detector's sensitivity floor.
* **HMM.** Two hidden states, binomial emissions, a sticky transition matrix
  (`P(clean→attack) = 0.05%` per round). Standard forward filter in log space.
* **SPRT.** Cumulative LLR against Wald's boundaries
  `A = log((1-beta)/alpha) = 11.46`, `B = log(beta/(1-alpha)) = -3.00`.
  Restarts at `B` instead of stopping, so it can monitor continuously.
* **CUSUM.** The same increments with a reflecting barrier at zero,
  `C_t = max(0, C_{t-1} + LLR_t)`, alarm at `C_t >= 11.46`. The barrier is the
  entire difference and it is what makes CUSUM a change-point detector. **This
  is the alarm the dashboard acts on.**

### The static baselines

Three fixed per-round thresholds, all fed the same data:

| baseline | line (at n≈100) | measured false-alarm rate on clean data |
|---|---|---|
| 11% BB84 abort threshold | 11.0% | 6.0% |
| Matched false-alarm budget | ~12.1% | 3.0% |
| Aggressive 3-sigma | 8.1% | **90.5%** |

The "matched" line is placed with the exact binomial distribution so a whole
400-round session carries the same 5% false-alarm budget we allow ourselves —
that is the fair comparison. The aggressive line is included precisely so the
obvious objection ("just lower the threshold") can be answered with a measured
number rather than an assertion. Our CUSUM's measured false-alarm rate on clean
data is **0.0%** over 200 runs.

---

## Measured results

Reproduce with `.venv/bin/python scripts/seed_study.py` (200 independent seeds
per scenario, ~1 minute):

```
FALSE ALARMS ON A CLEAN CHANNEL (200 runs x 400 rounds)
  cusum                  0.0%
  sprt                   0.0%
  hmm                    1.0%
  static_abort           6.0%
  static_matched         3.0%
  static_aggressive     90.5%

SLOW DRIFT (onset round 80)
  detected            200/200 runs
  our alert round     median 131  [112-154]
  lead over matched   median 42 rounds; positive in 93% of runs

REPLAY (onset 140)          lead median 52 rounds; matched never fired in 26 runs
CHANNEL MANIPULATION (100)  lead median 34 rounds; positive in 93% of runs
IMPERSONATION (onset 100)   lead median -2 rounds; positive in 12% of runs
```

The impersonation row is the honest counter-example and worth showing: against
a **loud** attack a fixed threshold is perfectly adequate and occasionally
beats us by a round or two. Our advantage is specifically against **quiet,
persistent** attacks. Do not claim otherwise.

The demo seed (`20079`) was picked from a 1000-seed sweep as a run whose
detection round and lead both sit near the sweep median — not a favourable one.

---

## Playback and the live strip

Playback advances the round by re-running the Streamlit script, and Streamlit
rebuilds every element whose content changed — including each Plotly chart.
When a dozen live readouts change at once, several times a second, the browser
paints part-way through the rebuild and the charts visibly blink.

Measured on this page (Chromium, `requestAnimationFrame` sampling, chart
teardowns per second during a slow-drift run):

| arrangement | teardowns/s | frames showing a gap |
|---|---|---|
| every readout live, free-running frames | 4.7 | 17% |
| live figures in one strip, paced frames | **1.8** | **4%** |

So the dashboard puts every per-round figure into a **single element** — the
five-column strip under the status banner — and defers the per-section
breakdowns until playback is paused. Frames are paced rather than free-running
(`SPEEDS` in `app.py`), since the blink rate tracks the rerun rate almost
exactly. Charts themselves are not the problem: six charts updating every frame
with no live text alongside them measures 0.4 teardowns/s.

Things that did *not* help, in case someone tries them again: `key=` on the
charts, keyed containers around them or around the text, `uirevision`, fixed
element geometry, and moving the dashboard into an `st.fragment` with
`st.rerun(scope="fragment")`.

## Layout

```
app.py                 Streamlit dashboard (the only UI)
qtwin/channel.py       Kraus operators, CPTP composition, Choi/Bell statistics
qtwin/scenarios.py     attack scenarios as schedules of channel parameters
qtwin/detectors.py     LLR, HMM forward filter, SPRT, CUSUM, static baselines
qtwin/pipeline.py      twin -> sampling -> detection; false-alarm calibration
qtwin/explain.py       decisions, "why?" reasoning, event timeline
qtwin/viz.py           Plotly figures
scripts/seed_study.py  offline robustness study
tests/                 142 tests, including headless tests of the app itself
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q          # everything (~85 s)
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_app.py   # fast (~6 s)
```

`tests/test_app.py` drives the real `app.py` through Streamlit's `AppTest`
harness — every button, every scenario, every demo step, playback, and the
jump-to-alert control — so a runtime error on some untried path is caught
before the judges find it. It needs no browser.

Notable invariants under test:

* the composed channel is CPTP for every parameter combination;
* depolarizing gives QBER `p/2` and intercept-resend gives `mu/4`;
* the LLR closed form equals the difference of exact binomial log-pmfs;
* the CUSUM trace is exactly the reflected cumulative sum of the LLRs;
* the alert round is exactly where the CUSUM first crosses its boundary;
* the clean scenario produces no alert, at any round, in the UI;
* the measured false-alarm rate lands on its design budget.

---

## Limitations to state to judges before they ask

1. **No physical quantum layer.** Everything about the channel is simulated.
   No hardware, no deployed QDS system, no real key material.
2. **No cryptographic response is executed.** The prototype outputs a decision
   and its evidence. Re-keying, revocation and session teardown are labelled
   *recommended response* and are not implemented.
3. **The clean baseline is assumed, not learned.** A real deployment would
   estimate `p0` from a trusted calibration period; drift in `p0` itself would
   need re-estimation, which this prototype does not do.
4. **Attacker model.** The adversary is modelled as a memoryless CPTP map with
   parameters that vary between rounds. An adversary who deliberately shapes
   its statistics to sit exactly at our zero-drift point (3.70%) is not
   detected by this configuration, and one who reacts adaptively to our alerts
   is out of scope.
5. **Replay is modelled at the statistics level.** At the sifted-error-rate
   level a replayed state is indistinguishable from an intercepted-and-resent
   one, so we model it with the same Kraus set and an abrupt onset. The twin
   does not reason about freshness or nonces.
6. **The detector uses only the pooled sifted error count.** The per-basis
   fingerprint (Z vs X error rate, Z-outcome bias) is displayed and does
   separate eavesdropping from link damping, but it does not currently feed the
   sequential test.
7. **Loud attacks do not need us.** Against a high-intensity burst a fixed
   threshold performs about as well or slightly better. See the impersonation
   row above.
8. **Single-link, single-hop.** No network topology, no multi-party trust, no
   authentication of the classical channel.

## Design decisions worth defending

* **`p1 = 4.5%`, not 6%.** The log-likelihood ratio has zero drift at 3.70%,
  the midpoint in likelihood terms. Setting `p1` closer to the baseline lowers
  that floor and lets the detector work on quieter attacks, at the cost of
  slightly noisier evidence.
* **CUSUM, not SPRT, raises the alarm.** They share increments and the same
  boundary; CUSUM's reflecting barrier at zero is what stops a long clean
  period from building a negative reserve that a later attack has to climb out
  of. Both are drawn on the same chart so the difference is visible.
* **~100 sifted bits per round.** Small enough that a single round is genuinely
  ambiguous, which is the regime where pooling across time actually matters.
  With thousands of bits per round a per-round test would do fine on its own.
