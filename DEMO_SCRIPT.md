# 2–3 minute judge demo

Before the judges arrive:

```bash
./run_demo.sh          # http://localhost:8501
```

Leave it on **No Attack**, round 0, **Judge Demo Mode ON**. Nothing else needs
to be touched. If anything ever looks wrong, press **RESET TO CLEAN**.

**One thing to know about the layout.** The five figures across the top are the
live strip, and they are the whole argument in one line:

```
ROUND 137 | THIS ROUND 10.10% | FIXED THRESHOLD  SAFE | P(ATTACK) 100% | EVIDENCE 119%
```

While playback runs, that strip and the charts are live and the per-section
breakdowns are replaced by a "pause to see the full breakdown" note. **Press
PAUSE and they all fill in for the round you stopped on.** That is deliberate:
fewer things redrawing per frame means the charts stay steady on a projector,
and nobody reads dense text while it is animating anyway. So the rhythm of the
demo is *play a stretch, pause, explain*.

---

### 0:00 — What this is (15 s)

> "This is a digital twin of a quantum channel. It's a software model — we
> don't have quantum hardware. It simulates the channel with Kraus operators
> and produces measurement outcomes, and then a classical detector watches
> those outcomes over time."

Point at the header: **CHANNEL STATE ● CLEAN**, and the disclaimer line under it.

### 0:15 — The clean baseline (20 s)

Press **[1 Clean baseline]**.

> "Sender, quantum channel, receiver. Two hundred qubits a round; about a
> hundred survive sifting. On a clean channel about 3% of them disagree.
> That's this flat white line — the 20-round mean. The grey spikes behind it
> are individual rounds; look how much a single round bounces around."

### 0:35 — Start the attack (25 s)

Press **[2 Start slow drift]**. It plays; the alert appears in about 9 seconds.

> "Now an eavesdropper starts intercepting qubits — slowly. It ramps from zero
> to thirteen percent of the traffic. Watch the individual rounds: nothing
> looks alarming. Watch the white pooled line: it creeps up."

### 1:00 — Evidence accumulating (25 s)

Press **[3 Show evidence]**, scroll to Sections 4 and 5.

> "The HMM has two hidden states, Clean and Under Attack. We never see them —
> we only see measurements. Its estimate is climbing. Below it, every round
> contributes a log-likelihood ratio: errors add evidence, agreements subtract
> it. The green curve is that running total. On a clean channel it sits at
> zero. It's been climbing for tens of rounds."

### 1:25 — Detection (25 s)

Press **[4 Detection]** (or **JUMP TO ALERT** at any time). Playback stops, so
every panel now shows its full reasoning for this round.

> "It crosses the decision boundary here, and we alert. The 'Why did we alert'
> panel gives the full audit trail — and note point four: **the round we
> alerted on was still below every fixed threshold.** No single measurement
> was decisive."

### 1:50 — The comparison (35 s)

Press **[5 Compare]**, scroll to Section 3.

> "Here's the same data drawn against the fixed thresholds a traditional
> detector uses. The 11% abort threshold is up here. The attack lives down
> here, and never gets near it. The fixed threshold doesn't fire until round
> 179 — forty-two rounds after us."

If a judge says *"just lower the threshold"* — that is the best question you
can get. Turn Demo Mode off → **Technical View → Calibration & robustness →
Run false-alarm calibration**:

> "We included exactly that detector — the aggressive 3-sigma line, which fires
> at round 22, before the attack even starts. On clean channels it false-alarms
> in **90% of sessions**. Ours: **zero**. The comparison threshold is placed to
> carry the same 5% false-alarm budget we allow ourselves."

### 2:25 — Close (15 s)

Press **[6 Reset]**.

> "Instead of waiting for one measurement to cross a fixed threshold, we
> continuously observe the channel and accumulate evidence over time, so a slow
> attack can be detected earlier."

---

## Likely questions, and honest answers

**"Is this running on a quantum computer?"**
No. It's a software digital twin — Kraus-operator simulation of the channel.
The detection layer is classical statistics: an HMM, Wald's SPRT and Page's
CUSUM. None of those is a quantum algorithm.

**"Is the attack result hard-coded?"**
No. Change the seed in Technical View → Run configuration and re-run; the alert
round moves. Over 200 seeds we detect the slow drift 200/200 times with a
median alert at round 131, range 112–154.

**"You picked a good seed."**
The demo seed came from a 1000-seed sweep, chosen because its detection round
and its lead both sit near the **median**. Technical View → Calibration &
robustness → Run detection-lead study shows the whole distribution live.

**"Does your detector always win?"**
No, and we show a case where it doesn't. Against the **Impersonation** scenario
— a loud, high-intensity burst — the fixed threshold is fine and beats us by a
round or two. Our advantage is specifically against quiet, persistent attacks.

**"What are the limitations?"** — see the list in README.md; state them before
being asked.
