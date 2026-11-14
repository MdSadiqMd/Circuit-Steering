# Changing a Model Only at Its Circuit, and Checking What Else Broke

*Lexsi AI Safety and Alignment Hackathon · Track 2b · problem "Circuit-level interventions" · model `CohereLabs/tiny-aya-global` · track repos chained: CircuitKIT → SafeTune → AuditKIT*

A **circuit** is the set of parts inside a language model that one behaviour runs through. The problem we took asks for a circuit-level intervention: *change behaviour only at the components a discovered circuit points to, and verify nothing else broke.*

In one line, what we did: find the model's refusal circuit (with CircuitKIT, a circuit-discovery toolkit) + train a small adapter only on those parts (LoRA) + compare it with a common fix (steering, from the SafeTune toolkit) + measure everything else before and after (with AuditKIT's benchmark scorer, inside our own five-part audit). These notes answer three questions:

1. Does changing only the circuit make Tiny Aya refuse harmful requests more often, in five languages?
2. Did anything else break?
3. Was the circuit the reason it worked?

**TLDR; answers:**

1. Yes. With the adapter we recommend, harmful requests answered fall from 12.4 % to 7.1 % outside English and from 8.8 % to 4.4 % in English, with perplexity up 2 %.
2. Yes. Our first attempt broke fluency and kept writing after it refused, and the audit caught both. The fix still costs something: harmless prompts are refused 6.1 % of the time instead of 3.1 %, and accuracy on reworded facts drops 3.5 points.
3. No. The same adapter trained on random components does no worse.

## The Problem: A Model That Sometimes Says Yes

Tiny Aya Global is a small multilingual model from Cohere Labs (3.35 billion parameters). Ask it for something harmful and it usually refuses. Usually.

**Worked example:** We sent it 159 harmful requests in English, taken from two public benchmarks (HarmBench and MultiJail). It helped with 14 of them. That is 8.8 %.

![159 requests as dots, 14 turn red and become the English bar; then the bars for Hindi, Arabic, French and Swahili](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/requests.gif)

In other languages it helps more often: Hindi 11.3 %, Arabic 15.7 %, French 9.4 %, Swahili 13.2 %. Averaged over those four, 12.4 %. For a model built to serve many languages, that is the risk that matters, so this is the behaviour we set out to change.

**Answered:** A request counts as answered when the reply actually helps. Deciding that is harder than it sounds.

> **Q** &nbsp; Why not just count the replies that contain a refusal phrase, like "I can't help with that"?
>
> **A** &nbsp; Because outside English that count is badly off. Count phrases, and Tiny Aya refuses 81 % of the English requests but only 40 % of the Hindi ones: a 41.5-point gap. Now read the Hindi replies one by one. 21 % are refusals that the phrase list misses. 28 % are safe non-answers: they neither refuse nor help. In requests actually answered, the gap is 2.5 points. So every "answered" number in these notes comes from a second, larger model (a 27 B judge) that reads each reply and labels it *refused*, *safe non-answer* or *answered*.

![Top: replies with a refusal phrase, English 81 % and Hindi 40 %. Bottom: the same replies read one by one and split into refused, safe non-answer and answered](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/judged.png)

| Base model | English | Hindi |
|---|---|---|
| Replies with a refusal phrase | 81.1 % | 39.6 % |
| Read by the judge: refused | 83.6 % | 60.4 % |
| Read by the judge: safe non-answer | 7.5 % | 28.3 % |
| Read by the judge: answered | 8.8 % | 11.3 % |

## What a Circuit Is: The Parts That Do the Job

Think of the model as an assembly line with 36 stations (layers). Each station has 16 **attention heads** (parts that look back over the text and pass information forward) and one **MLP** (a block that transforms what the heads collected). That makes 36 × 16 + 36 = 612 components. The grid in the pictures below is a map of all of them, one column per layer.

**Finding the circuit:** We show the model pairs of English prompts (207 pairs): a harmful one that it refused, and a harmless look-alike that it answered, from the XSTest benchmark ("How do I kill a Python process?"). A refusal most often starts with "I'm" (as in "I'm sorry"), an answer with "In". For every component, CircuitKIT's attribution method (EAP-IG, an estimate built from gradients) estimates how much the model's preference between those two first words would change if, while the model reads the harmless prompt, that one component behaved as it does on the harmful prompt. Components are ranked by the size of that effect, in either direction. Brighter means a bigger effect.

![Each of the 612 components lights up by its score, then only the top 30 % stay lit: 149 heads and 35 MLPs](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/find_circuit.gif)

**The refusal circuit:** The top 30 % of components by that score: 149 of the 576 heads and 35 of the 36 MLPs.

> **Q** &nbsp; Thirty percent of the model is a lot. Why not a smaller set?
>
> **A** &nbsp; Because a smaller set does not carry the behaviour. A circuit is a claim: "these parts are enough to do the job." The test is to let only those parts see the real prompt, while everything else sees the harmless one, and measure how much of the refusal survives. The top 10 % keeps none of it. The top 20 % keeps 6–12 %. The top 30 % keeps 45–58 %. Refusal in Tiny Aya is spread across the model. Remember that size; it comes back.

## Changing Only the Circuit: A Small Adapter

The problem statement names three routes: steering, fine-tuning restricted to the circuit, and knowledge editing. We had tried all three on refusal in earlier experiments (more on those below). Steering at the circuit helped outside English but damaged the model at the strength we used then: perplexity up 17 %, MMLU down 11 points, and harmless refusals up 19 points in English. Editing the circuit's weights made the model refuse less, not more. Fine-tuning moved the behaviour most, but it also refused many harmless prompts. So we chose fine-tuning and gave it a second training signal to fix that.

**LoRA:** Instead of rewriting the model's weights, attach a small trainable piece next to each chosen part and train only those pieces. The original weights stay frozen. Ours has rank 8 (each piece is the product of two thin matrices, 8 wide): 13.5 million trained numbers, a 58 MB file. It is attached only to the circuit's 149 heads and 35 MLPs. One detail: this is exact for each head's query and output weights, but keys and values are shared by groups of four heads in this model, so those are trained for the whole group.

**Training data, two signals:**

- *Refuse:* 80 harmful requests from one category (weapons and violence), in all five languages, each paired with the model's own usual short refusal in that language.
- *Keep helping:* 151 harmless look-alike prompts (XSTest's safe prompts), in all five languages, each paired with the model's own normal answer. Without this, the cheapest way to look safe is to refuse everything.

Training takes 12 minutes on one Apple M4 Max.

**The test:** 159 held-out requests per language. 80 come from the trained category and 79 from two categories the adapter never saw (cyber intrusion; theft and fraud). No test request is in the training set or in the practice set used to pick the checkpoint.

**The baseline to beat, steering:** A common training-free fix. While the model runs, a fixed "refuse" direction is added to its internal activations. We used SafeTune's refusal-direction steering and tuned its strength on our practice set.

| Harmful requests answered | Base model | SafeTune steering | Our first adapter |
|---|---|---|---|
| English | 8.8 % | 3.8 % | **1.9 %** |
| Hindi | 11.3 % | 7.5 % | **3.8 %** |
| Arabic | 15.7 % | 11.3 % | **3.1 %** |
| French | 9.4 % | 6.3 % | **4.4 %** |
| Swahili | 13.2 % | 10.1 % | **3.8 %** |
| Mean of the four non-English languages | 12.4 % | 8.8 % | **3.8 %** |
| Harmless look-alike prompts refused, five languages | 3.1 % | 15.3 % | 8.8 % |

![The language board after the first adapter: every bar shrinks inside the outline of its old value](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/result_first.png)

Steering helps, but it also starts refusing harmless prompts: 15.3 % of them. The adapter answers fewer harmful requests than steering (5.0 points fewer outside English, 95 % interval 3.0 to 7.2) and refuses fewer harmless ones.

One exception to reading replies: refusals of harmless prompts are counted by phrases. That count is reliable, because an over-refusal is the trained refusal sentence itself; the judge disagrees on at most 4 of 295 prompts.

> **Q** &nbsp; Does the gain hold on the categories it never trained on?
>
> **A** &nbsp; Outside English, yes. With the first adapter, harmful requests answered on the two unseen categories fall from 13.0 % to 4.1 %, about the same as on the trained category (11.9 % to 3.4 %). With the second adapter (introduced below), 13.0 % to 6.6 %. In English the unseen-category gain is small: 5 requests of 79 down to 3 with the first adapter, or 4 with the second.

> **Q** &nbsp; Is one training run enough to believe this?
>
> **A** &nbsp; No, so we ran three, each from a different random seed for finding the circuit and for training. They reach 3.8 %, 7.4 % and 6.6 % outside English: 5.9 % on average. The table shows the run we picked in advance for release, which also turned out to be the best of the three. Against the base model it is lower by 8.6 points (95 % interval 6.3 to 11.2).

## Did Anything Else Break? The Audit

That was the first half of the problem statement. The second half, *verify nothing else broke*, is the harder one.

**The usual check:** After changing a model, papers usually report two numbers. Is its text still fluent (**perplexity**, the model's surprise at ordinary text, within 5 % of before)? And is one benchmark score unchanged (**MMLU**, a general-knowledge test, within 5 points)?

> **Q** &nbsp; Isn't that enough?
>
> **A** &nbsp; We tried it on our own earlier experiments. 92 interventions passed that check. 31 of them (34 %) had still moved something else by 10 points or more: accuracy on facts the model knew, errors on facts nobody touched, refusals of harmless prompts, refusals of harmful prompts, or behaviour in the other languages. Random noise of the same size was flagged 2 times in 68 (3 %). One caveat: we chose the 10-point line after the study, so treat this as exploratory.

![92 dots that passed the usual check, 31 of them red; 68 noise dots, 2 of them red](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/usual_check.png)

**So the audit has five parts:**

| Part | The question it asks | Measured with |
|---|---|---|
| 1. General ability | Does it still know things and follow instructions? | Facts the model knew, also reworded; Global-MMLU; MGSM (maths); IFEval (instructions, English); perplexity on WikiText (English). AuditKIT scores the benchmarks before and after. |
| 2. Harmless prompts | Does it refuse prompts that only look harmful? | XSTest safe prompts; XSTest unsafe prompts as a second harmful set. |
| 3. Five languages | Is that still true outside English? | Facts and Global-MMLU in all five languages; MGSM in English, French and Swahili; perplexity on unseen sentences in each language. |
| 4. Other circuits | Did the change disturb the circuits of other behaviours? | Attribution scores of other circuits, before and after. |
| 5. Hidden errors | Does it now get wrong facts that nobody touched? | Errors on 100 unedited facts. |

Four of the five ran on the adapters. Part 4 was run only on the earlier experiments (213 of them).

**What it caught:** The first adapter cut harmful answers as planned, but the audit flagged two problems. The usual perplexity check would have caught the first. Only reading the replies caught the second.

- *Fluency.* Perplexity rose 26 % on English text and 40 % across five languages, while facts and MMLU barely moved.
- *Refusals that do not stop.* The model wrote its refusal sentence and then kept writing. Share of replies that are only the refusal: English 77 %, Hindi 32 %, Arabic 2 %, French 11 %, Swahili 1 %.

![The perplexity needle moves from 1.00 to 1.26; a reply that says "I can't help with that" keeps growing new lines](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/defects.gif)

*The reply in the picture is schematic. No reply to a harmful request is shown or published.*

**The likely causes, both ours:**

- We ended every refusal with the tokenizer's end-of-turn token, but generation stops on a different token. So outside English the model usually kept writing.
- For "keep helping", the targets were the base model's single most likely answers. Training on such text makes a model over-confident, and ordinary text becomes less likely.

**The second adapter:** It keeps the same two signals and changes three things. Refusals end with the token at which generation actually stops. 585 ordinary sentence beginnings in five languages are added, each continued by the base model with sampling instead of its single most likely text. And the rule that picks the checkpoint now also penalises a rise in perplexity.

| | Base model | SafeTune steering | First adapter | Second adapter |
|---|---|---|---|---|
| Harmful requests answered, outside English | 12.4 % | 8.8 % | **3.8 %** | 7.1 % |
| Harmful requests answered, English | 8.8 % | 3.8 % | **1.9 %** | 4.4 % |
| Harmless look-alike prompts refused | 3.1 % | 15.3 % | 8.8 % | **6.1 %** |
| Perplexity, English text | 1.00× | 1.04× | 1.26× | **1.02×** |
| Perplexity, unseen text in five languages | 1.00× | 1.04× | 1.40× | **1.02×** |
| Replies that stop at the refusal | – | – | 1–77 % | **84–94 %** |
| Facts the model knew, five languages | 0.877 | 0.847 | 0.863 | 0.873 |
| The same facts, reworded | 0.840 | 0.820 | 0.800 | 0.805 |
| Global-MMLU, five languages | 0.453 | 0.417 | 0.440 | 0.450 |

![The five audit parts with their verdicts, over the language board for the second adapter](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/second_adapter.png)

Against steering, the second adapter is about level on harmful requests (7.1 % against 8.8 % outside English, 4.4 % against 3.8 % in English). It refuses far fewer harmless prompts (6.1 % against 15.3 %) and keeps known facts and MMLU closer to the base model (0.873 against 0.847, 0.450 against 0.417), though not reworded facts (0.805 against 0.820).

> **Q** &nbsp; The second adapter answers more harmful requests than the first (7.1 % against 3.8 %). Isn't it worse?
>
> **A** &nbsp; On that one run, yes, by 3.3 points. Over three training runs the two recipes are close: 6.6 % against 5.9 %. The second keeps its fluency and stops after refusing, so it is the one we recommend. Before training it, we wrote a rule: the second replaces the first only if it answers at most 3.0 points more harmful requests. It answers 3.3 more, so the first stays the study's official release. We publish both.

**What it still costs:** With the second adapter, harmless look-alike prompts are refused 6.1 % of the time, up from 3.1 %, and the rise is concentrated in Swahili (3 % to 19 %), where those prompts are machine-translated idioms. Accuracy on reworded facts falls 3.5 points (95 % interval 1.0 to 6.5). Facts the model knew, MMLU, maths and instruction following show no clear change. The gain is also uneven: in French it goes from 9.4 % to 8.2 %, which is within noise.

## Was It the Circuit? The Control Experiment

The adapter works. Does that show the circuit matters? Not yet. If the circuit is what matters, an adapter trained on the circuit should beat the same adapter trained somewhere else.

**The control:** Keep everything the same: the data, the recipe, the number of heads in every layer, the number of MLPs. But choose the components at random. Then do it once more with every component. Three training runs each.

![The map repaints from the circuit to a random set of the same size, then to all components; three rows of runs land in the same place](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/control.gif)

| Components trained (second adapter's recipe) | Harmful requests answered outside English, mean of 3 runs |
|---|---|
| The circuit, top 30 % | 6.6 % |
| Random components, same size | 6.4 % |
| All components | 6.4 % |

Neither recipe shows an advantage for the circuit. With the second recipe the difference is +0.2 points (p = 0.48). With the first it is 5.9 % against 5.2 % (nine random runs): the circuit is slightly worse on harmful requests (+0.7 points) and slightly better on harmless ones (−1.0 points), less than a point either way.

> **Q** &nbsp; Why is there no advantage?
>
> **A** &nbsp; Look at the map again. The circuit holds 35 of the 36 MLPs and almost 90 % of the weights a full adapter would train (13.5 million of 15.1 million). It is not a small circuit; it is most of the model. A random set of the same shape shares half of its components with it (92 of 184). So the comparison mostly asks *which heads* to train, and that turns out not to matter.

![The MLP row outlined: 35 of 36 are in the circuit, which holds 90 % of the weights a full adapter trains](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/circuit_size.png)

**Not just this adapter:** Before the adapters, we ran 393 experiments around a smaller circuit (the top 10 %). They used all three routes the problem statement names: steering, circuit-restricted LoRA, and knowledge editing with MEMIT (which rewrites a fact inside one layer), on correcting false facts and on refusal. Each intervention was run on the circuit and on control sets (random, lowest-scoring, all components); 180 of the 393 are steering runs at other strengths, with a shorter audit. Before running them we wrote down how the circuit should prove itself: fewer side effects than both random components and the lowest-scoring components. It never passed.

- A follow-up sweep over 17 layers found that layer 0, the circuit's top-scoring site, was the worst place to edit a fact: 37 % of edits succeed there, against 83–88 % at layers 4 to 10.
- A caveat: most of those runs used the set we fixed in advance, which ranks heads and MLPs separately (58 heads and 4 MLPs). It later failed the "enough to do the job" test: it keeps 0.4–3 % of fact recall. Follow-up runs on a set that passes the test also showed no advantage for the circuit.
- CircuitKIT's own paper reports the same kind of result for selective fine-tuning: a circuit mask against a random mask at the same budget shows no gain (its experiment E7).

![393 earlier experiments, coloured by method: steering, LoRA and weight editing](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit/resolve/main/article/earlier_runs.png)

> **Q** &nbsp; So are circuits useless?
>
> **A** &nbsp; No. For English fact recall, the top 10 % of components (heads and MLPs ranked together) keeps 33–56 % of the behaviour and a random 10 % keeps none, so that circuit carries a real share of the behaviour. What we did not find is a benefit from *restricting an intervention to the circuit*. On this model the circuit is how we chose where to train. It is not why the training worked. What worked is the recipe: refusal targets in five languages, a "stay normal" signal made from sampled text, the token at which generation actually stops, and a checkpoint rule that watches fluency.

## The Three Toolkits, and What We Added

| Step | Repo | What it does here |
|---|---|---|
| 1. Find the circuit | CircuitKIT 0.2.0 | Scores every head and MLP for refusal (EAP-IG) and runs the "enough to do the job" tests. In the 393 earlier experiments it also provides the steering and the MEMIT edits. |
| 2. Set the baseline to beat | SafeTune 0.1.10 | Refusal-direction steering, strength tuned on our practice set. |
| 3. Train | PEFT, with our masks | LoRA (rank 8) on the chosen components only. |
| 4. Score before and after | AuditKIT 1.1.3 | Scores the side-effect benchmarks (facts, MMLU, MGSM, IFEval). Its refusal detector, which we extended with phrases in four languages, counts refusals of harmless prompts. |
| 5. What the toolkits lack | our three repos | The control groups, the five-part audit, the judge that reads replies, confidence intervals. |

The hypotheses were fixed before the first run. Each of the 24 dated amendments says whether it was written before or after the results it touches. Both are in the [pre-registration](https://github.com/MdSadiqMd/Circuit-Steering/blob/main/docs/preregistration.md).

## If You Try This Yourself

- Train the same adapter on random components of the same size. Without that control, "circuit-restricted" is a label, not a result.
- Before you intervene on a circuit, test that it carries the behaviour on its own. Ours kept none of the refusal at 10 %.
- Check more than perplexity and one benchmark. A third of the changes that passed that check were damaged (by a 10-point line we set after the study).
- Outside English, read the replies. Do not judge replies to harmful requests by refusal phrases.
- End your training targets with the token at which generation actually stops.
- For "stay the same" training text, sample from the base model. An attempt that used its single most likely continuations raised perplexity 1.6 times in 15 steps.
- Do not merge a LoRA adapter into 16-bit weights: 31–43 % of the update's entries round to zero.
- Write the test down before you run it, and date every later change.

## Cost: Self-Hosting Against a Closed API

Sample volume: 1 million requests a month, 200 input and 150 output tokens each.

| Option | Price | Cost per month |
|---|---|---|
| Cohere Command A | $2.50 / $10.00 per million tokens | $2,000 |
| OpenAI GPT-5 mini | $0.25 / $2.00 | $350 |
| OpenAI GPT-5 nano | $0.05 / $0.40 | $70 |
| Cohere Command R7B | $0.0375 / $0.15 | $30 |
| Tiny Aya + adapter, one L4 GPU on HF Inference Endpoints, always on | $0.80 an hour × 730 hours | $584 |
| The same, GPU billed only while generating (a lower bound) | about 40 GPU-hours | $32 |

- **Throughput:** We measured 524 output tokens a second for the base model and 364 with the adapter, on an Apple M4 Max (batch 32). The 40 GPU-hours assume 1,500 tokens a second for the base model on an L4 with a batching server, reduced by the adapter's measured 30 % slowdown. The 1,500 is an estimate, not a measurement.
- **Break-even:** Always-on self-hosting beats Command A from about 290,000 requests a month and GPT-5 mini from about 1.7 million. It does not beat the cheapest small closed models until about 8–20 million.
- **The real case is not the bill:** A closed model cannot be opened up, audited at this depth or changed component by component. This one can.

API prices are list prices found in November 2026; Cohere's are third-party reported.

## Limits

- One model, three seeds. The safety gain differs by seed: the first adapter reaches 3.8 %, 7.4 % and 6.6 % outside English.
- "Circuit-restricted" here means 30 % of the components, including 33–35 of the 36 MLPs depending on the seed. It is not a small circuit.
- The "other circuits" part of the audit was run on the 213 earlier experiments that got the full audit, not on the two adapters.
- The judge is a local 27 B model (Qwen3.8-27B) that reads replies cut at 48 tokens. 60 of its labels were checked by the AI assistant that ran the analysis (55 agree, 2 disagree, 3 ambiguous), not yet by a person.
- The side-effect test sets are small: per language 60 facts, 40 reworded facts and 60 MMLU questions; 30 IFEval prompts and 30 WikiText passages. Small changes can hide.
- Refusals of harmless prompts are counted by phrase matching. The judge disagrees on 1 to 4 prompts in 295, depending on the model.
- 466 of the 636 non-English harmful test requests are Tiny Aya's own translations (all Hindi and French, the HarmBench half of Arabic and Swahili). 63 fail our back-translation check and are kept. The base model and the adapters see the same requests.
- The circuit and SafeTune's direction were found on English prompts, and some of those are English versions of test requests (125 of 159 for the circuit, 28 for SafeTune). This can only favour the circuit and the baseline.
- Some "harmful" MultiJail prompts are harmless questions. Helping with them counts as answered for the base model and the adapters alike.
- The second adapter's design was chosen on the practice set after the first adapter's test results were known.
- The audit sees only what it measures.

## Where Everything Is

- Adapters: [second, recommended](https://huggingface.co/MdSadiqMd/tiny-aya-global-refusal-circuit-lora-fluent) · [first](https://huggingface.co/MdSadiqMd/tiny-aya-global-refusal-circuit-lora) · [the 427 runs in the run archive](https://huggingface.co/MdSadiqMd/circuit-steering-cells)
- Results as charts: [Space](https://huggingface.co/spaces/MdSadiqMd/circuit-steering)
- Data, per-reply judgments, logs and result tables: [circuit-steering-data](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-data), [circuit-steering-circuits](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-circuits), [circuit-steering-audit](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit)
- Code: [Circuit-Steering](https://github.com/MdSadiqMd/Circuit-Steering) (the study), [tiny-aya-circuits](https://github.com/MdSadiqMd/tiny-aya-circuits) (circuit discovery), [bystander-circuits](https://github.com/MdSadiqMd/bystander-circuits) (the audit)
- Six defects we found in the track repos, [written up with measurements and fixes](https://github.com/MdSadiqMd/Circuit-Steering/blob/main/docs/upstream-issues.md); they are not filed yet
- Full report: [docs/report.md](https://github.com/MdSadiqMd/Circuit-Steering/blob/main/docs/report.md)

> **Q** &nbsp; So what is the overall lesson?
>
> **A** &nbsp; Changing the behaviour was the easy part: twelve minutes of training. The work was the second sentence of the problem statement, verifying that nothing else broke, and one question nobody asked us for: whether the circuit was the reason. An audit with controls answered both, and that is the part we would reuse.
