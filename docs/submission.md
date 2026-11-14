# Submission sheet

Form: https://hackathon.lexsi.ai/submission/ (a Tally form). Deadline: Nov 15, 23:59 AoE, which is
Nov 16, 17:29 IST. You can submit more than once; the latest one counts. The form has a captcha, so
a person has to press submit.

## Page 1: Your team

| field | required | what to enter |
|---|---|---|
| Registered email | yes | the exact email you registered with |
| Abstract | yes, under 500 words and at most 3,500 characters | paste the body of `docs/abstract.md` (everything below the `# Abstract` heading; 487 words, 2,943 characters, plain text) |
| Discord username | no | your Discord handle |

## Page 2: Your artifacts

Every repo in both collections must be public by the deadline.

| field | required | what to enter |
|---|---|---|
| Hugging Face model collection | yes | https://huggingface.co/collections/MdSadiqMd/circuit-steering-on-tiny-aya-models-and-demo-6af87a1b0a621b28feed3bf7 |
| Hugging Face dataset collection | yes | https://huggingface.co/collections/MdSadiqMd/circuit-steering-on-tiny-aya-datasets-and-logs-6af87a1cee0c921b998574ef |
| Hugging Face article | yes | `https://github.com/MdSadiqMd/Circuit-Steering/blob/main/docs/article.md` (see "The article"). The organisers accept the write-up as a markdown file in a GitHub repo when the account has no HF PRO plan |
| GitHub repo | yes | https://github.com/MdSadiqMd/Circuit-Steering |

What goes in the model collection:

- `MdSadiqMd/tiny-aya-global-refusal-circuit-lora-fluent` (second adapter, recommended)
- `MdSadiqMd/tiny-aya-global-refusal-circuit-lora` (first adapter, strongest effect)
- `MdSadiqMd/circuit-steering-cells` (every intervention cell: factors and manifests)
- the Space, `MdSadiqMd/circuit-steering` (https://huggingface.co/spaces/MdSadiqMd/circuit-steering)

What goes in the dataset collection:

- `MdSadiqMd/circuit-steering-data` (prompts, splits, fact sets)
- `MdSadiqMd/circuit-steering-circuits` (circuit scores, edges, faithfulness)
- `MdSadiqMd/circuit-steering-audit` (audit results, per-item judgments, logs, result tables)

## The article

Publishing an article on Hugging Face needs a PRO, Team or Enterprise plan (the "New Article" button says
so on a free account). The organisers' clarification: without PRO, submit the write-up as a markdown file
in a GitHub repo; they will feature it in the org after the hackathon with you as the author.

1. Commit and push `docs/` (at least `docs/article.md`, and `docs/preregistration.md`, `docs/report.md`,
   `docs/upstream-issues.md`, which the article links to).
2. Open `https://github.com/MdSadiqMd/Circuit-Steering/blob/main/docs/article.md` in a private window.
   The pictures and clips must show (they are hosted in the `circuit-steering-audit` dataset under
   `article/`), and the poster at the top must open the film.
3. Paste that link into the form's article field. The field is a plain link input; its
   `https://huggingface.co/blog/…` text is only a placeholder.

## Page 3: Repos, models, tracks

| question | tick |
|---|---|
| Which repos did you use? | SafeTune, CircuitKIT, AuditKIT |
| Which models did you use? | Tiny Aya Global, Other |
| Which other models? | `Qwen3.8-27B (local LLM judge only, never trained or released)` |
| Which track? | Track 2b: AI Safety and Correction |
| Problem statements for Track 2b (at most 2) | Circuit-level interventions; Locating where safety lives |

## Page 4: Consent and checklist

| checkbox | required | tick? |
|---|---|---|
| Consent to the article being shared by Hugging Face and Cohere | no | your choice |
| Consent to being tagged for winning or featured submissions | no | your choice |
| All repos and collections are public and final as of Nov 15, 23:59 AoE | yes | tick only after the checks below pass |

## Before you tick the last box

1. Open both collection links in a private browser window. Every item must load without a login.
   On each of the three dataset pages the viewer must show a table, not an error.
2. Open both adapter pages in the private window. The model card must show, not a 401 page.
3. Open the article link in the private window. The figure must show and every link must open.
4. Open the Space. The four numbers at the top and every chart must show.
5. Open the GitHub repo. The README must show the headline table and the run instructions, and
   `docs/article.md`, `docs/preregistration.md`, `docs/report.md` and `docs/upstream-issues.md`
   must open.
6. Open https://github.com/MdSadiqMd/tiny-aya-circuits and
   https://github.com/MdSadiqMd/bystander-circuits in the private window. Both must load.
7. Do not push anything to the Hub or the collections after the deadline.
