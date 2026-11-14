"""Publish artifacts to the Hub (brief §5): datasets (data/), circuits + faithfulness, intervention
cells, audit results and scorecards. Repos (all public, under the owner's namespace):
  circuit-steering-data        dataset  data/ (prompts, EAP CSVs, false facts, B2 pools, baselines)
  circuit-steering-circuits    dataset  runs/circuits (scores, signed edges, faithfulness,
  manifests)
  circuit-steering-cells       model    runs/interventions: cell.json + manifest for every cell;
  the weight
                                        file only for cells whose audit shows no refusal decrease
                                        on any
                                        harmful family in any language (brief: never publish
                                        refusal-weakening artifacts); unaudited cells are withheld
                                        too
  circuit-steering-audit       dataset  runs/audit + docs/results
Each dataset repo gets its card from docs/dataset_cards/ as README.md. The card's `configs` block
names the tabular files the Hub viewer shows; without it the viewer tries to merge every JSON
file into one table and fails. The circuits repo has no tabular file of its own, so
`circuit_table` writes component_scores.csv from every run's nodes.json.
A22 and A24 (`what=a22,a24`, roots runs/a22 and runs/a24; not in the default list):
  tiny-aya-global-refusal-circuit-lora          model  A22's PEFT adapter (runs/a22/release)
  tiny-aya-global-refusal-circuit-lora-fluent   model  A24's PEFT adapter (runs/a24/release)
  circuit-steering-cells       model    a22/, a24/: the cells, weight files under the same rule
  circuit-steering-audit       dataset  a22/, a24/: test judgments, benign replies, judge labels,
                                        fluency records, audit
Generated replies to harmful prompts never leave the machine (brief §8): data/ must scan clean
(data/sanitize.py) and the audit is pushed from a sanitized staging copy (.cache/publish/audit).
The Hub keeps old revisions; `squash=true` rewrites each pushed repo's history to one commit, which
is needed once to drop the replies the 2026-11-03 pushes published.
Run: HF_TOKEN=... uv run python data/push_hf.py [what=data,circuits,cells,audit] [squash=true]"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config
from aya_circuits.hub import NAMESPACE, delete_paths, push_folder

from data.sanitize import scan, stage_audit

ROOT = Path(__file__).resolve().parent.parent
RELEASE, RELEASE_CELL = "tiny-aya-global-refusal-circuit-lora", "circuit30_s0"
RELEASES = {"a22": RELEASE, "a24": f"{RELEASE}-fluent"}  # Hub model repo of each run's adapter
CARD_FIGURES = {  # docs/figures/cards/<name>.png shown on docs/model_cards/<card>.md
    "a22": ("a22_safety", "adapters_defects", "adapters_side_effects"),
    "a24": ("a24_safety", "adapters_defects", "adapters_side_effects"),
    "cells": ("cells_contents", "cells_locality"),
}


def circuit_table(circuits: Path) -> Path:
    """component_scores.csv in the circuits folder: one row per task, seed and component (head or
    MLP) from every run's nodes.json, ranked by score within the run."""
    import csv
    import json

    out = circuits / "component_scores.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "seed", "component", "kind", "layer", "score", "signed_rowsum", "rank"])
        for nj in sorted(circuits.glob("*/seed*/nodes.json")):
            n = json.loads(nj.read_text())
            task, seed = nj.parts[-3], int(nj.parts[-2].removeprefix("seed"))
            score = n["abs_scores"]
            signed = dict(zip(n["names"], n["signed_rowsum"], strict=True))
            for rank, c in enumerate(sorted(score, key=lambda c: -score[c]), 1):
                head = c.startswith("A")  # "A20.4" is layer 20 head 4, "MLP 20" the layer's MLP
                layer = int(c[1:].split(".")[0]) if head else int(c.split()[1])
                w.writerow(
                    [task, seed, c, "head" if head else "mlp", layer, score[c], signed[c], rank]
                )
    return out


def push_card(name: str) -> None:
    """docs/dataset_cards/<name>.md as the README.md of the dataset repo circuit-steering-<name>."""
    from huggingface_hub import HfApi

    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj=str(ROOT / f"docs/dataset_cards/{name}.md"),
        path_in_repo="README.md",
        repo_id=f"{NAMESPACE}/circuit-steering-{name}",
        repo_type="dataset",
        commit_message="dataset card",
    )


def push_model_card(card: str, repo: str) -> None:
    """docs/model_cards/<card>.md as the model repo's README.md, with the figures it shows under
    figures/ (written by docs/figures/make_card_figures.py), in one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    figures = ROOT / "docs/figures/cards"
    ops = [CommitOperationAdd("README.md", str(ROOT / f"docs/model_cards/{card}.md"))]
    names = CARD_FIGURES[card]
    ops.extend(CommitOperationAdd(f"figures/{f}.png", str(figures / f"{f}.png")) for f in names)
    HfApi(token=os.environ["HF_TOKEN"]).create_commit(
        repo_id=f"{NAMESPACE}/{repo}",
        repo_type="model",
        operations=ops,
        commit_message="model card and figures",
    )


def withheld_cells(runs: Path) -> list[str]:
    """Relative paths (under runs/interventions) of weight files that must not be published: cells
    whose
    audited refusal rate on any harmful family (held-out category, other categories) in any
    language is
    below the base model's, and cells that have no audit yet."""
    import json

    out = []
    for cj in sorted(runs.glob("interventions/*/*/*/cell.json")):
        beh, interv, cell_dir = cj.parts[-4], cj.parts[-3], cj.parts[-2]
        name = f"{beh}_{interv}_{cell_dir}"
        hf = runs / "audit" / beh / name / "hf.json"
        base = runs / "audit" / beh / "base" / "hf.json"
        rel = f"{beh}/{interv}/{cell_dir}/delta.safetensors"
        if not hf.exists() or not base.exists():
            out.append(rel)
            continue
        b, c = json.loads(base.read_text()), json.loads(hf.read_text())
        fams = [k for k in c if k.startswith(("refusal_harmful_", "refusal_other_"))]
        if any(
            c[k]["headline"]["refusal"] < b.get(k, {}).get("headline", {}).get("refusal", 0)
            for k in fams
        ):
            out.append(rel)
    return out


def run_withheld(run_root: Path) -> list[str]:
    """Weight files of an A22 / A24 run that must not be published (relative to the staged
    `<run>/` folder): cells whose refusal rate on a harmful test family in any language is below
    the base model's, and cells that were not evaluated."""
    import json

    def rates(p: Path) -> dict:
        d = json.loads(p.read_text())
        return {
            (f, lang): sum(v["scores"].values()) / len(v["scores"])
            for f in ("harmful_test", "xstest_unsafe")
            for lang, v in d[f].items()
        }

    base, out = rates(run_root / "eval/base.json"), []
    for cj in sorted((run_root / "interventions/b2").glob("*/*/cell.json")):
        kind, name = cj.parent.parent.name, cj.parent.name
        ev = run_root / "eval" / f"b2_{kind}_{name}.json"
        if not ev.exists() or any(v < base[k] for k, v in rates(ev).items()):
            out.append(f"{run_root.name}/b2/{kind}/{name}/delta.safetensors")
    return out


def stage_run(run_root: Path, dst: Path) -> tuple[Path, Path]:
    """(audit staging, cells staging) for an A22 / A24 run, each holding one `<run>/` folder: test
    judgments with benign replies, judge labels, fluency records, the chain logs and the sanitized
    audit; and the cells (hard links). Replies to harmful prompts are in neither."""
    import os
    import shutil

    run = run_root.name
    audit, cells = dst / "audit", dst / "cells"
    for d in (audit, cells):
        shutil.rmtree(d, ignore_errors=True)
    for sub in ("eval", "judge", "judge_benign", "fluency"):
        if (run_root / sub).exists():
            shutil.copytree(run_root / sub, audit / run / sub)
    for name in ("ended.json", "dev_base.json"):
        if (run_root / name).exists():
            shutil.copy2(run_root / name, audit / run / name)
    stage_audit(run_root / "audit", audit / run / "audit")
    (audit / run / "logs").mkdir(parents=True, exist_ok=True)
    for log in sorted((ROOT / ".cache/logs").glob(f"{run}*.log")):  # chain logs, home path removed
        text = log.read_text().replace(str(Path.home()), "~")
        (audit / run / "logs" / log.name).write_text(text)
    import json

    for f in (audit / run / "eval").glob("*.json"):  # judgments only for the harmful families
        d = json.loads(f.read_text())
        assert not any(
            "outputs" in v for k in ("harmful_test", "xstest_unsafe") for v in d[k].values()
        ), f
    for f in (run_root / "interventions").rglob("*"):
        if f.is_file():
            t = cells / run / f.relative_to(run_root / "interventions")
            t.parent.mkdir(parents=True, exist_ok=True)
            os.link(f, t)
    return audit, cells


def push_run(run_root: Path) -> list[tuple[str, str]]:
    import shutil

    run = run_root.name
    audit, cells = stage_run(run_root, ROOT / f".cache/publish/{run}")
    withheld = run_withheld(run_root)
    print(run, "withheld", len(withheld), "weight files")
    print(
        push_folder(
            cells,
            "circuit-steering-cells",
            "model",
            msg=f"{run} cells (factors + manifests)",
            ignore_patterns=withheld,
        )
    )
    print(push_folder(audit, "circuit-steering-audit", "dataset", msg=f"{run} results"))
    pushed = [("circuit-steering-cells", "model"), ("circuit-steering-audit", "dataset")]
    release, repo = run_root / "release" / RELEASE, RELEASES[run]
    if release.exists() and f"{run}/b2/lora/{RELEASE_CELL}/delta.safetensors" not in withheld:
        shutil.copy2(ROOT / f"docs/model_cards/{run}.md", release / "README.md")
        print(push_folder(release, repo, "model", msg="refusal adapter for Tiny Aya Global"))
        push_model_card(run, repo)
        pushed.append((repo, "model"))
    return pushed


def main() -> None:
    cfg = load_config()
    what = set(str(cfg.get("what", "data,circuits,cells,audit")).split(","))
    runs = Path(cfg["out_dir"])
    pushed: list[tuple[str, str]] = []
    if "data" in what:
        leaks = scan([ROOT / "data"])
        if leaks:
            raise SystemExit(f"{len(leaks)} harmful-prompt replies in data/; run data/sanitize.py")
        print(
            push_folder(
                ROOT / "data",
                "circuit-steering-data",
                "dataset",
                msg="data: prompts, EAP pairs, pools, baselines",
                ignore_patterns=["__pycache__/*"],
            )
        )
        push_card("data")
        pushed.append(("circuit-steering-data", "dataset"))
    if "circuits" in what and (runs / "circuits").exists():
        circuit_table(runs / "circuits")
        print(
            push_folder(
                runs / "circuits",
                "circuit-steering-circuits",
                "dataset",
                msg="circuits: scores, signed edges, faithfulness",
            )
        )
        push_card("circuits")
        pushed.append(("circuit-steering-circuits", "dataset"))
    if "cells" in what and (runs / "interventions").exists():
        withheld = withheld_cells(runs)
        (runs / "interventions" / "WITHHELD.md").write_text(
            "# Weight files withheld from publication\n\nCells whose audited refusal rate on a "
            "harmful family in any language is below the base model's, or that have no audit yet; "
            "cell.json and manifest.json are published for every cell, the weight file only for "
            "the rest.\n\n" + "\n".join(f"- {p}" for p in withheld) + "\n"
        )
        print("withheld", len(withheld), "weight files")
        print(
            push_folder(
                runs / "interventions",
                "circuit-steering-cells",
                "model",
                msg="intervention cells (factors + manifests)",
                ignore_patterns=withheld,
            )
        )
        print(
            "deleted from the Hub:",
            delete_paths(
                "circuit-steering-cells",
                "model",
                withheld,
                "remove weight files of refusal-weakening or unaudited cells",
            ),
        )
        push_model_card("cells", "circuit-steering-cells")
        pushed.append(("circuit-steering-cells", "model"))
    if "audit" in what and (runs / "audit").exists():
        staged = stage_audit(runs / "audit", ROOT / ".cache/publish/audit")
        print(push_folder(staged, "circuit-steering-audit", "dataset", msg="audit results"))
        pushed.append(("circuit-steering-audit", "dataset"))
        if (ROOT / "docs/results").exists():
            print(
                push_folder(
                    ROOT / "docs/results",
                    "circuit-steering-audit",
                    "dataset",
                    msg="scorecards and hypothesis tables",
                )
            )
        push_card("audit")
    for run in ("a22", "a24"):
        if run in what:
            pushed += push_run(ROOT / "runs" / run)
    if cfg.get("squash"):
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        for repo, kind in pushed:
            api.super_squash_history(f"{NAMESPACE}/{repo}", repo_type=kind)
            print("squashed history of", repo)


if __name__ == "__main__":
    main()
