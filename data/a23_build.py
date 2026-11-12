"""Amendment A23 data: plain-text retain pairs and held-out fluency text in five languages.

  uv run python data/a23_build.py            # downloads two public datasets, needs the model
  uv run python data/a23_build.py --check    # selection logic only

Sources (public, un-gated, fetched into .cache/raw):
  facebook/xnli        premises of the validation and test splits in en/hi/ar/fr/sw: human
                       translations of the same sentences, so every language sees parallel text
  Salesforce/wikitext  wikitext-2-raw-v1 train (the audit reads the first 7,680 tokens of test)
Selection is by sha256 of the item index, the same indices in every language:
  retain   100 XNLI validation premises x 5 languages + 100 WikiText train paragraphs
  dev      40 other XNLI validation premises x 5 languages (checkpoint rule, fluency term)
  test     200 XNLI test premises x 5 languages (audit/a23_fluency.py)
A retain pair is (the first half of the text, a continuation). `variant=greedy` (A23 as sealed)
takes the base model's own greedy continuation; that sharpens the adapter's plain-text
predictions and raised dev perplexity 1.6x in 15 steps (docs/notes.md 2026-11-12). `sampled` draws
the continuation from the base model at temperature 1 (seed 0), so the loss estimates the
cross-entropy to the base model's distribution; `real` uses the text's own continuation. `out`
names the file. Text carries no chat template and no BOS, as in the audit.

Writes data/a23/plain.jsonl (retain pairs) and data/a23/fluency.jsonl (dev and test text).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / ".cache/raw", ROOT / "data/a23"
LANGS = ("en", "hi", "ar", "fr", "sw")
N_RETAIN, N_DEV, N_TEST, N_WIKI = 100, 40, 200, 100
XNLI_NEW, WIKI_PREFIX, WIKI_NEW = 24, 32, 32


def order(n: int, tag: str) -> list[int]:
    return sorted(range(n), key=lambda i: hashlib.sha256(f"{tag}:{i}".encode()).hexdigest())


def picks(n_val: int, n_test: int) -> dict[str, list[int]]:
    val = order(n_val, "xnli-validation")
    return {
        "retain": val[:N_RETAIN],
        "dev": val[N_RETAIN : N_RETAIN + N_DEV],
        "test": order(n_test, "xnli-test")[:N_TEST],
    }


def fetch() -> None:
    from huggingface_hub import hf_hub_download

    for lang in LANGS:
        for split in ("validation", "test"):
            hf_hub_download(
                "facebook/xnli",
                f"{lang}/{split}-00000-of-00001.parquet",
                repo_type="dataset",
                local_dir=RAW / "xnli",
            )
    hf_hub_download(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=RAW / "wikitext",
    )


def premises(lang: str, split: str) -> list[str]:
    """One premise per source sentence, in file order (each premise carries three hypotheses)."""
    import pandas as pd

    df = pd.read_parquet(RAW / f"xnli/{lang}/{split}-00000-of-00001.parquet")
    return df.premise.tolist()[::3]


def main() -> None:
    import pandas as pd
    import torch
    from aya_circuits.config import load_config
    from aya_circuits.model import load_hf

    cfg = load_config()
    # greedy (A23 as sealed) | sampled (temperature 1, seed 0) | real (the text's own continuation)
    variant = str(cfg.get("variant", "greedy"))
    plain_path = ROOT / str(cfg.get("out", "data/a23/plain.jsonl"))
    fetch()
    val = {lang: premises(lang, "validation") for lang in LANGS}
    test = {lang: premises(lang, "test") for lang in LANGS}
    assert len({len(v) for v in val.values()}) == 1 and len({len(v) for v in test.values()}) == 1
    pk = picks(len(val["en"]), len(test["en"]))
    OUT.mkdir(exist_ok=True)
    if not (OUT / "fluency.jsonl").exists():
        with (OUT / "fluency.jsonl").open("w") as f:
            for split, src in (("dev", val), ("test", test)):
                for i in pk[split]:
                    for lang in LANGS:
                        row = {"id": f"xnli:{split}:{i}:{lang}", "lang": lang, "split": split}
                        f.write(json.dumps(row | {"text": src[lang][i]}, ensure_ascii=False) + "\n")

    wiki = pd.read_parquet(RAW / "wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet")
    paras = [t.strip() for t in wiki.text.tolist() if len(t.strip()) > 200]
    todo = [
        (f"xnli:retain:{i}:{lang}", lang, "xnli", val[lang][i], None, XNLI_NEW)
        for i in pk["retain"]
        for lang in LANGS
    ] + [
        (f"wikitext:train:{i}", "en", "wikitext", paras[i], WIKI_PREFIX, WIKI_NEW)
        for i in order(len(paras), "wikitext-train")[:N_WIKI]
    ]
    model, tok = load_hf(cfg)
    stop = model.generation_config.eos_token_id
    tok.padding_side = "left"
    rows = []
    for rid, lang, source, text, k, n_new in todo:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        k = k or len(ids) // 2
        if k < 4 or len(ids) < k + 4:
            continue
        row = {"id": rid, "lang": lang, "source": source, "ids": ids[:k], "n_new": n_new}
        rows.append(row | {"rest": ids[k : k + n_new]})
    rows.sort(key=lambda r: (r["n_new"], len(r["ids"])))
    torch.manual_seed(0)
    sampling = {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0}
    for r in rows if variant == "real" else []:
        r["prefix"], r["continuation"] = tok.decode(r["ids"]), tok.decode(r["rest"])
    for i in range(0, len(rows) if variant != "real" else 0, 16):
        batch = rows[i : i + 16]
        n_new = max(r["n_new"] for r in batch)
        width = max(len(r["ids"]) for r in batch)
        inp = torch.full((len(batch), width), tok.pad_token_id)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for j, r in enumerate(batch):
            inp[j, width - len(r["ids"]) :] = torch.tensor(r["ids"])
            att[j, width - len(r["ids"]) :] = 1
        with torch.no_grad():
            g = model.generate(
                input_ids=inp.to(model.device),
                attention_mask=att.to(model.device),
                max_new_tokens=n_new,
                pad_token_id=tok.pad_token_id,
                **(sampling if variant == "sampled" else {"do_sample": False}),
            )
        for r, new in zip(batch, g[:, width:], strict=True):
            new = new[: r["n_new"]].tolist()
            cut = new.index(stop) if stop in new else len(new)
            r["prefix"] = tok.decode(r["ids"])
            r["continuation"] = tok.decode(new[:cut], skip_special_tokens=True)
    rows = [r for r in rows if r["continuation"].strip()]
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    with plain_path.open("w") as f:
        for r in sorted(rows, key=lambda r: r["id"]):
            keep = {k: r[k] for k in ("id", "lang", "source", "prefix", "continuation")}
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")
    print({s: sum(r["source"] == s for r in rows) for s in ("xnli", "wikitext")}, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        p = picks(830, 1670)
        assert [len(p[k]) for k in ("retain", "dev", "test")] == [100, 40, 200]
        assert not set(p["retain"]) & set(p["dev"]) and p == picks(830, 1670)
        assert max(p["retain"] + p["dev"]) < 830 and max(p["test"]) < 1670
        print("a23_build ok")
    else:
        main()
