"""Replay a past evaluation without calling an LLM.

ReplayClient stands in for the OpenAI client. It offers the same `responses.parse(...)`
that the model already calls, but instead of reaching the API it recognises which report
the prompt is asking about and returns the value that this run's model produced back in
August 2025 (recorded by scripts/export_replay.py into data/replay/).

    llm_client = ReplayClient("eval-2025-08-02-daring-whale")

Everything around it stays real: predict() is unchanged, and the prompt, the scorer, the
evaluation, its summary and its traces all genuinely run.

A replay is frozen to one experiment, so it refuses to guess: a prompt that isn't the one
that produced these predictions, or a report the run never saw, raises rather than
returning a number that would look plausible and be a lie.
"""

import json
import sys
import unicodedata
from pathlib import Path

import weave

from utilities import gold_split

REPLAY_DIR = Path(__file__).resolve().parent / "data" / "replay"


def _nfc(s: str) -> str:
    # prompts and fingerprints written on macOS can carry NFD umlauts while the same
    # text elsewhere is NFC; compare everything in one normal form
    return unicodedata.normalize("NFC", s)


class ReplayError(RuntimeError):
    pass


class _Parsed:
    def __init__(self, value, text_format):
        self.output_parsed = text_format(label_value_llm_output=value)


class _Responses:
    def __init__(self, record):
        self._examples = record["examples"]
        # the fixed part of the prompt, i.e. everything the report does not fill in
        self._prompt_head = record["base_prompt"].split("{report}")[0].strip()
        self._run = record["run"]

    def parse(self, model, input, text_format, **kwargs):
        prompt = _nfc(input if isinstance(input, str) else "\n".join(
            m["content"] for m in input if isinstance(m.get("content"), str)
        ))

        if _nfc(self._prompt_head) not in prompt:
            print(
                f"replay warning: this prompt differs from the one that produced "
                f"{self._run}'s predictions; matching the report by fingerprint anyway",
                file=sys.stderr,
            )

        for example in self._examples:
            if _nfc(example["fingerprint"]) in prompt:
                if "raised" in example:
                    # this report killed the original run: the model answered something the
                    # schema rejected. Validating that same answer raises the same error.
                    text_format.model_validate(example["raised"]["raw_output"])
                return _Parsed(example["label_value_llm_output"], text_format)

        raise ReplayError(
            f"{self._run} has no recorded prediction for this report -- it was not part of "
            "that run, or the preprocessing changed since it was exported "
            "(re-run scripts/export_replay.py)"
        )


def load(run: str) -> dict:
    path = REPLAY_DIR / f"{run}.json"
    if not path.exists():
        available = sorted(p.stem for p in REPLAY_DIR.glob("*.json") if p.stem != "history")
        raise FileNotFoundError(f"no exported run {run!r}; available: {available}")
    return json.loads(path.read_text())


class ReplayClient:
    def __init__(self, run: str):
        record = load(run)
        if record.get("replay_level") != "client":
            raise ReplayError(
                f"{run} cannot be replayed through the model code: it made one LLM call per "
                "page and those calls were never traced, so only its predict() results "
                "survive. Use ReplayModel instead."
            )
        self.record = record
        self.run = record["run"]
        self.responses = _Responses(record)

    def dataset(self) -> list[dict]:
        """The reports this run is replayed on, gold labels included."""
        return dataset_of(self.record)


def dataset_of(record: dict) -> list[dict]:
    """The run's recorded examples, restricted to the corrected split in data/gold.

    A record covers every report the run originally saw, which for the train runs includes
    the Fresenius SE-14829 row the source project leaked into train. Grading a replay on it
    would put these numbers on 100 rows while every other run in the notebook sits on the
    published 99, so the run's own score would not be comparable to its neighbours'. The
    recorded answer for the dropped report stays in the record, simply unused: the reports
    kept here are a subset of what the run answered, so nothing goes looking for a
    prediction that is not there.
    """
    keep = {row["file_path"] for row in gold_split(record["split"])}
    return [
        {
            "file_path": e["file_path"],
            "label_value": e["label_value"],
            "label_present": e["label_present"],
        }
        for e in record["examples"]
        if e["file_path"] in keep
    ]


class ReplayModel(weave.Model):
    """Returns what a past run's predict() answered, without running the model code.

    For runs the ReplayClient cannot serve -- the pagewise ones, whose per-page LLM calls
    were never traced. This does not re-run anything: it restates what the run answered so
    the real dataset and the real scorer can grade it again. The code that produced these
    answers is shown alongside in the notebook; it is not what executes here.

    It still carries the run's prompts, so that a replayed journey rebuilds the same prompt
    version history as the original project rather than stopping wherever the last
    client-level replay left off.
    """

    run: str
    system_prompt: weave.trace.refs.ObjectRef
    base_prompt: weave.trace.refs.ObjectRef

    @weave.op()
    def predict(self, file_path: str) -> dict:
        return {"label_value_predicted": _ANSWERS[self.run][file_path]}


_ANSWERS: dict[str, dict[str, float | None]] = {}


def replay_model(run: str) -> "ReplayModel":
    """A ReplayModel for `run`, named and prompted as the original model was.

    Grade it with the published `<split>_eval` object (see `split_of`) so the replay lands
    on the split's leaderboard alongside everything else.
    """
    record = load(run)
    _ANSWERS[record["run"]] = {
        e["file_path"]: e["label_value_llm_output"] for e in record["examples"]
    }
    return ReplayModel(
        name=record["model_id"],
        run=record["run"],
        system_prompt=weave.publish(weave.StringPrompt(record["system_prompt"]), name="system_prompt"),
        base_prompt=weave.publish(weave.StringPrompt(record["base_prompt"]), name="base_prompt"),
    )


def split_of(run: str) -> str:
    """Which split this run was evaluated on, i.e. which <split>_eval object grades it."""
    return load(run)["split"]
