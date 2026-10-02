"""Self-contained In-context Retrieval (ICR) tasks for the CRIR benchmark.

These are the zero-shot cloze-completion tasks used in "Preconditioned DeltaNet:
Curvature-aware Sequence Modeling for Linear Recurrences" (arXiv:2604.21100,
Table 4), following the Based / Just Read Twice evaluation suite (Arora et al.,
2024). The upstream lm-evaluation-harness only ships ``fda`` / ``swde`` /
``squad_completion`` in very recent releases and does not ship the cloze
TriviaQA / DROP tasks at all, so all five are vendored here to keep the run
reproducible regardless of the installed harness version.

Every task is ``generate_until`` with ``until=["\n"]`` and ``max_gen_toks=48``;
the reported score is "contains" (case-insensitive substring match of the gold
answer inside the generated continuation), averaged into an accuracy.
"""

from __future__ import annotations

import re

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


def contains_score(prediction: str, labels) -> float:
    labels = list(labels)
    return max(
        int(bool(re.search(re.compile(re.escape(str(label)), re.IGNORECASE), prediction)))
        for label in labels
    )


class _CompletionTask(ConfigurableTask):
    """FDA / SWDE / SQuAD: the full prompt is already stored in ``doc["text"]``."""

    DATASET_PATH = None
    DATASET_NAME = "default"

    def __init__(self):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return doc["text"].strip()

    def doc_to_target(self, doc):
        return doc["value"].strip()

    def construct_requests(self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs):
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"], "max_gen_toks": 48}),
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        return {"contains": contains_score(results[0], [self.doc_to_target(doc)])}

    def aggregation(self):
        return {"contains": np.mean}

    def higher_is_better(self):
        return {"contains": True}


class FDA(_CompletionTask):
    VERSION = 1
    DATASET_PATH = "hazyresearch/based-fda"


class SWDE(_CompletionTask):
    VERSION = 1
    DATASET_PATH = "hazyresearch/based-swde-v2"


class SQUADCompletion(_CompletionTask):
    VERSION = 1
    DATASET_PATH = "hazyresearch/based-squad"


class _QATask(ConfigurableTask):
    """TriviaQA / DROP: prompt is built from ``context`` + ``question``."""

    DATASET_PATH = None
    DATASET_NAME = "default"

    def __init__(self):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_target(self, doc):
        answer_list = doc.get("answers") or []
        return " " + (answer_list[0] if answer_list else "unanswerable")

    def _doc_to_text(self, doc):
        context = self._clean_context(doc["context"].strip())
        question = doc["question"].strip()
        while question and context.lower().endswith(question.lower()):
            context = context[: -len(question)]
        return context.strip().strip(".") + ". " + question

    def construct_requests(self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs):
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"], "max_gen_toks": 48}),
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        return {"contains": contains_score(results[0], doc["answers"])}

    def aggregation(self):
        return {"contains": np.mean}

    def higher_is_better(self):
        return {"contains": True}


class BasedTriviaQA(_QATask):
    VERSION = 1
    DATASET_PATH = "hazyresearch/based_triviaqa"

    @staticmethod
    def _clean_context(context):
        return re.sub(r"\[PAR\]|\[DOC\]|\[TLE\]", "", context)

    def doc_to_text(self, doc):
        return self._doc_to_text(doc)


class BasedDrop(_QATask):
    VERSION = 1
    DATASET_PATH = "hazyresearch/based_drop"

    @staticmethod
    def _clean_context(context):
        return context

    def doc_to_text(self, doc):
        return self._doc_to_text(doc)
