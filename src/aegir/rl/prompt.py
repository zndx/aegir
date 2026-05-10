"""Prompt construction for the GRPO rollout.

The policy is asked to emit a JSON list of CompositionEntry dicts
that, when scored by the locked verifier R(O, I), yield high reward.
The prompt has three parts:

1. **System** — explains the slot DSL and the JSON output shape.
2. **Few-shot context** — 2-3 catalog rows shown as concrete
   ``(template_id, slot_fillers)`` examples drawn from the
   foundation batch.
3. **User instruction** — describes the target SDG context
   (e.g., "build a 6-template ontology that observes a kernel
   syscall, attests compliance to a directive, and includes a
   belief interval").

The prompt is intentionally compact (~600 tokens) so it leaves
room for ``DecodingConfig.max_new_tokens`` outputs in the
context window of the 27B base model.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegir.ontology.schema import Catalog


@dataclass
class PromptConfig:
    n_few_shot_examples: int = 3
    target_n_compositions: int = 6
    user_template: str = (
        "Construct a {target_n} template composition that demonstrates "
        "cross-context cousining across at least two of: "
        "(observation, governance, lineage, belief). "
        "Use catalog template_ids only; pick slot fillers in the sdg: "
        "namespace where appropriate."
    )


SYSTEM_PROMPT = """You are an ontology-composition policy. Your task is to emit a
JSON list of CompositionEntry objects. Each object has:

  {
    "template_id": "<one of the catalog template ids>",
    "slot_fillers": {"<slot_name>": "<filler class IRI>", ...}
  }

The CompositionEntry must select a template_id from the SDG ontology
catalog. Slot fillers must be class IRIs (typically in the sdg:, cco:,
or bfo: namespace). The JSON list as a whole forms an ontology
composition that will be scored by a verifier that rewards:

- structural correctness (R_A, hard gate),
- complex-class density (R_B),
- semantic richness (R_C),
- topic alignment with the target corpus (R_D).

Do not invent template_ids that are not in the catalog. Do not invent
slot names. Emit a single JSON list and nothing else.
"""


def render_few_shot(catalog: Catalog, n: int = 3) -> list[dict]:
    """Pick a small set of complex catalog rows as few-shot examples.

    The few-shot set is deterministic given catalog ordering: we take
    the first ``n`` rows whose ``is_complex=True`` and whose
    ``slot_types`` contain at least two slots, biasing toward F2 or
    F5 patterns rather than F1 basic forms.
    """
    out: list[dict] = []
    for tmpl in catalog.templates:
        if not tmpl.is_complex:
            continue
        if len(tmpl.slot_types) < 2:
            continue
        # toy fillers for the demo
        fillers = {
            slot_name: f"sdg:Demo{i}{slot_name}"
            for i, slot_name in enumerate(tmpl.slot_types.keys())
        }
        out.append({"template_id": tmpl.template_id, "slot_fillers": fillers})
        if len(out) >= n:
            break
    return out


def build_messages(
    catalog: Catalog,
    cfg: PromptConfig,
) -> list[dict]:
    """Build a chat-format message list ready to feed the tokenizer's
    ``apply_chat_template`` method."""
    examples = render_few_shot(catalog, n=cfg.n_few_shot_examples)
    examples_block = "Few-shot examples:\n"
    import json as _json
    for ex in examples:
        examples_block += _json.dumps(ex, indent=2) + "\n"

    user_msg = cfg.user_template.format(target_n=cfg.target_n_compositions)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip() + "\n\n" + examples_block.strip()},
        {"role": "user", "content": user_msg},
    ]
