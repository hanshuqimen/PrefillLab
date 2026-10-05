"""Module heuristics in one place, with user-supplied rules taking precedence."""

import re

from prefilllab.core.models import ModuleCategory


class ModuleClassifier:
    def __init__(self, rules: list[tuple[str, ModuleCategory]] | None = None) -> None:
        self.rules = [(re.compile(pattern, re.I), category) for pattern, category in (rules or [])]

    def classify(self, name: str, class_name: str = "") -> ModuleCategory:
        for pattern, category in self.rules:
            if pattern.search(name):
                return category
        leaf = name.rsplit(".", 1)[-1].lower()
        if leaf in {
            "q_proj",
            "k_proj",
            "v_proj",
            "qkv_proj",
            "query_key_value",
            "c_attn",
            "wqkv",
            "wq",
            "wk",
            "wv",
        }:
            return ModuleCategory.QKV
        if leaf in {"o_proj", "out_proj", "wo"} or (
            leaf == "c_proj" and re.search(r"attn|attention", name, re.I)
        ):
            return ModuleCategory.OUTPUT
        if re.search(r"norm|layernorm", leaf + " " + class_name, re.I):
            return ModuleCategory.NORM
        if re.search(r"embed|^wte$|^wpe$", leaf + " " + class_name, re.I):
            return ModuleCategory.EMBEDDING
        if re.search(
            r"(^|\.)(mlp|ffn|feed_forward|gate_proj|up_proj|down_proj|c_fc)(\.|$)", name, re.I
        ):
            return ModuleCategory.MLP
        if re.search(r"attn|attention", name, re.I):
            return ModuleCategory.ATTENTION
        return ModuleCategory.OTHER
