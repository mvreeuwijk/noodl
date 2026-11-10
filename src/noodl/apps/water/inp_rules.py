"""The `[RULES]` section of an EPANET 2.2 `.inp`, read into `Rule`s in SI."""

from __future__ import annotations

from noodl.apps.water.controls import Action, Premise, Rule, status_code
from noodl.apps.water.inp_times import clock_time, time_with_unit

_RULE_KEYWORDS = ("RULE", "IF", "THEN", "ELSE", "AND", "OR", "PRIORITY")
_RULE_RELATIONS = {
    "=": "=", "IS": "=", "<>": "<>", "NOT": "<>", "<": "<", "BELOW": "<", ">": ">",
    "ABOVE": ">", "<=": "<=", ">=": ">=",
}


def read_rules(
    lines, path, *, flow, length, pressure, power, junctions, tanks, reservoirs,
    link_kinds, link_setting,
) -> list[Rule]:
    """The `[RULES]` section as `Rule`s, in SI (Appendix C.2.19).

    A rule is `RULE <id>`, one `IF` and any number of `AND`/`OR` premises, one `THEN` and
    any number of `AND` actions, optionally `ELSE` with its own, and an optional `PRIORITY`.
    Clauses may be broken over lines anywhere, so the words are split at the keywords
    `RULE IF THEN ELSE AND OR PRIORITY` as `wntr` does. GRADE is read as HEAD.
    A premise is `<NODE|LINK|SYSTEM> <id>
    <attribute> <relation> <value>` (no id after SYSTEM); its value is converted by its
    attribute: DEMAND and FLOW by the flow unit, HEAD and LEVEL as lengths, PRESSURE as a
    pressure, POWER to kW, FILLTIME and DRAINTIME from hours, TIME and CLOCKTIME to seconds,
    a link's SETTING as the link kind's setting and a STATUS to its code.
    """
    clauses: list[tuple[str, list[str], object]] = []
    for line in lines:
        for word in line.fields:
            if word.upper() in _RULE_KEYWORDS:
                clauses.append((word.upper(), [], line))
            elif clauses:
                clauses[-1][1].append(word)
            else:
                raise ValueError(f"{path}: line {line.number}: [RULES] content "
                                 f"{line.raw.strip()!r} before the first RULE")

    def node_scale(attribute: str, line) -> float:
        if attribute == "DEMAND":
            return flow
        if attribute in ("HEAD", "LEVEL"):
            return length
        if attribute == "PRESSURE":
            return pressure
        if attribute in ("FILLTIME", "DRAINTIME"):
            return 3600.0
        raise ValueError(f"{path}: line {line.number}: unknown node attribute {attribute!r} "
                         f"in a rule")

    def number(text: str, line) -> float:
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{path}: line {line.number}: {text!r} is not a number in a "
                             f"rule") from exc

    def premise(conjunction: str, words: list[str], line) -> Premise:
        if not words:
            raise ValueError(f"{path}: line {line.number}: an empty {conjunction} clause")
        kind = words[0].upper()
        if kind == "SYSTEM":
            ident, rest = "", words[1:]
            category = "SYSTEM"
        elif kind in ("NODE", "JUNCTION", "RESERVOIR", "TANK"):
            ident, rest, category = (words[1] if len(words) > 1 else ""), words[2:], "NODE"
        elif kind in ("LINK", "PIPE", "PUMP", "VALVE"):
            ident, rest, category = (words[1] if len(words) > 1 else ""), words[2:], "LINK"
        else:
            raise ValueError(f"{path}: line {line.number}: a rule premise starts with NODE, "
                             f"LINK or SYSTEM, got {words[0]!r}")
        if len(rest) < 3:
            raise ValueError(f"{path}: line {line.number}: a rule premise needs an "
                             f"attribute, a relation and a value, got {' '.join(words)!r}")
        attribute, relation = rest[0].upper(), _RULE_RELATIONS.get(rest[1].upper())
        if attribute == "GRADE":  # EPANET (rules.c) reads GRADE as a synonym of HEAD
            attribute = "HEAD"
        if relation is None:
            raise ValueError(f"{path}: line {line.number}: unknown relation {rest[1]!r} in a "
                             f"rule")
        text = rest[2]
        if category == "SYSTEM":
            if attribute == "TIME":
                value = time_with_unit(rest, 2, path, line)
            elif attribute == "CLOCKTIME":
                value = clock_time(text, rest[3] if len(rest) > 3 else None, path, line)
            elif attribute == "DEMAND":
                value = number(text, line) * flow
            elif attribute == "DAY":
                value = number(text, line)
            else:
                raise ValueError(f"{path}: line {line.number}: unknown SYSTEM attribute "
                                 f"{attribute!r} in a rule")
        elif category == "NODE":
            if ident not in junctions and ident not in tanks and ident not in reservoirs:
                raise ValueError(f"{path}: line {line.number}: a rule names node {ident!r}, "
                                 f"which is not a node of this network")
            value = number(text, line) * node_scale(attribute, line)
        else:
            if ident not in link_kinds:
                raise ValueError(f"{path}: line {line.number}: a rule names link {ident!r}, "
                                 f"which is not a link of this network")
            if attribute == "STATUS":
                code = status_code(text)
                if code is None:
                    raise ValueError(f"{path}: line {line.number}: a link STATUS is OPEN, "
                                     f"CLOSED or ACTIVE, got {text!r}")
                value = float(code)
            elif attribute == "FLOW":
                value = number(text, line) * flow
            elif attribute == "POWER":
                value = number(text, line) * power
            elif attribute == "SETTING":
                value = link_setting(ident, number(text, line), line)
            else:
                raise ValueError(f"{path}: line {line.number}: unknown link attribute "
                                 f"{attribute!r} in a rule")
        return Premise(conjunction, category, ident, attribute, relation, value)

    def action(words: list[str], line) -> Action:
        if len(words) != 5 or words[3].upper() != "IS":
            raise ValueError(f"{path}: line {line.number}: a rule action reads 'LINK <id> "
                             f"STATUS|SETTING IS <value>', got {' '.join(words)!r}")
        if words[0].upper() not in ("LINK", "PIPE", "PUMP", "VALVE"):
            raise ValueError(f"{path}: line {line.number}: a rule action acts on a link, "
                             f"got {words[0]!r}")
        link, attribute, text = words[1], words[2].upper(), words[4]
        if link not in link_kinds:
            raise ValueError(f"{path}: line {line.number}: a rule acts on link {link!r}, "
                             f"which is not a link of this network")
        if attribute == "STATUS":
            code = status_code(text)
            if code is None:
                raise ValueError(f"{path}: line {line.number}: a link STATUS is OPEN, "
                                 f"CLOSED or ACTIVE, got {text!r}")
            return Action(link, "STATUS", float(code))
        if attribute == "SETTING":
            return Action(link, "SETTING", link_setting(link, number(text, line), line))
        raise ValueError(f"{path}: line {line.number}: a rule action sets STATUS or SETTING, "
                         f"got {words[2]!r}")

    rules: list[Rule] = []
    current: dict | None = None
    mode = ""
    for keyword, words, line in clauses:
        if keyword == "RULE":
            if len(words) != 1:
                raise ValueError(f"{path}: line {line.number}: RULE takes one name")
            current = {"name": words[0], "if": [], "then": [], "else": [], "priority": 0.0}
            rules.append(current)  # type: ignore[arg-type]
            mode = "RULE"
            continue
        if current is None:
            raise ValueError(f"{path}: line {line.number}: {keyword} before the first RULE")
        if keyword == "IF":
            mode = "IF"
            current["if"].append(premise("IF", words, line))
        elif keyword in ("AND", "OR") and mode == "IF":
            current["if"].append(premise(keyword, words, line))
        elif keyword == "THEN" or (keyword == "AND" and mode == "THEN"):
            mode = "THEN"
            current["then"].append(action(words, line))
        elif keyword == "ELSE" or (keyword == "AND" and mode == "ELSE"):
            mode = "ELSE"
            current["else"].append(action(words, line))
        elif keyword == "PRIORITY":
            if len(words) != 1:
                raise ValueError(f"{path}: line {line.number}: PRIORITY takes one number")
            current["priority"] = number(words[0], line)
        else:
            raise ValueError(f"{path}: line {line.number}: {keyword} out of place in a rule")
    return [
        Rule(r["name"], tuple(r["if"]), tuple(r["then"]), tuple(r["else"]), r["priority"])
        for r in rules  # type: ignore[index]
    ]
