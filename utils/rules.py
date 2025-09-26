"""Rule configuration helpers for transaction type inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Pattern, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when PyYAML is unavailable
    yaml = None  # type: ignore

@dataclass(frozen=True)
class RuleSet:
    """Container for transaction type inference rules."""

    source_aliases: Mapping[str, str]
    rules_regex: Tuple[Tuple[Pattern, str], ...]
    keyword_rules: Tuple[Tuple[str, str], ...]


_DEFAULT_SOURCE_ALIASES = {
    "D": "DEBIT",
    "DR": "DEBIT",
    "DBT": "DEBIT",
    "WITHDRAWAL": "DEBIT",
    "W/D": "DEBIT",
    "WD": "DEBIT",
    "DEPOSIT": "DEP",
    "DEP": "DEP",
    "C": "CREDIT",
    "CR": "CREDIT",
    "XFR": "XFER",
    "TRANSFER": "XFER",
    "PAY": "PAYMENT",
    "PMT": "PAYMENT",
}

_DEFAULT_RULES_REGEX = (
    (re.compile(r"\b(?:VENMO|CASH\s+APP|ZELLE|APPLE\s+CASH|P2P)\b", re.I), "PAYMENT"),
    (re.compile(r"\bMOBILE\s+DEPOSIT\b", re.I), "DEP"),
    (re.compile(r"\bCHECK\b", re.I), "CHECK"),
    (re.compile(r"\bATM\b", re.I), "ATM"),
    (re.compile(r"\bCASH(?:\s+(?:WITHDRAWAL|DEPOSIT))?\b", re.I), "CASH"),
    (re.compile(r"\bPOS(?:\s+PURCHASE)?\b", re.I), "POS"),
    (
        re.compile(
            r"\b(?:UTIL(?:ITY)?|ACH|PPD|CCD|(?:AUTO|AUTOP|BILL|DIRECT|ONLINE|WEB|E(?:LECTRONIC)?)[-_/.\s]*(?:P"
            r"(?:AY)?(?:MENT|MNT|MT)?|PMT|PMNT|PYMT|PYMNT))\b",
            re.I,
        ),
        "DIRECTDEBIT",
    ),
    (
        re.compile(
            r"\b(?:PAYROLL|IRS(?:\s*REFUND)?|SSA|SOCIAL\s+SECURITY|(?:STATE\s*)?TREAS)\b",
            re.I,
        ),
        "DIRECTDEP",
    ),
    (re.compile(r"\b(?:TRANSFER|(?:EXT-|EXTERNAL\s*)?XFER|XFR)\b", re.I), "XFER"),
    (re.compile(r"\b(?:INT(?:EREST)?|FINANCE\s*CHARGE|APR)\b", re.I), "INT"),
    (
        re.compile(
            r"\b(?:SERVICE\s*CHARGE|MONTHLY\s*SERVICE|MAINT(?:ENANCE)?\s*FEE)\b",
            re.I,
        ),
        "SRVCHG",
    ),
    (
        re.compile(
            r"\b(?:OVERDRAFT|NSF|WIRE\s*FEE|RTN\s*ITEM(?:\s*FEE)?|STOP\s*PAY(?:MENT)?\s*FEE|FEE)\b",
            re.I,
        ),
        "FEE",
    ),
    (re.compile(r"\bDIV(?:IDEND)?\b", re.I), "DIV"),
    (re.compile(r"\bREV(?:ERSAL)?\b", re.I), "CREDIT"),
    (re.compile(r"\bRETURN(?!ED\s+ITEM\s+FEE)\b", re.I), "CREDIT"),
    (re.compile(r"\bE-?PAY(?:MENT)?\b", re.I), "DIRECTDEBIT"),
    (re.compile(r"\bPAYMNT\b", re.I), "DIRECTDEBIT"),
    (re.compile(r"\bACH\s*PAY(?:MENT)?\b", re.I), "DIRECTDEBIT"),
    (re.compile(r"\bDISCOVER\s*E-?PAYMENT\b", re.I), "DIRECTDEBIT"),
)

_DEFAULT_KEYWORD_RULES = (
    (r"\bDEPOSIT\b", "DEP"),
    (r"\bINTEREST\b", "INT"),
    (r"\bINT\b", "INT"),
    (r"\bDIVIDEND\b", "DIV"),
    (r"\bDIV\b", "DIV"),
    (r"\bSRVCHG\b", "FEE"),
    (r"\bFEE\b", "FEE"),
    (r"\bCHECK\b", "CHECK"),
    (r"\bATM\b", "ATM"),
    (r"\bPURCHASE\b", "POS"),
    (r"\bPOS\b", "POS"),
    (r"\bTRANSFER\b", "XFER"),
    (r"\bXFER\b", "XFER"),
    (r"\bXFR\b", "XFER"),
    (r"\bWITHDRAW\b", "DEBIT"),
    (r"\bWD\b", "DEBIT"),
    (r"\bPAYMENT\b", "PAYMENT"),
    (r"\bPMT\b", "PAYMENT"),
    (r"\bREFUND\b", "CREDIT"),
    (r"\bPAYROLL\b", "DIRECTDEP"),
    (r"\bCASH\b", "CASH"),
)

DEFAULT_RULES = RuleSet(
    source_aliases=dict(_DEFAULT_SOURCE_ALIASES),
    rules_regex=tuple(_DEFAULT_RULES_REGEX),
    keyword_rules=tuple(_DEFAULT_KEYWORD_RULES),
)

_FLAG_MAP = {
    "ASCII": re.ASCII,
    "A": re.ASCII,
    "IGNORECASE": re.IGNORECASE,
    "I": re.IGNORECASE,
    "LOCALE": re.LOCALE,
    "L": re.LOCALE,
    "MULTILINE": re.MULTILINE,
    "M": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "S": re.DOTALL,
    "UNICODE": re.UNICODE,
    "U": re.UNICODE,
    "VERBOSE": re.VERBOSE,
    "X": re.VERBOSE,
}


def load_rules(
    config_path: Optional[Union[str, Path]] = None,
    *,
    base_rules: RuleSet = DEFAULT_RULES,
) -> RuleSet:
    """Load a :class:`RuleSet` from an optional JSON or YAML configuration file."""

    if config_path is None:
        return base_rules

    path = Path(config_path)
    overrides = _load_config_data(path)
    return apply_rule_overrides(base_rules, overrides)


def apply_rule_overrides(base_rules: RuleSet, overrides: Mapping[str, Any]) -> RuleSet:
    """Create a new :class:`RuleSet` by applying overrides to *base_rules*."""

    if not overrides:
        return base_rules

    aliases = _merge_aliases(base_rules.source_aliases, overrides.get("source_aliases"))
    regex_rules = _merge_rule_sequences(
        base_rules.rules_regex,
        overrides.get("rules_regex"),
        _parse_regex_rule,
    )
    keyword_rules = _merge_rule_sequences(
        base_rules.keyword_rules,
        overrides.get("keyword_rules"),
        _parse_keyword_rule,
    )

    return RuleSet(
        source_aliases=aliases,
        rules_regex=tuple(regex_rules),
        keyword_rules=tuple(keyword_rules),
    )


def _load_config_data(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Rule override file not found: {path}")

    text = path.read_text()
    if not text.strip():
        return {}

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML rule files")
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported rule file format: {path.suffix}")

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError("Rule configuration must be a mapping")
    return data


def _merge_aliases(
    base_aliases: Mapping[str, str],
    override: Optional[Any],
) -> MutableMapping[str, str]:
    aliases: MutableMapping[str, str] = dict(base_aliases)
    if override is None:
        return aliases

    if isinstance(override, Mapping) and ("extend" in override or "replace" in override):
        if "replace" in override:
            replacement = override.get("replace") or {}
            if not isinstance(replacement, Mapping):
                raise TypeError("Alias replacement must be a mapping")
            aliases = dict(replacement)
        if "extend" in override:
            extension = override.get("extend") or {}
            if not isinstance(extension, Mapping):
                raise TypeError("Alias extensions must be a mapping")
            aliases.update(extension)
    elif isinstance(override, Mapping):
        aliases.update(override)
    else:
        raise TypeError("Alias override must be a mapping")

    return aliases


def _merge_rule_sequences(
    base_rules: Sequence[Any],
    override: Optional[Any],
    parser,
) -> list:
    if override is None:
        return list(base_rules)

    if isinstance(override, Mapping):
        result = list(base_rules)
        if "replace" in override:
            replace_values = override.get("replace") or []
            result = [_parse_rule_entry(parser, item) for item in _ensure_iterable(replace_values)]
        if "extend" in override:
            extend_values = override.get("extend") or []
            result.extend(
                _parse_rule_entry(parser, item) for item in _ensure_iterable(extend_values)
            )
        return result

    return [_parse_rule_entry(parser, item) for item in _ensure_iterable(override)]


def _parse_rule_entry(parser, entry: Any):
    if isinstance(entry, Mapping):
        return parser(entry)
    if isinstance(entry, (list, tuple)):
        return parser(entry)
    raise TypeError("Rule entries must be mappings or sequences")


def _parse_regex_rule(entry: Any) -> Tuple[Pattern, str]:
    if isinstance(entry, Mapping):
        pattern = entry.get("pattern")
        output = entry.get("trntype") or entry.get("output")
        flags = entry.get("flags")
    else:
        if len(entry) < 2:
            raise ValueError("Regex rule entries must have at least two elements")
        pattern, output, *rest = entry
        flags = rest[0] if rest else None

    if pattern is None or output is None:
        raise ValueError("Regex rule entries require 'pattern' and 'trntype'/'output'")

    compiled = re.compile(str(pattern), _parse_regex_flags(flags))
    return compiled, str(output)


def _parse_keyword_rule(entry: Any) -> Tuple[str, str]:
    if isinstance(entry, Mapping):
        pattern = entry.get("pattern")
        output = entry.get("trntype") or entry.get("output")
    else:
        if len(entry) < 2:
            raise ValueError("Keyword rule entries must have at least two elements")
        pattern, output = entry[:2]

    if pattern is None or output is None:
        raise ValueError("Keyword rule entries require 'pattern' and 'trntype'/'output'")

    return str(pattern), str(output)


def _parse_regex_flags(flags: Any) -> int:
    if flags is None:
        return re.IGNORECASE
    if isinstance(flags, int):
        return flags
    if isinstance(flags, str):
        tokens = [flags]
    elif isinstance(flags, Sequence) and not isinstance(flags, (bytes, str)):
        tokens = list(flags)
    else:
        raise TypeError("Regex flag overrides must be an int, string, or sequence of strings")

    result = 0
    for token in tokens:
        if isinstance(token, int):
            result |= token
            continue
        name = str(token).upper()
        if name not in _FLAG_MAP:
            raise ValueError(f"Unsupported regex flag token: {token}")
        result |= _FLAG_MAP[name]
    return result or 0


def _ensure_iterable(value: Any) -> Iterable:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


__all__ = [
    "RuleSet",
    "DEFAULT_RULES",
    "load_rules",
    "apply_rule_overrides",
]
