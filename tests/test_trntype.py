import json
import pandas as pd

from utils.rules import load_rules
from utils.trntype import infer_trntype_series


def test_infer_trntype_series_uses_custom_rules(tmp_path):
    config_path = tmp_path / "rules.json"
    config_path.write_text(
        json.dumps(
            {
                "source_aliases": {"XYZ": "PAYMENT"},
                "keyword_rules": {
                    "extend": [
                        {
                            "pattern": r"\bESPRESSO\b",
                            "trntype": "CASH",
                        }
                    ]
                },
            }
        )
    )

    custom_rules = load_rules(config_path)

    amounts = pd.Series([10.0, -12.5])
    trntype_text = pd.Series(["xyz", None])
    cleaned_desc = pd.Series([None, "Morning espresso run"])

    result = infer_trntype_series(
        amounts,
        trntype_text,
        cleaned_desc,
        rules=custom_rules,
    )

    assert list(result) == ["PAYMENT", "CASH"]
