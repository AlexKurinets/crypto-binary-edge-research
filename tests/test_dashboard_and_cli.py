import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from edgelab.cli import main

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASHBOARD))


@pytest.mark.parametrize("page", ["overview", "calibration", "hypotheses", "conditions", "paper"])
def test_every_dashboard_page_renders(pipeline, page):
    at = AppTest.from_file(str(DASHBOARD / "views" / f"{page}.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value


def test_app_entry_point_renders(pipeline):
    at = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception


def test_power_command(capsys):
    assert main(["power", "--edge", "0.02"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["independent_outcomes_needed"] == 3865
    assert out["fee_per_contract_at_price"] == pytest.approx(0.0176)
    assert out["breakeven_win_rate_at_ask"] == pytest.approx(0.5176)


def test_report_command(pipeline, tmp_path, capsys):
    out = tmp_path / "report.md"
    assert main(["report", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Hypotheses" in text and "H02" in text and "## Paper period" in text
