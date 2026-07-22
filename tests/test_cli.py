from __future__ import annotations

from maablackflow.cli import main


def test_cli_success_output(capsys) -> None:
    code = main(["solve", "examples/maps/basic.json"])
    output = capsys.readouterr().out
    assert code == 0
    assert "是否到达出口: 是" in output
    assert "总收益:" in output
    assert "加工品使用情况:" in output


def test_cli_failure_output(capsys) -> None:
    code = main(["solve", "examples/maps/no_solution.json"])
    output = capsys.readouterr().out
    assert code == 1
    assert "是否到达出口: 否" in output
    assert "无解原因:" in output
