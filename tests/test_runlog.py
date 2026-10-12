import pytest

from common.runlog import RunLog


def test_tee_copies_output_and_errors(tmp_path, capsys):
    log = RunLog(tmp_path / "logs" / "run.log")
    with log.tee("step one"):
        print("hello 42")
    with pytest.raises(ValueError):
        with log.tee("step two"):
            raise ValueError("boom")
    text = (tmp_path / "logs" / "run.log").read_text(encoding="utf-8")
    assert "hello 42" in text and "step one done" in text
    assert "FAILED" in text and "boom" in text
    assert "hello 42" in capsys.readouterr().out
