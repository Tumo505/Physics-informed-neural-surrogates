import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import qualify_reference_restart as restart
from registry import artifact, atomic, sha


@pytest.mark.parametrize('marker', [None, restart.TIMEOUT, 'different error'])
def test_registered_timeout_with_optional_marker(tmp_path, monkeypatch, marker):
    monkeypatch.setattr(restart, 'ROOT', tmp_path)
    phase = 'reference_000547'
    parent = tmp_path / 'parent'
    target = tmp_path / 'target'
    target.mkdir()
    log = parent / phase / 'attempt-001/command.log'
    log.parent.mkdir(parents=True)
    log.write_text(f'RuntimeError: {restart.TIMEOUT}\n')
    artifact(tmp_path, parent, log, 'Executed command log')
    state = dict(phase=phase, attempts={phase: 1})

    if marker is not None:
        atomic(target / 'failure.json', dict(error=marker))
    if marker == 'different error':
        with pytest.raises(AssertionError, match='different failure marker'):
            restart.failure_evidence(parent, target, state)
        return

    expected = {} if marker is None else {'failure.json': sha(target / 'failure.json')}
    assert restart.failure_evidence(parent, target, state) == (log, expected)
    log.write_text('Changed log')
    with pytest.raises(AssertionError, match='Unregistered or changed artifact'):
        restart.failure_evidence(parent, target, state)
