from dataclasses import asdict

import numpy as np
import pytest

from goav.bank import CandidateGroup, CheapTest, TrustedSidecar, bank_hash, read_bank, trainer_view, write_bank
from goav.events import AuditDesignEvent, AuditOutcomeEvent, AuditEventLog
from goav.ledger import CostLedger
from goav.manifest import ArtifactManifest
from goav.sandbox import FakeSandbox


def _group():
    tests = tuple(
        CheapTest(f"{kind}-{i}", kind, f"test-{kind}-{i}")
        for kind, count in (("curated", 12), ("generated", 12), ("fault_targeted", 12), ("metamorphic", 8), ("control", 4))
        for i in range(count)
    )
    return CandidateGroup("p-1", "ckpt-2", tuple(f"c-{i}" for i in range(8)), tests, np.zeros((8, 48), dtype=np.int8))


def test_candidate_group_uses_problem_checkpoint_split_and_registered_test_composition():
    group = _group()
    assert group.split_group == ("p-1", "ckpt-2")
    assert group.test_composition == {"curated": 12, "generated": 12, "fault_targeted": 12, "metamorphic": 8, "control": 4}
    with pytest.raises(ValueError, match="composition"):
        CandidateGroup(group.problem_id, group.checkpoint_id, group.candidate_ids, group.tests[:-1], group.cheap_outcomes[:, :-1])


def test_trainer_view_cannot_access_trusted_labels():
    group = _group()
    sidecar = TrustedSidecar.create({("p-1", "ckpt-2"): np.array([1, 0, 1, 0, 1, 0, 1, 0])})
    view = trainer_view(group)
    assert not hasattr(view, "trusted_labels")
    assert set(asdict(view)) == {"problem_id", "checkpoint_id", "candidate_ids", "tests", "cheap_outcomes", "prompt", "candidate_texts", "duplicate_cluster_id"}
    with pytest.raises(PermissionError):
        sidecar.labels(view, evaluator_token="trainer")
    np.testing.assert_array_equal(sidecar.labels(group.split_group, evaluator_token=sidecar.evaluator_token), [1, 0, 1, 0, 1, 0, 1, 0])


def test_audit_outcome_requires_committed_design_event(tmp_path):
    log = AuditEventLog(tmp_path / "events.jsonl")
    outcome = AuditOutcomeEvent("run", "p-1", "d-1", (1, 0), (1.0,))
    with pytest.raises(RuntimeError, match="design.*before.*outcome"):
        log.append_outcome(outcome)
    design = AuditDesignEvent.create("run", "p-1", "ckpt", "d-1", [0.25, 0.25, 0.25, 0.25], [0.5, 0.5], [[0.5, 0.25], [0.25, 0.5]], "rng:7", bank_hash="sha256:" + "a" * 64, group_hash="sha256:" + "b" * 64, candidate_order_hash="sha256:" + "c" * 64, draw_index=0, uniform_draw=0.3)
    log.append_design(design)
    log.append_outcome(outcome)
    assert [type(event).__name__ for event in log.read()] == ["AuditDesignEvent", "AuditOutcomeEvent"]
    assert log.read()[0].distribution_hash.startswith("sha256:")
    assert log.read()[0].expected_budget == 1.0


def test_event_log_rejects_non_finite_outcomes(tmp_path):
    log = AuditEventLog(tmp_path / "finite-events.jsonl")
    design = AuditDesignEvent.create("run", "p", "ckpt", "design", [0.5, 0.5], [0.5], [[0.5]], "seed:1", bank_hash="sha256:" + "a" * 64, group_hash="sha256:" + "b" * 64, candidate_order_hash="sha256:" + "c" * 64, draw_index=0, uniform_draw=0.75)
    log.append_design(design)
    with pytest.raises(ValueError, match="finite"):
        log.append_outcome(AuditOutcomeEvent("run", "p", "design", (1,), (float("nan"),)))


def test_event_log_rejects_mask_inconsistent_with_registered_uniform(tmp_path):
    log = AuditEventLog(tmp_path / "draw-mask.jsonl")
    design = AuditDesignEvent.create("run", "p", "ckpt", "d", [0.5, 0.5], [0.5], [[0.5]], "seed:1", bank_hash="sha256:" + "a" * 64, group_hash="sha256:" + "b" * 64, candidate_order_hash="sha256:" + "c" * 64, draw_index=0, uniform_draw=0.75)
    log.append_design(design)
    with pytest.raises(ValueError, match="uniform draw"):
        log.append_outcome(AuditOutcomeEvent("run", "p", "d", (0,), ()))
    log.append_outcome(AuditOutcomeEvent("run", "p", "d", (1,), (1.0,)))
    rows = (tmp_path / "draw-mask.jsonl").read_text(encoding="utf-8").splitlines()
    payload = __import__("json").loads(rows[1])
    payload["audited"] = [0]
    payload["outcomes"] = []
    rows[1] = __import__("json").dumps(payload)
    (tmp_path / "draw-mask.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uniform draw"):
        log.read()


def test_three_cost_ledgers_and_deterministic_fake_sandbox():
    ledger = CostLedger()
    sandbox = FakeSandbox(ledger)
    first = sandbox.run("print(1)", "assert output == 1")
    second = sandbox.run("print(1)", "assert output == 1")
    assert first == second
    assert ledger.tokens > 0
    assert ledger.test_executions == 2
    assert ledger.cpu_seconds > 0
    assert sandbox.security_boundary == "none_deterministic_test_double"
    with pytest.raises(ValueError):
        ledger.charge(tokens=-1)


def test_canonical_bank_roundtrip_and_manifest_rejects_tampering(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    array_path = tmp_path / "outcomes.npz"
    write_bank(bank_path, array_path, [_group()])
    restored = read_bank(bank_path, array_path)
    assert restored[0].split_group == ("p-1", "ckpt-2")
    np.testing.assert_array_equal(restored[0].cheap_outcomes, _group().cheap_outcomes)
    manifest = ArtifactManifest.create([bank_path, array_path])
    manifest.verify()
    bank_path.write_text(bank_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="tamper"):
        manifest.verify()


def test_bank_bytes_are_canonical_across_writes(tmp_path):
    first_jsonl, first_npz = tmp_path / "first.jsonl", tmp_path / "first.npz"
    second_jsonl, second_npz = tmp_path / "second.jsonl", tmp_path / "second.npz"
    write_bank(first_jsonl, first_npz, [_group()])
    write_bank(second_jsonl, second_npz, [_group()])
    assert first_jsonl.read_bytes() == second_jsonl.read_bytes()
    assert first_npz.read_bytes() == second_npz.read_bytes()


def test_prepared_bank_roundtrips_prompt_and_candidate_text_for_real_backend(tmp_path):
    group = _group()
    prepared = CandidateGroup(group.problem_id, group.checkpoint_id, group.candidate_ids, group.tests, group.cheap_outcomes, prompt="solve this", candidate_texts=tuple(f"code-{i}" for i in range(8)), duplicate_cluster_id="cluster-7")
    write_bank(tmp_path / "bank.jsonl", tmp_path / "bank.npz", [prepared])
    restored = read_bank(tmp_path / "bank.jsonl", tmp_path / "bank.npz")[0]
    assert restored.prompt == "solve this"
    assert restored.candidate_texts == tuple(f"code-{i}" for i in range(8))
    assert restored.base_split_id == "cluster-7"


def test_bank_hash_binds_full_test_content_and_candidate_order():
    group = _group()
    changed_tests = list(group.tests)
    changed_tests[0] = CheapTest(changed_tests[0].test_id, changed_tests[0].kind, "different-content")
    changed = CandidateGroup(group.problem_id, group.checkpoint_id, group.candidate_ids, tuple(changed_tests), group.cheap_outcomes)
    reordered = CandidateGroup(group.problem_id, group.checkpoint_id, tuple(reversed(group.candidate_ids)), group.tests, group.cheap_outcomes[::-1])
    assert len({bank_hash([group]), bank_hash([changed]), bank_hash([reordered])}) == 3
