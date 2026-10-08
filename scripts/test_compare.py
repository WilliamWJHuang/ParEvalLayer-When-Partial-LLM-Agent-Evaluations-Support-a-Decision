from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_stream_decision_adapter import consume_stream  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class CompareWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        parent = ROOT / "outputs"
        parent.mkdir(exist_ok=True)
        self.base = parent / f"test-compare-{uuid.uuid4().hex}"
        self.base.mkdir()

    def tearDown(self) -> None:
        if self.base.parent == ROOT / "outputs" and self.base.name.startswith("test-compare-"):
            shutil.rmtree(self.base, ignore_errors=True)

    def write_csv(
        self,
        name: str,
        rows: list[tuple[str, float, float, str]],
        *,
        group: bool = True,
    ) -> Path:
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            header = ["task_id", "tested_score", "comparison_score"]
            if group:
                header.append("group")
            writer.writerow(header)
            for task, tested, comparison, group_name in rows:
                values = [task, tested, comparison]
                if group:
                    values.append(group_name)
                writer.writerow(values)
        return path

    def run_compare(
        self,
        input_path: Path,
        output_name: str,
        *extra: str,
        tested: str = "tested system",
        comparison: str = "comparison system",
        cwd: Path = ROOT,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        output = self.base / output_name
        command = [
            sys.executable,
            str(ROOT / "compare.py"),
            "--input",
            str(input_path),
            "--tested",
            tested,
            "--comparison",
            comparison,
            "--threshold",
            "0",
            "--output-dir",
            str(output),
            *extra,
        ]
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True), output

    def simple_rows(self, differences: list[float]) -> list[tuple[str, float, float, str]]:
        return [
            (f"task_{index}", difference, 0.0, "all")
            for index, difference in enumerate(differences, start=1)
        ]

    def test_positive_reversed_equal_and_mixed_actions(self) -> None:
        cases = {
            "positive": ([0.4, 0.3, 0.2, 0.5], "promote"),
            "reversed": ([-0.4, -0.3, -0.2, -0.5], "reject"),
            "equal": ([0.0, 0.0, 0.0, 0.0], "reject"),
            "mixed": ([1.0, -1.0, 1.0, -1.0], "abstain"),
        }
        for name, (differences, expected) in cases.items():
            with self.subTest(name=name):
                source = self.write_csv(f"{name}.csv", self.simple_rows(differences))
                proc, output = self.run_compare(source, name, "--bootstrap-samples", "2000")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(read_jsonl(output / "decisions.jsonl")[0]["decision"]["action"], expected)

    def test_coverage_abstains_when_prefix_omits_declared_group(self) -> None:
        rows = [
            ("a1", 1.0, 0.0, "a"),
            ("a2", 1.0, 0.0, "a"),
            ("a3", 1.0, 0.0, "a"),
            ("b1", 1.0, 0.0, "b"),
        ]
        source = self.write_csv("coverage.csv", rows)
        proc, output = self.run_compare(source, "coverage", "--budget-caps", "0.5")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        decision = read_jsonl(output / "decisions.jsonl")[0]["decision"]
        self.assertEqual(decision["action"], "abstain")
        self.assertEqual(decision["abstention_reason"], "missing_required_strata")

    def test_row_order_is_preserved_and_deterministic(self) -> None:
        rows = [
            ("z", 3.0, 1.0, "g2"),
            ("a", 2.0, 1.0, "g1"),
            ("m", 4.0, 1.0, "g2"),
        ]
        source = self.write_csv("order.csv", rows)
        proc, output = self.run_compare(source, "order")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        events = read_jsonl(output / "stream.jsonl")[1:]
        self.assertEqual([event["task_key"] for event in events[::2]], ["z", "a", "m"])
        proc2, output2 = self.run_compare(source, "order-again", "--run-id", "fixed")
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        first = read_jsonl(output / "decisions.jsonl")[0]["decision"]
        second = read_jsonl(output2 / "decisions.jsonl")[0]["decision"]
        self.assertEqual(first, second)

    def test_normalized_stream_matches_generic_adapter_exactly(self) -> None:
        source = self.write_csv("parity.csv", self.simple_rows([0.4, 0.3, 0.2, 0.1]))
        proc, output = self.run_compare(
            source, "parity", "--budget-caps", "0.5", "1", "--run-id", "parity-run"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        _, _, direct = consume_stream(
            output / "stream.jsonl", [0.5, 1.0], 0.0, 0.05, 500, 20260613, None
        )
        self.assertEqual(read_jsonl(output / "decisions.jsonl"), direct)

    def test_later_scores_do_not_change_earlier_prefix_outcome(self) -> None:
        first = self.write_csv("prefix-a.csv", self.simple_rows([0.5, 0.5, 0.1, 0.1]))
        second = self.write_csv("prefix-b.csv", self.simple_rows([0.5, 0.5, -20.0, 20.0]))
        proc1, out1 = self.run_compare(first, "prefix-a", "--budget-caps", "0.5", "--source", "fixed")
        proc2, out2 = self.run_compare(second, "prefix-b", "--budget-caps", "0.5", "--source", "fixed")
        self.assertEqual(proc1.returncode, 0, proc1.stderr)
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        card1 = read_jsonl(out1 / "decisions.jsonl")[0]
        card2 = read_jsonl(out2 / "decisions.jsonl")[0]
        card1["event_stream"].pop("run_id")
        card2["event_stream"].pop("run_id")
        self.assertEqual(card1, card2)

    def test_arbitrary_finite_scale_and_score_swap(self) -> None:
        rows = [("a", -5.0, -8.0, "all"), ("b", 12.0, 9.0, "all")]
        source = self.write_csv("scale.csv", rows)
        swapped = self.write_csv(
            "scale-swapped.csv", [(task, comparison, tested, group) for task, tested, comparison, group in rows]
        )
        proc1, out1 = self.run_compare(source, "scale")
        proc2, out2 = self.run_compare(swapped, "scale-swapped")
        self.assertEqual(proc1.returncode, 0, proc1.stderr)
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        mean1 = read_jsonl(out1 / "decisions.jsonl")[0]["decision"]["partial_delta"]
        mean2 = read_jsonl(out2 / "decisions.jsonl")[0]["decision"]["partial_delta"]
        self.assertEqual(mean1, -mean2)
        self.assertGreater(mean1, 1.0)

    def test_labels_do_not_change_decisions(self) -> None:
        source = self.write_csv("labels.csv", self.simple_rows([0.5, 0.4, 0.3]))
        proc1, out1 = self.run_compare(source, "labels-a", tested="A", comparison="B")
        proc2, out2 = self.run_compare(source, "labels-b", tested="renamed A", comparison="renamed B")
        self.assertEqual(proc1.returncode, 0, proc1.stderr)
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        self.assertEqual(
            read_jsonl(out1 / "decisions.jsonl")[0]["decision"],
            read_jsonl(out2 / "decisions.jsonl")[0]["decision"],
        )

    def test_missing_group_column_defaults_to_all(self) -> None:
        source = self.write_csv("no-group.csv", self.simple_rows([0.2, 0.3]), group=False)
        proc, output = self.run_compare(source, "no-group")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        manifest = read_jsonl(output / "stream.jsonl")[0]
        self.assertEqual(manifest["group_counts"], {"all": 2})

    def test_rejects_malformed_csvs(self) -> None:
        invalid = {
            "missing.csv": "task_id,tested_score\na,1\nb,2\n",
            "unknown.csv": "task_id,tested_score,comparison_score,extra\na,1,0,x\nb,2,0,x\n",
            "duplicate-header.csv": "task_id,tested_score,tested_score,comparison_score\na,1,1,0\nb,2,2,0\n",
            "short-row.csv": "task_id,tested_score,comparison_score\na,1\nb,2,0\n",
            "blank-task.csv": "task_id,tested_score,comparison_score\n,1,0\nb,2,0\n",
            "blank-group.csv": "task_id,tested_score,comparison_score,group\na,1,0,\nb,2,0,g\n",
            "duplicate-task.csv": "task_id,tested_score,comparison_score\na,1,0\na,2,0\n",
            "missing-score.csv": "task_id,tested_score,comparison_score\na,,0\nb,2,0\n",
            "nonfinite.csv": "task_id,tested_score,comparison_score\na,NaN,0\nb,2,0\n",
            "overflow-difference.csv": "task_id,tested_score,comparison_score\na,1e308,-1e308\nb,2,0\n",
            "one-row.csv": "task_id,tested_score,comparison_score\na,1,0\n",
        }
        for name, content in invalid.items():
            with self.subTest(name=name):
                path = self.base / name
                path.write_text(content, encoding="utf-8")
                proc, output = self.run_compare(path, f"invalid-{name}")
                self.assertNotEqual(proc.returncode, 0)
                self.assertFalse(output.exists())

    def test_rejects_invalid_options_and_labels(self) -> None:
        source = self.write_csv("options.csv", self.simple_rows([0.2, 0.3]))
        cases = [
            ("negative-threshold", ["--threshold", "-1"]),
            ("nan-threshold", ["--threshold", "nan"]),
            ("zero-cap", ["--budget-caps", "0"]),
            ("large-cap", ["--budget-caps", "1.1"]),
            ("duplicate-cap", ["--budget-caps", "0.5", "0.5"]),
            ("zero-alpha", ["--alpha", "0"]),
            ("zero-samples", ["--bootstrap-samples", "0"]),
            ("negative-seed", ["--seed", "-1"]),
        ]
        for name, extra in cases:
            with self.subTest(name=name):
                proc, _ = self.run_compare(source, f"bad-{name}", *extra)
                self.assertNotEqual(proc.returncode, 0)
        proc, _ = self.run_compare(source, "same-label", tested="same", comparison="same")
        self.assertNotEqual(proc.returncode, 0)
        proc, _ = self.run_compare(source, "blank-label", tested="   ")
        self.assertNotEqual(proc.returncode, 0)

    def test_refuses_nonempty_output_without_overwrite(self) -> None:
        source = self.write_csv("overwrite.csv", self.simple_rows([0.2, 0.3]))
        proc, output = self.run_compare(source, "occupied")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        before = (output / "run.json").read_bytes()
        proc2, _ = self.run_compare(source, "occupied")
        self.assertNotEqual(proc2.returncode, 0)
        self.assertIn("Choose a new --output-dir", proc2.stderr)
        self.assertEqual((output / "run.json").read_bytes(), before)

    def test_paths_with_spaces_and_invocation_outside_project_cwd(self) -> None:
        rows = self.simple_rows([0.2, 0.3])
        source = self.write_csv("inputs with spaces/paired data.csv", rows)
        proc, output = self.run_compare(
            source, "results with spaces", cwd=ROOT.parent
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((output / "report.md").is_file())
        self.assertIn(str(output.resolve()), proc.stdout)

    def test_artifact_set_and_report_scope_language(self) -> None:
        source = self.write_csv("artifacts.csv", self.simple_rows([0.2, 0.3]))
        proc, output = self.run_compare(source, "artifacts")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {"stream.jsonl", "decisions.jsonl", "trace.jsonl", "summary.csv", "report.md", "run.json"},
        )
        report = (output / "report.md").read_text(encoding="utf-8")
        for phrase in ("not establish that the tested system is worse", "fixed-prefix retrospective", "no live monitoring", "without multiplicity adjustment", "not p-values"):
            self.assertIn(phrase, report)

    def test_packaged_example_is_clearly_illustrative_and_has_eight_pairs(self) -> None:
        path = ROOT / "examples" / "paired_results.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 8)
        self.assertTrue(all(row["task_id"].startswith("illustrative_") for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
