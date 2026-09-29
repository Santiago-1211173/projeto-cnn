"""
Unit and Integration Test Suite for Phase 4:
Avaliacao e Metricas EAAI (evaluate_hybrid_global.py).

Validates all Phase 4 requirements and acceptance criteria from docs/Plano_de_Acao_EAAI.md:
1. Interface contract: run_baselines(noise_levels, capacity) and export_metrics(results, output_path).
2. All 5 baselines produce results: B0, B1, B2, B3, B4.
3. Output files generated: outputs/eaai_metrics.csv and outputs/eaai_metrics.json.
4. Acceptance criterion: results["B4"]["accuracy_noise_0.4"] > results["B2"]["accuracy_noise_0.4"].
5. Metric completeness: Accuracy under noise, Latency, Peak RAM, Forgetting Rate,
   Drift Restoration Time, Cache Hit Rate, Eviction Distribution Match (KL Div).
"""

import os
import sys
import json
import tempfile
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate_hybrid_global import run_baselines, export_metrics


def test_export_metrics():
    print("Testing export_metrics function...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_target = os.path.join(tmp_dir, "test_metrics.csv")
        json_target = os.path.join(tmp_dir, "test_metrics.json")

        dummy_results = {
            "B0": {"accuracy_noise_0.0": 98.0, "latency_ms": 0.05, "ram_peak_mb": 0.5},
            "B4": {"accuracy_noise_0.0": 98.5, "latency_ms": 1.20, "ram_peak_mb": 5.0},
        }

        # 1. Export via CSV target
        export_metrics(dummy_results, csv_target)
        assert os.path.exists(csv_target), "CSV file was not created"
        assert os.path.exists(json_target), "JSON complementary file was not created"

        # Validate JSON content
        with open(json_target, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "B0" in data
        assert "B4" in data
        assert np.isclose(data["B4"]["accuracy_noise_0.0"], 98.5)

    print("  -> export_metrics verified successfully.")


def test_run_baselines_micro_run():
    print("Testing run_baselines micro execution and acceptance criteria...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
        capacity = 500
        samples_per_level = 100

        results = run_baselines(
            noise_levels=noise_levels,
            capacity=capacity,
            samples_per_level=samples_per_level,
            output_dir=tmp_dir,
        )

        # 1. Check all 5 baselines are present
        assert len(results) == 5, f"Expected 5 baselines, got {len(results)}"
        for b in ["B0", "B1", "B2", "B3", "B4"]:
            assert b in results, f"Baseline {b} missing from results"

        # 2. Check output files generated in directory
        csv_file = os.path.join(tmp_dir, "eaai_metrics.csv")
        json_file = os.path.join(tmp_dir, "eaai_metrics.json")
        dashboard_file = os.path.join(tmp_dir, "eaai_evaluation_dashboard.png")

        assert os.path.exists(csv_file), f"File {csv_file} does not exist"
        assert os.path.exists(json_file), f"File {json_file} does not exist"
        assert os.path.exists(dashboard_file), f"File {dashboard_file} does not exist"

        # 3. Check all required engineering metrics exist
        required_keys = [
            "accuracy_noise_0.0",
            "accuracy_noise_0.2",
            "accuracy_noise_0.4",
            "accuracy_noise_0.6",
            "accuracy_noise_0.8",
            "mean_accuracy_under_noise",
            "overall_accuracy",
            "latency_ms",
            "ram_peak_mb",
            "forgetting_rate",
            "drift_restoration_time",
            "cache_hit_rate",
            "eviction_kl_divergence",
        ]

        for b in ["B0", "B1", "B2", "B3", "B4"]:
            for k in required_keys:
                assert k in results[b], f"Metric {k} missing from baseline {b}"

        # 4. Check acceptance criteria: B4 (RL Active Memory) beats B2 (FIFO) on noise 0.4
        acc_b4_04 = results["B4"]["accuracy_noise_0.4"]
        acc_b2_04 = results["B2"]["accuracy_noise_0.4"]
        assert acc_b4_04 > acc_b2_04, (
            f"Acceptance criterion failed: B4 ({acc_b4_04:.2f}%) <= B2 ({acc_b2_04:.2f}%)"
        )
        print(f"  -> Confirmed: B4 ({acc_b4_04:.2f}%) > B2 ({acc_b2_04:.2f}%) at noise 0.4")

        # 5. Check RAM metric integrity: B1 (unbounded) has higher peak RAM than bounded B2/B4
        assert results["B1"]["ram_peak_mb"] >= results["B2"]["ram_peak_mb"]

    print("  -> run_baselines micro execution passed successfully.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHASE 4 TEST SUITE")
    print("=" * 70)
    test_export_metrics()
    test_run_baselines_micro_run()
    print("=" * 70)
    print("ALL PHASE 4 TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
