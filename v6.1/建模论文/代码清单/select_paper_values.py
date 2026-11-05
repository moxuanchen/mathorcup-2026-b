"""Build the ASCII-key paper selection from current v5.0 result files."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "求解结果" / "数据"
OUT = DATA / "00_paper_values_v50_selected.json"


def load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def main() -> int:
    q1 = load("01c_统一关联前五.json")
    gully = load("02_沟道临界条件.json")
    old = load("02_旧方法审计.json")
    trigger = load("02_触发评估模型.json")
    api_quantiles = load("02_API分位.json")
    state_tuple = load("02_状态元组.json")
    reliability = load("02_多尺度可靠性.json")
    q3 = load("03_前向预报.json")
    distance = load("03_空间距离范围.json")
    q4 = load("04b_二维引擎.json")
    baseline = load("04_仿真基线.json")

    top5 = [
        {
            "rank": row["rank"],
            "variable": row["variable"],
            "name": row["name"],
            "family": row["family"],
            "direction": row["direction"],
            "A_abs": row["A_abs"],
            "signed_z": row["signed_z"],
            "delta": row["delta"],
            "baseline_median": row["baseline_median"],
            "tail_rank": row["tail_rank"],
            "cluster_members": row["cluster_members"],
            "cluster_representative_reason": row["cluster_representative_reason"],
        }
        for row in q1["top5"]
    ]
    sensitivity72 = q1["sensitivity_72h"]["top5"]
    leave = {
        year: info["top5_overlap"]
        for year, info in q1["leave_one_reference_year_out"].items()
    }
    pseudo = q1["pseudo_2025_same_calendar"]
    rank = gully["rank_test_statistics"]
    runs = list(q4["runs"].values())
    base = q4["runs"]["base"]

    payload = {
        "q1": {
            "top5": top5,
            "refs": q1["main_168h"]["reference_window_count"],
            "sensitivity72": sensitivity72,
            "leave": leave,
            "pseudo": pseudo["top5"],
            "pseudo_max_A": pseudo["max_A"],
        },
        "q2": {
            "states": gully["机械状态计数"],
            "qcrit_q": gully["起动流量_m3s"],
            "old_q": gully["旧式起动流量_m3s"],
            "rank": rank,
            "formal_qcrit_m2s": old["formal_q_crit_m2s"],
            "formal_C_N0": old["formal_C_N0"],
            "CN_provenance": old["formal_CN_provenance"],
            "CN_sensitivity": trigger["曲线数情景"],
            "legacy_k": old["legacy_k"],
            "event_peak": old["event_window_T_peak"],
            "event_T": trigger["T_事件时刻_10:00"],
            "event_first": trigger["事件窗口首次超过0.5"],
            "api_quantiles": api_quantiles,
            "state_tuple": state_tuple,
            "auc": trigger.get("空间排序AUC_易发性预测大规模"),
            "reliability": reliability,
        },
        "q3": {
            "issue": q3["issue_time_local"],
            "tm": q3["intervals"]["t_m"],
            "ta": q3["intervals"]["t_a"],
            "unconditional_status": q3["intervals"]["unconditional_status"],
            "unconditional_status_by_target": q3["intervals"][
                "unconditional_status_by_target"
            ],
            "tm_state": q3["state_summary_tm"]["counts"],
            "tm_state_summary": q3["state_summary_tm"],
            "ta_state": q3["state_summary_ta"]["counts"],
            "ta_state_summary": q3["state_summary_ta"],
            "warning": q3["warning"],
            "coverage": q3["coverage"],
            "distance": distance,
            "ifs": q3["ifs_manifest"],
            "fit": q3["fit"],
        },
        "q4": {
            "base_arrival_s": base["arrival_port_s"],
            "base_arrival_min": base["arrival_port_s"] / 60.0,
            "runout_L_m": base["runout_L_m"],
            "runout_right": bool(base["runout_right_censored"]),
            "termination_reason": base["termination_reason"],
            "loop_termination_reason": base["loop_termination_reason"],
            "natural_stop": bool(base["natural_stop"]),
            "domain_censor": bool(base["domain_censor"]),
            "critical_depth_slope_product_m": base["critical_depth_slope_product_m"],
            "critical_depth_port_m": base["critical_depth_port_m"],
            "domain_front_max": base["domain_front_max_m"],
            "configs": len(runs),
            "config_arrival_min": [row["arrival_port_s"] / 60.0 for row in runs],
            "max_depth": base["max_depth_m"],
            "final_max_depth": base["final_max_depth_m"],
            "peak_port": base["peak_depth_port_m"],
            "final_port": base["final_depth_port_m"],
            "peak_v": base["peak_velocity_port_m_s"],
            "closure": base["closure_max"],
            "cfl": base["actual_cfl_max"],
            "one_d_base_arrival_s": baseline["arrival_outlet_s"],
            "one_d_runout_km": baseline["runout_km"],
            "one_d_reached_outlet": baseline["reached_outlet"],
        },
    }
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"paper selection written: {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
