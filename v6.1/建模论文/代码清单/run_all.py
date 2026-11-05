"""一键复现：按依赖顺序运行全部计算与图件脚本。

用法
----
    python run_all.py            # 全流程
    python run_all.py --skip-fig # 只跑计算
    python run_all.py --only 04  # 只跑问题4，需已准备依赖

依赖顺序
--------
step00 → step04 → step04b → step01 → step01b → step01c → step02 → step02_figures → step04_figures
       → step03 → step03_figures
问题1/2 的起动流量 q_c 以问题4 的中心线比降为唯一口径（建模成果/问题2.md §2.4.2），
故 step04 必须先于 step01 运行——它产出 04_中心线校验.json 与 04_仿真基线.json。
step04b（二维过程引擎）消费 step04 的 04_初始体积.json（链A 源区与体积族），
产出 04b_二维引擎.json（到达口岸样本 + 质量账本），供 step03 的同断面情景对照消费；
两者都以口岸像元（85.3826°E, 28.2826°N）为同一到达目标。
问题3 依赖问题2（触发序列）与问题4/04b（运行段到达时间），排在 02 之后。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = sys.executable
LOG_PATH = ROOT / "求解结果" / "运行记录" / "实验日志.jsonl"


def log_event(record: dict) -> None:
    """把一条运行事实追加到 JSONL 实验日志。

    日志按行独立成 JSON，便于逐行解析与追加，不需要读入全文件。
    写入失败不阻断主流程——复现性记录不应让计算失败。
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), **record}
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:                                    # pragma: no cover
        print(f"      警告：实验日志写入失败（{exc}）")

STEPS = [
    ("prep", "prepare_inputs.py", "原始附件清点与解压"),
    ("00", "step00_data_inventory.py", "数据清单与基础统计"),
    ("04", "step04_simulation.py", "问题4 仿真环境与参数（产出中心线口径）"),
    ("04b", "step04b_sim2d.py", "问题4 栅格路由过程引擎（数值验收与情景对照）"),
    ("04c", "step04c_dynamics.py", "问题4 有限库存分段动力对照与解析验收"),
    ("01", "step01_boundary_layer.py", "问题1 边界层结构与关联性排序"),
    ("01b", "step01b_preimpact.py", "问题1 静力修正与标签外冲击前旁证"),
    ("01c", "step01c_unified_association.py", "问题1 31项统一候选与环境关联前五"),
    ("02", "step02_trigger_threshold.py", "问题2 临界条件与触发评估模型"),
    ("02b", "step02b_spatial.py", "问题2 像元环境对照与分组验证"),
    ("02c", "step02c_spatial_matched.py", "问题2 共同沟谷域与内外层空间隔离"),
    ("02f", "step02_figures.py", "问题2 图件"),
    ("04f", "step04_figures.py", "问题4 图件"),
    ("04bf", "step04b_figures.py", "问题4 栅格路由过程图件"),
    ("04bmf", "step04b_map.py", "问题4 首达时间场与终态停积双联地图"),
    ("03", "step03_forecast.py", "问题3 预报模型与区间"),
    ("03s", "step03_time_support_sensitivity.py", "问题3 历史小时支持组合敏感性"),
    ("03f", "step03_figures.py", "问题3 图件"),
    ("06", "step06_data_audit.py", "同日界外部核验与目标坐标敏感性"),
    ("07", "archive_external.py", "外部响应哈希归档（离线复核）"),
    ("05s", "select_paper_values.py", "从当前结果生成本版论文精选值"),
    ("v50", "validate_v50.py", "v5.0 全链确定性验收"),
    ("05", "export_summary.py", "当前结果与逐值来源导出"),
]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fig", action="store_true", help="跳过图件脚本")
    ap.add_argument("--only", default=None, help="只运行以该前缀开头的步骤")
    ap.add_argument("--from-step", default=None, help="resume from an existing dependency checkpoint")
    args = ap.parse_args()

    steps = STEPS
    if args.from_step:
        index=next(i for i,s in enumerate(steps) if s[0]==args.from_step)
        steps=steps[index:]
    if args.skip_fig:
        steps = [s for s in steps if not s[0].endswith("f")]
    if args.only:
        k = args.only
        steps = [s for s in steps if s[0].startswith(k)]

    if not steps:
        print("没有匹配的步骤")
        return 1

    env_note = f"python = {PY}\n工作目录 = {ROOT}"
    print(env_note)
    print("=" * 72)

    log_event({
        "event": "run_start",
        "python": sys.version.split()[0],
        "executable": PY,
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "steps": [{"tag": t, "script": s, "desc": d} for t, s, d in steps],
        "argv": sys.argv[1:],
    })

    t_all = time.time()
    batch_id = datetime.now().strftime("resume-%Y%m%d-%H%M%S")
    def fingerprints():
        found = {}
        for folder in (HERE, ROOT / "求解结果" / "数据", ROOT / "求解结果" / "图表"):
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix not in (".pyc",):
                    stat = p.stat()
                    found[p.relative_to(ROOT).as_posix()] = {
                        "path": p.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
        return found
    results = []
    output_owners = {}
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    for tag, script, desc in steps:
        path = HERE / script
        if not path.exists():
            print(f"[{tag}] 缺失 {script}，跳过")
            results.append((tag, script, "MISSING", 0.0))
            continue
        print(f"\n[{tag}] {desc}")
        print(f"      {script}")
        before = fingerprints()
        started_at = datetime.now().astimezone().isoformat()
        t0 = time.time()
        r = subprocess.run([PY, str(path)], cwd=str(HERE),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=child_env)
        dt = time.time() - t0
        ok = r.returncode == 0
        run_dir = ROOT / "求解结果" / "运行记录" / f"{batch_id}-{tag}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "stdout.txt").write_text(r.stdout or "", encoding="utf-8")
        (run_dir / "stderr.txt").write_text(r.stderr or "", encoding="utf-8")
        after = fingerprints()
        outputs = [v for k, v in after.items() if before.get(k) != v and k.startswith("求解结果/")]
        for record in outputs:
            output_owners.setdefault(record["path"], []).append(tag)
        manifest = {
            "schema_version": 2, "run_id": run_dir.name, "return_code": r.returncode,
            "started_at": started_at, "ended_at": datetime.now().astimezone().isoformat(),
            "command": [PY, str(path)], "seed": 20260826, "python": PY,
            "script": path.relative_to(ROOT).as_posix(), "script_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "inputs": list(before.values()),
            "raw_archive_manifest": str(ROOT / "临时产物" / "原始附件清单.json"),
            "outputs": outputs, "duration_seconds": round(dt, 3),
            "goal": desc, "baseline": "B1/B2 inherited methods plus explicit repaired theory",
            "success_criterion": "process success; scientific assertions inside steps; review still required",
            "failure_interpretation": "failed steps block downstream execution; no failure is a passed result",
            "stdout": str((run_dir / "stdout.txt").relative_to(ROOT)),
            "stderr": str((run_dir / "stderr.txt").relative_to(ROOT))}
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        for line in (r.stdout or "").splitlines()[-8:]:
            print(f"      {line}")
        if not ok:
            print(f"      失败（exit={r.returncode}）")
            for line in (r.stderr or "").splitlines()[-25:]:
                print(f"      {line}")
        print(f"      {'完成' if ok else '失败'}，耗时 {dt:.1f} s")
        results.append((tag, script, "OK" if ok else "FAIL", dt))
        log_event({
            "event": "step_end",
            "tag": tag,
            "script": script,
            "desc": desc,
            "return_code": r.returncode,
            "seconds": round(dt, 1),
            "status": "OK" if ok else "FAIL",
            "stdout_tail": (r.stdout or "").splitlines()[-5:],
            "stderr_tail": (r.stderr or "").splitlines()[-5:] if not ok else [],
        })
        if not ok:
            break

    print("\n" + "=" * 72)
    print(f"{'步骤':<5}{'状态':<8}{'耗时/s':>9}  脚本")
    for tag, script, st, dt in results:
        print(f"{tag:<5}{st:<8}{dt:>9.1f}  {script}")
    print(f"\n总耗时 {time.time() - t_all:.1f} s")

    bad = [r for r in results if r[2] != "OK"]
    duplicate_owners = {
        path: owners for path, owners in output_owners.items() if len(owners) > 1
    }
    if duplicate_owners:
        print("\n输出所有权冲突（同一求解结果被多个步骤写入）：")
        for path, owners in sorted(duplicate_owners.items()):
            print(f"  {path}: {', '.join(owners)}")
    # Bind the settled result tree only after downstream steps have finished.
    if not bad and not duplicate_owners:
        records = fingerprints()
        aggregate = ROOT / "求解结果" / "运行记录" / f"{batch_id}-aggregate"
        aggregate.mkdir()
        source_records = [v for k, v in records.items() if k.startswith("支撑材料/")]
        result_records = [v for k, v in records.items() if k.startswith("求解结果/")]
        production_prefixes = ("求解结果/数据/", "求解结果/图表/")
        production_outputs = {
            path for path in records
            if path.startswith(production_prefixes)
            and not path.startswith("求解结果/数据/cache/")
        }
        internal_cache_outputs = {
            path for path in records
            if path.startswith("求解结果/数据/cache/")
        }
        unowned_outputs = sorted(production_outputs - set(output_owners))
        untracked_internal_cache = sorted(internal_cache_outputs - set(output_owners))
        owner_check_ok = not unowned_outputs
        raw_manifest = ROOT / "临时产物" / "原始附件清单.json"
        source_records.append({"path": raw_manifest.relative_to(ROOT).as_posix(),
                               "sha256": hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
                               "bytes": raw_manifest.stat().st_size})
        receipt = {"run_id": aggregate.name, "return_code": 0, "seed": 20260826,
                   "command": [PY, str(Path(__file__).resolve()), *sys.argv[1:]],
                   "inputs": source_records, "outputs": result_records,
                   "step_runs": [f"{batch_id}-{t}" for t, _, _, _ in results],
                   "goal": "settled complete pipeline result binding",
                   "all_steps_ok": True,
                   "output_owner_check": {
                        "status": "pass" if owner_check_ok else "fail",
                        "tracked_outputs": len(production_outputs),
                        "production_outputs": len(production_outputs),
                        "unowned_outputs": unowned_outputs,
                       "duplicate_owners": {},
                        "all_result_output_records": len(result_records),
                        "internal_cache": {
                            "count": len(internal_cache_outputs),
                            "untracked_reused_count": len(untracked_internal_cache),
                            "scope": (
                                "internal deterministic caches excluded from deliverable ownership; "
                                "their current hashes remain bound in outputs"
                            ),
                        },
                   }}
        (aggregate / "run_manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        if not owner_check_ok:
            bad.append(("aggregate", "run_all.py", "FAIL", 0.0))
            print("\n输出归属覆盖不完整：")
            for path in unowned_outputs:
                print(f"  {path}")
    log_event({
        "event": "run_end",
        "total_seconds": round(time.time() - t_all, 1),
        "steps": [{"tag": t, "script": s, "status": st, "seconds": round(dt, 1)}
                  for t, s, st, dt in results],
        "all_ok": not bad,
    })
    return 0 if not bad and not duplicate_owners else 1


if __name__ == "__main__":
    raise SystemExit(main())
