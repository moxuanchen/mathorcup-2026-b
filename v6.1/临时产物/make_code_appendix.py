import os
import shutil

root = r"D:\Users\14709\Desktop\建模\mathor cup\B增强\v6.1"
src = os.path.join(root, "支撑材料", "源码")
copy_dir = os.path.join(root, "建模论文", "代码清单")
out = os.path.join(root, "建模论文", "appendix_code.tex")

groups = [
    ("主流程", [
        ("run_all.py", "全流程入口、步骤顺序与断点续跑"),
        ("prepare_inputs.py", "附件清点、解压与输入准备"),
    ]),
    ("公共库", [
        ("mmlib/__init__.py", "公共库入口"),
        ("mmlib/finite_inventory.py", "有限库存团块动力"),
        ("mmlib/geodesy.py", "坐标与距离换算"),
        ("mmlib/glacier_masks.py", "冰川编目掩膜与面积交比"),
        ("mmlib/hydro.py", "水文过程与SCS-CN径流"),
        ("mmlib/interval_metrics.py", "区间与覆盖指标"),
        ("mmlib/io_data.py", "数据读写与校验记录"),
        ("mmlib/meteo.py", "气象产品读取与时间对齐"),
        ("mmlib/ranking.py", "稳健异常与关联排序"),
        ("mmlib/terrain.py", "地形、D8路由与断面"),
        ("mmlib/terrain_prep.py", "DEM填洼与网格准备"),
        ("mmlib/trigger.py", "床面力平衡与触发判别"),
    ]),
    ("问题一", [
        ("step01_boundary_layer.py", "边界层结构与静力订正"),
        ("step01b_preimpact.py", "冲击前窗口异常"),
        ("step01c_unified_association.py", "31项统一候选关联与去冗余"),
    ]),
    ("问题二", [
        ("step02_trigger_threshold.py", "沟道临界条件与机械状态"),
        ("step02b_spatial.py", "空间像元对照"),
        ("step02c_spatial_matched.py", "共同域空间隔离"),
        ("step02_figures.py", "问题二图件"),
    ]),
    ("问题三", [
        ("step03_forecast.py", "3000成员前向预报"),
        ("step03_time_support_sensitivity.py", "时间支持敏感性"),
        ("step03_figures.py", "问题三图件"),
    ]),
    ("问题四", [
        ("step04_simulation.py", "一维纵剖面仿真"),
        ("step04b_sim2d.py", "D8网络路由与质量收支"),
        ("step04b_map.py", "到达时间与停积制图"),
        ("step04c_dynamics.py", "有限库存与分段动力"),
        ("step04_figures.py", "问题四图件"),
        ("step04b_figures.py", "D8图件"),
    ]),
    ("图表生成", [
        ("step05_figures.py", "论文图件生成"),
    ]),
    ("数据整理与校验", [
        ("step00_data_inventory.py", "附件清单与元数据"),
        ("step06_data_audit.py", "数据可用性核查"),
        ("archive_external.py", "外部响应归档"),
        ("select_paper_values.py", "论文取值汇总"),
        ("validate_v50.py", "确定性结果验收"),
        ("export_summary.py", "汇总结果导出"),
    ]),
    ("测试", [
        ("test01b_hydrostatic.py", "静力层压测试"),
        ("test01b_preimpact.py", "冲击前窗口测试"),
        ("test02c03_protocols.py", "空间与预报协议测试"),
        ("test04c_dynamics.py", "有限库存动力测试"),
        ("test_scs_event_runoff.py", "SCS-CN事件径流测试"),
        ("test_time_support.py", "时间支持测试"),
        ("test_v32_data_fixes.py", "数据修复回归测试"),
    ]),
]

REPL = {"℃": "°C", "⁄": "/", "∥": "||"}

if os.path.isdir(copy_dir):
    shutil.rmtree(copy_dir)
os.makedirs(copy_dir)


def esc(s):
    return s.replace("_", r"\_")


lines = []
count = 0
for title, files in groups:
    lines.append(r"\subsection*{%s}" % title)
    for rel, desc in files:
        p = os.path.join(src, rel.replace("/", os.sep))
        assert os.path.exists(p), p
        text = open(p, encoding="utf-8").read()
        for a, b in REPL.items():
            text = text.replace(a, b)
        dst = os.path.join(copy_dir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="utf-8", newline="").write(text)
        count += 1
        lines.append(r"\noindent\textbf{\texttt{%s}}\quad %s\par\nopagebreak" % (esc(rel), desc))
        lines.append(r"\lstinputlisting[language=Python]{代码清单/%s}" % rel)
    lines.append("")

text = "\n".join(lines)
open(out, "w", encoding="utf-8").write(text)
print("files", count, "chars", len(text))
