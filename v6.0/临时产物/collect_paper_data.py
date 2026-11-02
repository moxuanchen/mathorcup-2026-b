import json
import os
import shutil

root = r"D:\Users\14709\Desktop\建模\mathor cup\B增强\v6.0"
src_dir = os.path.join(root, "求解结果", "数据")
dst_dir = os.path.join(root, "支撑材料", "数据")
os.makedirs(dst_dir, exist_ok=True)

prov = json.load(open(os.path.join(src_dir, "05_value_provenance.json"), encoding="utf-8"))
files = []
for s in prov["sources"]:
    rel = s["source_file"].replace("求解结果/数据/", "")
    files.append(rel)
files += ["05_paper_values.json", "05_value_provenance.json"]

missing = []
copied = 0
total = 0
for rel in sorted(set(files)):
    src = os.path.join(src_dir, rel.replace("/", os.sep))
    if not os.path.exists(src):
        missing.append(rel)
        continue
    dst = os.path.join(dst_dir, os.path.basename(rel))
    shutil.copy2(src, dst)
    copied += 1
    total += os.path.getsize(dst)

print("copied", copied)
print("missing", missing)
print("bytes", total, "MB", round(total / 1e6, 2))
print("files on disk", len(os.listdir(dst_dir)))
