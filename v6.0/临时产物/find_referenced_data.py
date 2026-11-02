import json
import os

src = r"D:\Users\14709\Desktop\建模\mathor cup\B增强\v6.0\求解结果\数据"
prov = json.load(open(os.path.join(src, "05_value_provenance.json"), encoding="utf-8"))

found = set()


def walk(x):
    if isinstance(x, dict):
        for v in x.values():
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
    elif isinstance(x, str) and "求解结果/数据/" in x:
        found.add(x)


walk(prov)
names = {os.path.basename(p) for p in found if os.path.exists(os.path.join(src, p.replace("求解结果/数据/", "").replace("/", os.sep)))}
sources = {os.path.basename(s["source_file"]) for s in prov["sources"]}
print("referenced existing", len(names))
print("not in sources:", sorted(names - sources))
