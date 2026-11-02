import fitz
from PIL import Image

pdf = r"D:\Users\14709\Desktop\建模\mathor cup\B增强\v6.0\建模论文\main.pdf"
outdir = r"D:\Users\14709\Desktop\建模\mathor cup\B增强\v6.0\临时产物"
doc = fitz.open(pdf)


def quad(pages, name):
    ims = []
    for n in pages:
        pix = doc[n - 1].get_pixmap(dpi=110)
        ims.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    w = max(i.width for i in ims)
    h = max(i.height for i in ims)
    canvas = Image.new("RGB", (w * 2, h * 2), "white")
    for k, im in enumerate(ims):
        canvas.paste(im, ((k % 2) * w, (k // 2) * h))
    path = outdir + "\\" + name
    canvas.save(path)
    print(path, canvas.size)


quad([5, 18, 20, 25], "quad_tables_a.png")
quad([27, 29, 33, 34], "quad_tables_b.png")
