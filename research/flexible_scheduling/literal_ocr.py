"""Read literal equipment identifiers from camera bytes using the fixed local OCR engine."""
import csv
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image, ImageOps

INSTALL = Path(__file__).resolve().parents[2] / "outputs/continuing/research-tools/ocr/root"
TEXT = re.compile(r"(?<![A-Z0-9])([A-Z]{1,6}-[0-9]{2,6})(?![A-Z0-9])")
SOURCE = re.compile(r"\bASSET\s*ID\s*[:#]?\s*([A-Z]{1,6}-[0-9]{2,6})\b")

ENGINE_SHA256 = "e9fd1390c8e4b50fc1abab2dddb41c96da4876c0a6149657e9cd524d34f35ed8"
LANGUAGE_SHA256 = "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"


@lru_cache(maxsize=1)
def engine_verified():
    files = ((INSTALL/"usr/bin/tesseract", ENGINE_SHA256),
             (INSTALL/"usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata", LANGUAGE_SHA256))
    return all(p.is_file() and sha256(p.read_bytes()).hexdigest() == expected for p, expected in files)


def decode_rgb(rgb, bbox=(), single_line_plaque=False, *, timeout_s=15):
    binary=INSTALL/"usr/bin/tesseract"
    image=Image.open(BytesIO(rgb)).convert("L")
    w,h=image.size
    left,top,right,bottom=bbox or (0.,0.,1.,1.)
    box=(max(0,int(left*w)),max(0,int(top*h)),min(w,int(np.ceil(right*w))),min(h,int(np.ceil(bottom*h))))
    if box[2]<=box[0] or box[3]<=box[1]:
        return []
    region=image.crop(box)
    scale=4 if bbox else 2
    region=ImageOps.expand(ImageOps.autocontrast(region.resize((region.width*scale,region.height*scale),
                                                              Image.Resampling.LANCZOS if hasattr(Image,"Resampling") else Image.LANCZOS)),8,fill=255)
    stream=BytesIO()
    region.save(stream,format="PNG")
    cmd=[str(binary),"stdin","stdout","--tessdata-dir",str(INSTALL/"usr/share/tesseract-ocr/4.00/tessdata"),
         "-l","eng","--psm","7" if bbox or single_line_plaque else "11","-c","tessedit_create_tsv=1"]
    p=subprocess.run(cmd,input=stream.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     env={**os.environ,"OMP_THREAD_LIMIT":"1"},timeout=timeout_s,check=True)
    lines=defaultdict(list)
    reader=csv.DictReader(StringIO(p.stdout.decode("utf8")),delimiter="\t")
    if not {"level","text","conf","line_num"} <= set(reader.fieldnames or ()):
        raise ValueError("OCR did not emit word-level TSV")
    for r in reader:
        if r.get("level")=="5" and (r.get("text") or "").strip():
            lines[(r["block_num"],r["par_num"],r["line_num"])].append(r)
    detections=[]
    for words in lines.values():
        line=" ".join(r["text"] for r in words).upper()
        pattern=TEXT if bbox else SOURCE
        for m in pattern.finditer(line):
            conf=min(float(r["conf"]) for r in words)
            if conf<75:
                continue
            x0=min(int(r["left"]) for r in words); x1=max(int(r["left"])+int(r["width"]) for r in words)
            y0=min(int(r["top"]) for r in words); y1=max(int(r["top"])+int(r["height"]) for r in words)
            detections.append(dict(value=m[1],confidence=conf,
                center=[(box[0]+((x0+x1)/2-8)/scale)/w,(box[1]+((y0+y1)/2-8)/scale)/h]))
    return detections
