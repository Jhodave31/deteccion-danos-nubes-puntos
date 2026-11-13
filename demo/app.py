"""Demo: detección de daños estructurales en nubes de puntos.

Servidor web local (solo biblioteca estándar + el modelo entrenado). Abre el navegador en
http://localhost:8000, elige una nube del dataset o sube un .ply y el modelo pinta en 3D las
edificaciones dañadas, sin daño, vías y árboles.

Uso (desde la carpeta demo/):
    python app.py                        # usa el modelo indicado en demo/config.json
    python app.py --run ../runs/pn2      # otro modelo entrenado
"""
import argparse
import base64
import json
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from predictor import CLASS_COLORS, CLASS_NAMES, Predictor, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "3DAeroRelief"
STATIC = HERE / "static"
MAX_VIEW = 700_000  # puntos máximos enviados al visor 3D

predictor = None
lock = threading.Lock()
cache = {}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def examples():
    out = []
    if RAW.exists():
        for f in sorted(RAW.glob("Area_*/pp*.ply"), key=lambda p: (int(p.parent.name.split("_")[1]), int(p.stem[2:]))):
            area = f.parent.name
            role = {"Area_2": "prueba", "Area_8": "validación"}.get(area, "entrenamiento")
            out.append(dict(id=f"{area}/{f.stem}", label=f"{area.replace('Area_', 'Área ')} / {f.stem} ({role})",
                            test=area == "Area_2", mb=round(f.stat().st_size / 1e6, 1)))
    return out


DISK_CACHE = HERE / "cache"


def analyze(ply_path, key, tta, persist=False):
    """Analiza una nube. Las nubes del dataset se guardan en demo/cache/ para abrirlas al instante."""
    if key in cache:
        return cache[key]
    disk = DISK_CACHE / predictor.run.name / (key.replace("/", "_").replace("|", "_tta") + ".json")
    if persist and disk.exists():
        cache[key] = json.loads(disk.read_text())
        return cache[key]
    t0 = time.time()
    print(f"[demo] preprocesando {ply_path.name} ...", flush=True)
    c = predictor.load_cloud(ply_path)
    t1 = time.time()
    print(f"[demo] {c['n_raw']:,} -> {len(c['xyz']):,} puntos en {t1 - t0:.1f} s; ejecutando {predictor.name} ...", flush=True)
    pred = predictor.predict(c, tta=tta)
    t2 = time.time()
    print(f"[demo] listo en {t2 - t1:.1f} s", flush=True)
    label = c.get("label")
    summ = summarize(pred, label)
    idx = np.arange(len(pred))
    if len(idx) > MAX_VIEW:
        idx = np.sort(np.random.default_rng(0).choice(idx, MAX_VIEW, replace=False))
    xyz = c["xyz"][idx]
    xyz = xyz - np.array([(xyz[:, 0].min() + xyz[:, 0].max()) / 2, (xyz[:, 1].min() + xyz[:, 1].max()) / 2, xyz[:, 2].min()])
    title = f"{ply_path.parent.name.replace('Area_', 'Área ')} / {c['name']}" if ply_path.parent.name.startswith("Area_") else c["name"]
    res = dict(name=title, model=predictor.name, n_raw=int(c["n_raw"]), n_vox=int(len(pred)), n_view=int(len(idx)),
               t_pre=round(t1 - t0, 1), t_pred=round(t2 - t1, 1), tta=tta, summary=summ,
               xyz=b64(xyz.astype(np.float32)), rgb=b64(c["rgb"][idx].astype(np.uint8)),
               pred=b64(pred[idx].astype(np.uint8)),
               label=b64(label[idx].astype(np.uint8)) if label is not None else None)
    cache[key] = res
    if persist:
        disk.parent.mkdir(parents=True, exist_ok=True)
        disk.write_text(json.dumps(res))
    return res


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")  # el navegador siempre pide la versión actual de la página
        super().end_headers()

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/info":
            return self.send_json(dict(model=predictor.name, run=str(predictor.run.name), examples=examples(),
                                       classes=[dict(name=n, color=c) for n, c in zip(CLASS_NAMES, CLASS_COLORS)]))
        if u.path == "/api/analyze":
            q = parse_qs(u.query)
            ex = q.get("id", [""])[0]
            tta = q.get("tta", ["0"])[0] == "1"
            path = (RAW / (ex + ".ply")).resolve()
            if RAW.resolve() not in path.parents or not path.exists():
                return self.send_json(dict(error="nube no encontrada"), 404)
            try:
                with lock:
                    return self.send_json(analyze(path, f"{ex}|{int(tta)}", tta, persist=True))
            except Exception as e:
                traceback.print_exc()
                return self.send_json(dict(error=str(e)), 500)
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/snapshot":  # guarda una imagen del visor 3D (para el documento)
            n = int(self.headers.get("Content-Length", 0))
            (HERE / "snapshot.png").write_bytes(self.rfile.read(n))
            return self.send_json(dict(ok=True))
        if u.path != "/api/upload":
            return self.send_json(dict(error="ruta desconocida"), 404)
        q = parse_qs(u.query)
        tta = q.get("tta", ["0"])[0] == "1"
        n = int(self.headers.get("Content-Length", 0))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "subida.ply"
            with open(p, "wb") as f:
                remaining = n
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 8 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            try:
                with lock:
                    res = analyze(p, f"upload-{time.time()}", tta)
                res["name"] = q.get("name", ["nube subida"])[0]
                return self.send_json(res)
            except Exception as e:
                traceback.print_exc()
                return self.send_json(dict(error=f"no se pudo procesar el archivo: {e}"), 400)


def main():
    global predictor
    cfg = json.loads((HERE / "config.json").read_text()) if (HERE / "config.json").exists() else {}
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / cfg.get("run", "runs/pn2")))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    print("Cargando modelo desde", a.run, "...")
    predictor = Predictor(a.run, device="cpu" if a.cpu else None)
    print("Modelo:", predictor.name, "| dispositivo:", predictor.device)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    print("Demo lista en", url, "(Ctrl+C para cerrar)")
    if not a.no_browser:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
