#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenVINO Converter GUI (single-file, Tkinter)

- Supports:
  - PyTorch (.pth/.pt) regression models (Donkey-style) via model_catalog.get_model/load_model_weights
  - YOLO (.pt) via Ultralytics export(format="openvino")
  - ONNX (.onnx) via OpenVINO Model Optimizer

- Precisions:
  - FP32: default IR
  - FP16: IR compressed to FP16 (MO compress_to_fp16)
  - INT8: Post-training quantization using NNCF (requires calibration images folder)

Notes:
- INT8 is NOT a "simple MO flag". It needs calibration data.
- Run this GUI on your PC. (Quantization especially.)
"""

import os
import sys
import time
import traceback
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple, List

# -----------------------------
# Logging helper (thread-safe)
# -----------------------------
class UILogger:
    def __init__(self, q: "queue.Queue[str]"):
        self.q = q

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.q.put(f"[{ts}] {msg}\n")

# -----------------------------
# Conversion core
# -----------------------------
@dataclass
class ConvertOptions:
    model_path: str
    output_dir: str
    precision: str  # "FP32"|"FP16"|"INT8"
    # for regression models
    model_type: Optional[str] = None
    input_hw: Tuple[int, int] = (224, 224)  # (H, W)
    dynamic_batch: bool = False
    # for YOLO
    yolo_input_size: int = 640
    # for INT8
    calib_dir: Optional[str] = None
    calib_max_images: int = 300  # keep it reasonable

def _require(pkg: str, logger: UILogger, pip_hint: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:
        logger.log(f"ERROR: '{pkg}' が見つかりません。")
        logger.log(f"  インストール例: {pip_hint}")
        return False

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _path_noext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def _safe_join(dirpath: str, name: str) -> str:
    return os.path.join(dirpath, name)

def convert_onnx_to_openvino(onnx_path: str, output_stem: str, precision: str, logger: UILogger) -> Tuple[str, str]:
    """
    onnx_path -> output_stem.xml/.bin
    """
    if not _require("openvino", logger, "pip install openvino-dev"):
        raise RuntimeError("openvino missing")
    from openvino.tools import mo
    from openvino.runtime import serialize

    want_fp16 = (precision.upper() == "FP16")
    if precision.upper() == "INT8":
        # INT8 is handled AFTER we have an FP32/FP16 IR
        logger.log("INT8選択: まずFP32 IRを作成してから量子化します。")
        want_fp16 = False

    logger.log(f"MO変換: {onnx_path} -> {output_stem}.xml/.bin (compress_to_fp16={want_fp16})")
    ov_model = mo.convert_model(onnx_path, compress_to_fp16=want_fp16)
    xml_path = f"{output_stem}.xml"
    bin_path = f"{output_stem}.bin"
    serialize(ov_model, xml_path, bin_path)
    logger.log(f"出力: {xml_path}")
    logger.log(f"出力: {bin_path}")
    return xml_path, bin_path

def convert_pytorch_regression_to_openvino(
    model_path: str,
    model_type: str,
    output_stem: str,
    input_hw: Tuple[int, int],
    dynamic_batch: bool,
    precision: str,
    logger: UILogger
) -> Tuple[str, str]:
    """
    PyTorch (state_dict/torchscript) regression -> ONNX -> OpenVINO IR
    Requires model_catalog.py (get_model/load_model_weights) to exist in PYTHONPATH.
    """
    if not _require("torch", logger, "pip install torch"):
        raise RuntimeError("torch missing")
    import torch

    # Try to load Donkey-style model via model_catalog
    try:
        from model_catalog import get_model, load_model_weights
    except Exception as e:
        logger.log("ERROR: model_catalog.py を import できません。")
        logger.log("  これは Donkey系の学習コード構成（model_catalog.py がある環境）前提です。")
        logger.log(f"  import error: {e}")
        raise

    if not _require("openvino", logger, "pip install openvino-dev"):
        raise RuntimeError("openvino missing")

    H, W = int(input_hw[0]), int(input_hw[1])

    device = torch.device("cpu")
    logger.log(f"PyTorchモデルを構築: model_type={model_type}, input_size={(H,W)}")
    model = get_model(model_type, pretrained=False, input_size=(H, W))
    model = load_model_weights(model, model_path, device)
    model = model.to(device)
    model.eval()

    # ONNX export
    temp_onnx = output_stem + "_temp.onnx"
    dummy = torch.randn(1, 3, H, W, device=device)

    dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None
    logger.log(f"ONNXエクスポート: {temp_onnx} (dynamic_batch={dynamic_batch})")
    torch.onnx.export(
        model, dummy, temp_onnx,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    logger.log("ONNX作成OK")

    # ONNX -> OpenVINO
    xml_path, bin_path = convert_onnx_to_openvino(temp_onnx, output_stem, precision, logger)

    # cleanup temp onnx
    try:
        os.remove(temp_onnx)
        logger.log("一時ONNXを削除しました")
    except Exception:
        pass

    return xml_path, bin_path

def convert_yolo_to_openvino_ultralytics(
    model_path: str,
    output_dir: str,
    input_size: int,
    precision: str,
    logger: UILogger
) -> str:
    """
    YOLO (.pt) -> Ultralytics export(format='openvino')
    Returns output directory produced by Ultralytics.
    """
    if not _require("ultralytics", logger, "pip install ultralytics"):
        raise RuntimeError("ultralytics missing")

    from ultralytics import YOLO

    half = (precision.upper() == "FP16")
    int8 = (precision.upper() == "INT8")

    logger.log(f"YOLO export: half={half}, int8={int8}, imgsz={input_size}")
    model = YOLO(model_path)

    # Ultralytics chooses output folder under runs by default.
    # We'll set project/name to put it inside output_dir.
    _ensure_dir(output_dir)
    name = _path_noext(model_path) + f"_openvino_{precision.lower()}"
    res = model.export(
        format="openvino",
        half=half,
        int8=int8,
        imgsz=input_size,
        project=output_dir,
        name=name
    )
    logger.log(f"YOLO OpenVINO出力: {res}")
    return str(res)

def _collect_images(calib_dir: str, max_images: int) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    for root, _, files in os.walk(calib_dir):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths[:max_images]

def quantize_openvino_int8_nncf(fp32_xml_path, out_xml_path, calib_dir, logger, max_images=500):
    def _log(msg):
    # loggerが関数ならそのまま呼ぶ
        if callable(logger):
            logger(msg)
            return
        # よくあるメソッド名を順番に試す
        for name in ("info", "log", "write", "append", "add"):
            if hasattr(logger, name):
                getattr(logger, name)(msg)
                return
        # 最後の手段
        print(msg)

    try:
        import nncf
        from nncf import Dataset
    except Exception:
        raise RuntimeError("nncf missing")

    import os
    import glob
    import cv2
    import numpy as np
    from openvino.runtime import Core

    core = Core()
    model = core.read_model(fp32_xml_path)

    # 入力名を取得（1入力前提）
    input_port = model.inputs[0]
    input_name = input_port.get_any_name()
    
    shape = input_port.partial_shape

    # 入力shape（例: [1,3,224,224] or [-1,3,224,224]）
    def dim_to_int(dim, default):
        return dim.get_length() if dim.is_static else default
    # H,W を取りたいので [N,C,H,W] 前提
    # dynamicの場合があるので 224x224 を優先しつつ安全に取る
    C = dim_to_int(shape[1], 3)
    H = dim_to_int(shape[2], 224)
    W = dim_to_int(shape[3], 224)

    # 校正画像を集める
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for e in exts:
        image_paths += glob.glob(os.path.join(calib_dir, e))
    image_paths = sorted(image_paths)[:max_images]

    if len(image_paths) == 0:
        raise RuntimeError(f"no calibration images in: {calib_dir}")

    _log(f"校正画像: {len(image_paths)} 枚  input={input_name} size={H}x{W}")

    def transform_fn(img_path: str):
        img = cv2.imread(img_path)
        if img is None:
            # 壊れてる画像は適当にスキップ用の黒画像
            img = np.zeros((H, W, 3), dtype=np.uint8)

        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 0-1 正規化（学習側が mean/std 使ってるなら合わせる）
        x = img.astype(np.float32) / 255.0

        # HWC -> CHW
        x = np.transpose(x, (2, 0, 1))

        # batch次元追加: 1,C,H,W
        x = np.expand_dims(x, axis=0)

        # NNCFは dict で入力を返すのが安定
        return {input_name: x}

    calibration_dataset = Dataset(image_paths, transform_fn)

    # 量子化
    # subset_size は NNCFが内部でサンプル数制御するのに使う（無くても動くが入れた方が安全）
    q_model = nncf.quantize(
        model,
        calibration_dataset=calibration_dataset,
        subset_size=min(len(image_paths), 300),  # まず300くらいで十分なことが多い
    )

    # 保存（out_xml_path の .xml/.bin を作る）
    # openvino.runtime.serialize が使えます
    from openvino.runtime import serialize
    from pathlib import Path
    from openvino.runtime import serialize

    out_xml_path = Path(out_xml_path)
    out_xml_path.parent.mkdir(parents=True, exist_ok=True)

    out_bin_path = out_xml_path.with_suffix(".bin")
    if out_bin_path.exists():
        bak = out_bin_path.with_suffix(".bin.bak")
        try:
            out_bin_path.rename(bak)
        except Exception:
            pass  # ロックされてたら諦める
    
    #if out_bin_path.exists():
        #out_bin_path.unlink()
    
    serialize(q_model, str(out_xml_path))

    logger.log(f"INT8出力: {out_xml_path}")

    #def dataset():
    #    for p in img_paths:
    #        img = cv2.imread(p, cv2.IMREAD_COLOR)
    #        if img is None:
    #            continue
    #        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #        if img.shape[0] != H or img.shape[1] != W:
    #            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            # float32 0..1
    #        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    #        yield {inp.any_name: x}

    #logger.log("NNCF quantize 実行中（しばらく待つやつ）")
    #q_model = nncf.quantize(
    #    model,
    #   dataset(),
    #    preset=nncf.QuantizationPreset.MIXED
    #)

    # Save quantized model
    #ov.save_model(q_model, out_xml)
    #logger.log(f"INT8出力: {out_xml} (+ .bin)")
    #return out_xml

def convert_dispatch(opts: ConvertOptions, logger: UILogger):
    model_path = os.path.abspath(opts.model_path)
    out_dir = os.path.abspath(opts.output_dir)
    precision = opts.precision.upper()

    _ensure_dir(out_dir)

    ext = os.path.splitext(model_path)[1].lower()
    base_stem = _safe_join(out_dir, _path_noext(model_path))

    # INT8のときは、FP32中間IRは別名で作る（衝突回避）
    if precision == "INT8":
        stem = base_stem + "_fp32_tmp"
    else:
        stem = base_stem + f"_{precision.lower()}"

    # Decide route
    is_yolo_hint = ("yolo" in os.path.basename(model_path).lower()) or (opts.model_type and "yolo" in opts.model_type.lower())
    if ext == ".onnx":
        # INT8のときは、まずFP32 IRを作る（ファイル名は _fp32_tmp ）
        export_precision = "FP32" if precision == "INT8" else precision

        xml, _ = convert_pytorch_regression_to_openvino(
            model_path=model_path,
            model_type=opts.model_type,
            output_stem=stem,
            input_hw=opts.input_hw,
            dynamic_batch=opts.dynamic_batch,
            precision=export_precision,
            logger=logger
        )

        if precision == "INT8":
            if not opts.calib_dir:
                raise RuntimeError("INT8には校正画像フォルダが必要です")
            fp_xml = xml  # これは _fp32_tmp.xml のはず
            out_xml = _safe_join(out_dir, _path_noext(model_path) + "_int8.xml")
            quantize_openvino_int8_nncf(fp_xml, out_xml, opts.calib_dir, logger, max_images=opts.calib_max_images)
    
    elif ext in (".pt", ".pth"):
        if is_yolo_hint:
            # YOLO route
            convert_yolo_to_openvino_ultralytics(
                model_path=model_path,
                output_dir=out_dir,
                input_size=int(opts.yolo_input_size),
                precision=precision,
                logger=logger
            )
        else:
            if not opts.model_type:
                raise RuntimeError("回帰PyTorchモデルには model_type が必要です（例: donkeycar / edgenext など）")
            xml, _ = convert_pytorch_regression_to_openvino(
                model_path=model_path,
                model_type=opts.model_type,
                output_stem=stem,
                input_hw=opts.input_hw,
                dynamic_batch=opts.dynamic_batch,
                precision=precision,
                logger=logger
            )
            if precision == "INT8":
                if not opts.calib_dir:
                    raise RuntimeError("INT8には校正画像フォルダが必要です")
                fp_xml = xml  # created FP32
                out_xml = _safe_join(out_dir, _path_noext(model_path) + "_int8.xml")
                quantize_openvino_int8_nncf(fp_xml, out_xml, opts.calib_dir, logger, max_images=opts.calib_max_images)
    else:
        raise RuntimeError(f"未対応の拡張子: {ext}")

# -----------------------------
# Tkinter GUI
# -----------------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PyTorch/ONNX → OpenVINO Converter (FP32/FP16/INT8)")
        self.geometry("920x620")

        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.logger = UILogger(self.log_q)
        self.worker: Optional[threading.Thread] = None

        # Vars
        self.var_model_path = tk.StringVar()
        self.var_out_dir = tk.StringVar(value=os.getcwd())
        self.var_precision = tk.StringVar(value="FP16")
        self.var_model_type = tk.StringVar()
        self.var_h = tk.IntVar(value=224)
        self.var_w = tk.IntVar(value=224)
        self.var_dynamic = tk.BooleanVar(value=False)
        self.var_yolo_imgsz = tk.IntVar(value=640)
        self.var_calib_dir = tk.StringVar()
        self.var_calib_max = tk.IntVar(value=300)

        self._build()
        self.after(80, self._drain_log)

    def _build(self):
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        # Model path
        row = ttk.Frame(frm)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="モデルファイル (.pth/.pt/.onnx)").pack(side="left")
        ttk.Entry(row, textvariable=self.var_model_path, width=72).pack(side="left", padx=8)
        ttk.Button(row, text="Browse", command=self._browse_model).pack(side="left")

        # Output dir
        row = ttk.Frame(frm)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="出力フォルダ").pack(side="left")
        ttk.Entry(row, textvariable=self.var_out_dir, width=72).pack(side="left", padx=8)
        ttk.Button(row, text="Browse", command=self._browse_outdir).pack(side="left")

        # Precision
        row = ttk.Frame(frm)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Precision").pack(side="left")
        for p in ("FP32", "FP16", "INT8"):
            ttk.Radiobutton(row, text=p, value=p, variable=self.var_precision, command=self._on_precision).pack(side="left", padx=10)

        # Regression options
        box = ttk.LabelFrame(frm, text="回帰モデル（Donkey系）オプション")
        box.pack(fill="x", **pad)

        row = ttk.Frame(box)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="model_type (例: donkeycar / edgenext_xx_small …)").pack(side="left")
        ttk.Entry(row, textvariable=self.var_model_type, width=40).pack(side="left", padx=8)
        ttk.Checkbutton(row, text="dynamic batch", variable=self.var_dynamic).pack(side="left", padx=10)

        row = ttk.Frame(box)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="入力サイズ (H×W)").pack(side="left")
        ttk.Spinbox(row, from_=32, to=1024, textvariable=self.var_h, width=6).pack(side="left", padx=6)
        ttk.Label(row, text="×").pack(side="left")
        ttk.Spinbox(row, from_=32, to=1024, textvariable=self.var_w, width=6).pack(side="left", padx=6)

        # YOLO options
        box2 = ttk.LabelFrame(frm, text="YOLOオプション（Ultralytics）")
        box2.pack(fill="x", **pad)
        row = ttk.Frame(box2)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="imgsz").pack(side="left")
        ttk.Spinbox(row, from_=128, to=2048, increment=32, textvariable=self.var_yolo_imgsz, width=8).pack(side="left", padx=6)
        ttk.Label(row, text="※ファイル名に yolo が入ってるか、model_type に yolo を含めるとYOLO扱い").pack(side="left", padx=10)

        # INT8 calibration
        box3 = ttk.LabelFrame(frm, text="INT8校正（NNCF, 画像フォルダ）")
        box3.pack(fill="x", **pad)

        row = ttk.Frame(box3)
        row.pack(fill="x", **pad)
        self.lbl_calib = ttk.Label(row, text="校正画像フォルダ")
        self.lbl_calib.pack(side="left")
        self.ent_calib = ttk.Entry(row, textvariable=self.var_calib_dir, width=64)
        self.ent_calib.pack(side="left", padx=8)
        self.btn_calib = ttk.Button(row, text="Browse", command=self._browse_calibdir)
        self.btn_calib.pack(side="left")

        row = ttk.Frame(box3)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="最大使用枚数").pack(side="left")
        self.spin_calib = ttk.Spinbox(row, from_=20, to=5000, increment=20, textvariable=self.var_calib_max, width=8)
        self.spin_calib.pack(side="left", padx=6)
        ttk.Label(row, text="（多いほど安定しやすいが時間が伸びる）").pack(side="left", padx=10)

        # Buttons
        row = ttk.Frame(frm)
        row.pack(fill="x", **pad)
        self.btn_run = ttk.Button(row, text="変換開始", command=self._run)
        self.btn_run.pack(side="left")
        ttk.Button(row, text="ログをクリア", command=self._clear_log).pack(side="left", padx=10)
        ttk.Button(row, text="終了", command=self.destroy).pack(side="right")

        # Log window
        self.txt = tk.Text(self, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)
        self._on_precision()  # init

    def _browse_model(self):
        f = filedialog.askopenfilename(filetypes=[
            ("Models", "*.pth *.pt *.onnx"),
            ("All", "*.*"),
        ])
        if f:
            self.var_model_path.set(f)

    def _browse_outdir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_out_dir.set(d)

    def _browse_calibdir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_calib_dir.set(d)

    def _clear_log(self):
        self.txt.delete("1.0", "end")

    def _on_precision(self):
        is_int8 = (self.var_precision.get().upper() == "INT8")
        state = "normal" if is_int8 else "disabled"
        for w in (self.ent_calib, self.btn_calib, self.spin_calib):
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _drain_log(self):
        try:
            while True:
                s = self.log_q.get_nowait()
                self.txt.insert("end", s)
                self.txt.see("end")
        except queue.Empty:
            pass
        self.after(80, self._drain_log)

    def _set_running(self, running: bool):
        self.btn_run.configure(state=("disabled" if running else "normal"))

    def _run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Running", "すでに変換中です。")
            return

        model_path = self.var_model_path.get().strip()
        out_dir = self.var_out_dir.get().strip()
        if not model_path:
            messagebox.showerror("Error", "モデルファイルを指定してください。")
            return
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "モデルファイルが見つかりません。")
            return
        if not out_dir:
            messagebox.showerror("Error", "出力フォルダを指定してください。")
            return

        prec = self.var_precision.get().upper()
        calib_dir = self.var_calib_dir.get().strip() if prec == "INT8" else None
        if prec == "INT8":
            if not calib_dir or not os.path.isdir(calib_dir):
                messagebox.showerror("Error", "INT8には校正画像フォルダが必要です。")
                return

        opts = ConvertOptions(
            model_path=model_path,
            output_dir=out_dir,
            precision=prec,
            model_type=self.var_model_type.get().strip() or None,
            input_hw=(int(self.var_h.get()), int(self.var_w.get())),
            dynamic_batch=bool(self.var_dynamic.get()),
            yolo_input_size=int(self.var_yolo_imgsz.get()),
            calib_dir=calib_dir,
            calib_max_images=int(self.var_calib_max.get()),
        )

        self._set_running(True)
        self.logger.log("========================================")
        self.logger.log("変換開始")
        self.logger.log(f"model={opts.model_path}")
        self.logger.log(f"out_dir={opts.output_dir}")
        self.logger.log(f"precision={opts.precision}")
        if opts.model_type:
            self.logger.log(f"model_type={opts.model_type}")
        if opts.precision == "INT8":
            self.logger.log(f"calib_dir={opts.calib_dir} (max={opts.calib_max_images})")

        def job():
            try:
                convert_dispatch(opts, self.logger)
                self.logger.log("✅ 完了")
            except Exception as e:
                self.logger.log("❌ 失敗: " + str(e))
                self.logger.log(traceback.format_exc())
            finally:
                self.after(0, lambda: self._set_running(False))

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
