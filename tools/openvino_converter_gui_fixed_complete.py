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
# Path fix for GUI execution
# -----------------------------
# Tkinter GUI では実行時のカレントディレクトリ(cwd)がズレやすく、
# 同じフォルダにある model_catalog.py を import できない事があります。
# このスクリプト自身の場所を sys.path に入れて解決します。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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


def _debug_import_paths(logger: Optional['UILogger']=None):
    """model_catalog import 失敗時に原因を特定しやすくするための情報をログ出しする。"""
    try:
        import importlib.util
        lines = []
        lines.append(f"__file__={__file__}")
        lines.append(f"SCRIPT_DIR={SCRIPT_DIR}")
        lines.append(f"cwd={os.getcwd()}")
        lines.append("sys.path (top 15):")
        for p in sys.path[:15]:
            lines.append(f"  {p}")
        spec = importlib.util.find_spec('model_catalog')
        lines.append(f"find_spec('model_catalog')={spec}")
        for s in lines:
            if logger:
                logger.log(s)
            else:
                print(s)
    except Exception:
        # デバッグログは落としても致命的ではない
        pass


def import_model_catalog(logger: Optional['UILogger']=None):
    """model_catalog を確実に import し、失敗時は traceback 付きでログを残す。"""
    try:
        import importlib
        mc = importlib.import_module('model_catalog')
        # どれを掴んだかも出す（同名別モジュール事故対策）
        try:
            if logger:
                logger.log(f"model_catalog loaded from: {getattr(mc, '__file__', 'unknown')}")
        except Exception:
            pass
        return mc
    except Exception as e:
        if logger:
            logger.log("ERROR: model_catalog.py を import できません。")
            logger.log(f"  import error: {e}")
            logger.log(traceback.format_exc())
            _debug_import_paths(logger)
        raise

def convert_onnx_to_openvino(onnx_path: str, output_stem: str, precision: str, logger: UILogger) -> Tuple[str, str]:
    """
    onnx_path -> output_stem.xml/.bin
    """
    if not _require("openvino", logger, "pip install openvino-dev"):
        raise RuntimeError("openvino missing")
    from openvino.tools import mo
    # OpenVINO 2024+ では serialize が非推奨、save_model を使用
    try:
        from openvino import save_model
    except ImportError:
        from openvino.runtime import serialize as _serialize
        def save_model(model, path, **kwargs):
            _serialize(model, path)

    want_fp16 = (precision.upper() == "FP16")
    if precision.upper() == "INT8":
        # INT8 is handled AFTER we have an FP32/FP16 IR
        logger.log("INT8選択: まずFP32 IRを作成してから量子化します。")
        want_fp16 = False

    logger.log(f"MO変換: {onnx_path} -> {output_stem}.xml/.bin (compress_to_fp16={want_fp16})")
    ov_model = mo.convert_model(onnx_path, compress_to_fp16=want_fp16)
    xml_path = f"{output_stem}.xml"
    bin_path = f"{output_stem}.bin"

    # save_model で保存を試行
    logger.log(f"save_model 開始: {xml_path}")
    save_model(ov_model, xml_path)

    # ファイル存在チェック — save_model がサイレントに失敗するケースへの対策
    import time as _time
    _time.sleep(0.5)  # ファイルシステムのフラッシュ待ち

    if not os.path.exists(xml_path):
        logger.log(f"WARNING: save_model後に {xml_path} が見つかりません。代替方法を試行...")
        # 代替方法1: openvino.runtime.serialize を直接使用
        try:
            from openvino.runtime import serialize as _fallback_serialize
            _fallback_serialize(ov_model, xml_path, bin_path)
            _time.sleep(0.5)
            logger.log(f"fallback serialize 完了")
        except Exception as e:
            logger.log(f"fallback serialize 失敗: {e}")

    if not os.path.exists(xml_path):
        logger.log(f"WARNING: まだ {xml_path} が見つかりません。OVModelSerializer を試行...")
        # 代替方法2: pass_manager 経由
        try:
            from openvino.runtime.passes import Manager
            pass_manager = Manager()
            pass_manager.run_passes(ov_model)
            ov_model.serialize(xml_path, bin_path)
            _time.sleep(0.5)
        except Exception as e:
            logger.log(f"OVModelSerializer 失敗: {e}")

    if os.path.exists(xml_path):
        file_size = os.path.getsize(xml_path)
        logger.log(f"出力OK: {xml_path} ({file_size} bytes)")
    else:
        raise RuntimeError(
            f"モデルの保存に失敗しました。ファイルが作成されません: {xml_path}\n"
            f"出力フォルダへの書き込み権限を確認してください。"
        )

    if os.path.exists(bin_path):
        file_size = os.path.getsize(bin_path)
        logger.log(f"出力OK: {bin_path} ({file_size} bytes)")
    else:
        logger.log(f"WARNING: {bin_path} が見つかりません")

    return xml_path, bin_path


def load_weights_flexible(model, weight_path: str, device, logger: UILogger, strict: bool = False):
    """checkpoint の形式揺れに耐える重みロード。
    - state_dict 直
    - dict に state_dict/model/model_state_dict 等が入っている
    - DataParallel の module. prefix
    - まれに model object そのもの
    """
    import torch

    ckpt = torch.load(weight_path, map_location="cpu")
    logger.log(f"torch.load -> type={type(ckpt)}")

    state_dict = None

    # 1) dict の中から候補キーを探す
    if isinstance(ckpt, dict):
        try:
            logger.log(f"checkpoint keys sample: {list(ckpt.keys())[:30]}")
        except Exception:
            pass

        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            v = ckpt.get(k, None)
            if isinstance(v, dict) and len(v) > 0:
                state_dict = v
                logger.log(f"using ckpt['{k}'] as state_dict")
                break

        # dict 自体が state_dict の場合
        if state_dict is None:
            any_key = next(iter(ckpt.keys()), "")
            if isinstance(any_key, str) and ("weight" in any_key or "bias" in any_key):
                state_dict = ckpt
                logger.log("treating checkpoint dict itself as state_dict")

    # 2) model object の可能性
    if state_dict is None:
        if hasattr(ckpt, "state_dict") and callable(getattr(ckpt, "state_dict")):
            try:
                sd = ckpt.state_dict()
                if isinstance(sd, dict) and len(sd) > 0:
                    state_dict = sd
                    logger.log("checkpoint looks like a model object with state_dict()")
            except Exception:
                pass

    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise RuntimeError("state_dict を抽出できませんでした（checkpoint形式が想定外）")

    # DataParallel の module. を剥がす
    if any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        logger.log("detected 'module.' prefix, stripping it")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    logger.log(f"load_state_dict(strict={strict}) done")
    logger.log(f"missing keys: {len(missing)} / unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        logger.log(f"missing sample: {missing[:20]}")
    if len(unexpected) > 0:
        logger.log(f"unexpected sample: {unexpected[:20]}")

    model = model.to(device)
    model.eval()
    return model

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

    # Try to load Donkey-style model via model_catalog (GUIのcwdズレ対策込み)
    mc = import_model_catalog(logger)
    if not hasattr(mc, 'get_model'):
        raise RuntimeError("model_catalog.py に get_model が見つかりません")
    get_model = getattr(mc, 'get_model')
    load_model_weights = getattr(mc, 'load_model_weights', None)

    if not _require("openvino", logger, "pip install openvino-dev"):
        raise RuntimeError("openvino missing")

    H, W = int(input_hw[0]), int(input_hw[1])

    device = torch.device("cpu")
    logger.log(f"PyTorchモデルを構築: model_type={model_type}, input_size={(H,W)}")
    model = get_model(model_type, pretrained=False, input_size=(H, W))
    # load_model_weights があればまずそれを試し、ダメなら柔軟ローダーへフォールバック
    if load_model_weights is not None:
        try:
            model = load_model_weights(model, model_path, device)
            logger.log("load_model_weights() succeeded")
        except Exception as e:
            logger.log(f"load_model_weights() failed, fallback: {e}")
            logger.log(traceback.format_exc())
            model = load_weights_flexible(model, model_path, device, logger, strict=False)
    else:
        logger.log("model_catalog に load_model_weights が無いので flexible loader を使います")
        model = load_weights_flexible(model, model_path, device, logger, strict=False)
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

def quantize_openvino_int8_nncf(fp32_xml_path: str,
                               out_xml_path: str,
                               calib_dir: str,
                               logger: "UILogger",
                               max_images: int = 300) -> str:
    """FP32 OpenVINO IR(.xml/.bin) を NNCF で INT8 量子化して保存する。"""
    import os, glob, time, gc
    import cv2
    import numpy as np

    # logger は UI 用オブジェクトなので「呼び出す」のではなく log/info を使う
    def _log(msg: str):
        if logger is None:
            print(msg)
            return
        for name in ("log", "info", "write", "append", "add"):
            if hasattr(logger, name):
                getattr(logger, name)(msg)
                return
        print(msg)

    try:
        import nncf
        from nncf import Dataset
    except Exception as e:
        raise RuntimeError("nncf missing (pip install nncf)") from e

    # OpenVINO 2024+ では serialize が非推奨、save_model を使用
    try:
        from openvino import save_model, Core
    except ImportError:
        from openvino.runtime import Core, serialize as _serialize
        def save_model(model, path, **kwargs):
            _serialize(model, path)

    calib_dir = os.path.abspath(calib_dir)
    out_xml_path = os.path.abspath(out_xml_path)
    out_bin_path = os.path.splitext(out_xml_path)[0] + ".bin"

    # 既存の出力があると Windows ではロックや衝突で失敗しがちなので先に消す
    for p in (out_xml_path, out_bin_path):
        try:
            if os.path.exists(p):
                os.remove(p)
        except PermissionError:
            gc.collect()
            time.sleep(0.2)
            if os.path.exists(p):
                os.remove(p)

    core = Core()
    model = core.read_model(fp32_xml_path)

    # 入力（1入力前提）
    input_port = model.inputs[0]
    input_name = input_port.get_any_name()
    shape = input_port.partial_shape  # [N,C,H,W] を期待

    def dim_to_int(dim, default):
        try:
            return int(dim.get_length()) if dim.is_static else default
        except Exception:
            return default

    C = dim_to_int(shape[1], 3)
    H = dim_to_int(shape[2], 224)
    W = dim_to_int(shape[3], 224)

    # 校正画像を収集
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for e in exts:
        image_paths += glob.glob(os.path.join(calib_dir, e))
    image_paths = sorted(image_paths)[:max_images]

    if not image_paths:
        raise RuntimeError(f"no calibration images in: {calib_dir}")

    _log(f"校正画像: {len(image_paths)} 枚  input={input_name} size={H}x{W}  (C={C})")

    def transform_fn(img_path: str):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((H, W, 3), dtype=np.uint8)

        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 0..1 float32, CHW, NCHW
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        return {input_name: x}

    calibration_dataset = Dataset(image_paths, transform_fn)

    _log("NNCF quantize 開始（少し待つ）...")
    q_model = nncf.quantize(
        model,
        calibration_dataset=calibration_dataset,
        subset_size=min(len(image_paths), max_images),
    )

    _log(f"INT8モデルを保存: {out_xml_path} (+ .bin)")
    save_model(q_model, out_xml_path)

    if not os.path.exists(out_xml_path) or not os.path.exists(out_bin_path):
        raise RuntimeError(f"serialize succeeded but output missing: {out_xml_path} / {out_bin_path}")

    _log("INT8 変換完了")
    return out_xml_path


def convert_dispatch(opts: ConvertOptions, logger: UILogger):
    model_path = os.path.abspath(opts.model_path)
    out_dir = os.path.abspath(opts.output_dir)
    precision = opts.precision.upper()

    _ensure_dir(out_dir)

    ext = os.path.splitext(model_path)[1].lower()
    stem = _safe_join(out_dir, _path_noext(model_path) + f"_{precision.lower()}")

    # Decide route
    is_yolo_hint = ("yolo" in os.path.basename(model_path).lower()) or (opts.model_type and "yolo" in opts.model_type.lower())
    if ext == ".onnx":
        xml, _ = convert_onnx_to_openvino(model_path, stem, precision, logger)
        if precision == "INT8":
            if not opts.calib_dir:
                raise RuntimeError("INT8には校正画像フォルダが必要です")
            fp_xml = xml  # created FP32
            out_xml = _safe_join(out_dir, _path_noext(model_path) + "_int8_quantized.xml")
            quantize_openvino_int8_nncf(fp_xml, out_xml, opts.calib_dir, logger, max_images=opts.calib_max_images)
            # 量子化完了後、FP32中間ファイルを削除
            for tmp in (fp_xml, fp_xml.replace(".xml", ".bin")):
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                        logger.log(f"FP32中間ファイル削除: {tmp}")
                except Exception:
                    pass

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
                fp_xml = xml  # FP32中間ファイル（stem_int8.xml として保存済み）
                # INT8出力先はFP32中間とは別のパスにする
                out_xml = _safe_join(out_dir, _path_noext(model_path) + "_int8_quantized.xml")
                quantize_openvino_int8_nncf(fp_xml, out_xml, opts.calib_dir, logger, max_images=opts.calib_max_images)
                # 量子化完了後、FP32中間ファイルを削除
                for tmp in (fp_xml, fp_xml.replace(".xml", ".bin")):
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                            logger.log(f"FP32中間ファイル削除: {tmp}")
                    except Exception:
                        pass
    else:
        raise RuntimeError(f"未対応の拡張子: {ext}")

# -----------------------------
# Tkinter GUI
# -----------------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def load_model_types(logger=None):
    """model_catalog.py から利用可能な model_type を列挙。
    GUI実行時のcwdズレ等で import が失敗しがちなので、失敗時は空リストにフォールバック。
    """
    try:
        mc = import_model_catalog(logger)
    except Exception:
        return []
    # いくつかの候補名に対応
    for fn in ("list_available_models", "list_all_available_models", "list_models"):
        if hasattr(mc, fn):
            try:
                vals = getattr(mc, fn)()
                if isinstance(vals, (list, tuple)):
                    return sorted({str(v) for v in vals})
            except Exception:
                pass
    # ダメなら get_model のエラーメッセージに頼らず、空で返す
    return []


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
        self.var_out_dir = tk.StringVar(value=SCRIPT_DIR)
        self.var_precision = tk.StringVar(value="FP16")
        self.var_model_type = tk.StringVar()
        self.model_types = load_model_types()  # model_catalog.py から取得（無ければ空）
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
        self.cmb_model_type = ttk.Combobox(row, textvariable=self.var_model_type, values=self.model_types, width=38)
        self.cmb_model_type.configure(state='normal')  # 候補から選べるし手入力もできる
        self.cmb_model_type.pack(side='left', padx=8)
        ttk.Button(row, text='Reload', command=self._reload_model_types).pack(side='left')
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

    def _reload_model_types(self):
        """Reload model types from model_catalog.py and refresh the combobox."""
        try:
            self.model_types = load_model_types(logger=self.logger)
        except Exception as e:
            # keep previous values if reload fails
            try:
                self.logger.log(f"モデル一覧の再読み込みに失敗: {e}")
            except Exception:
                pass
            return

        if hasattr(self, 'cmb_model_type'):
            self.cmb_model_type['values'] = self.model_types

        cur = (self.var_model_type.get() or '').strip()
        if cur and (cur not in self.model_types):
            self.var_model_type.set('')

        try:
            self.logger.log(f"モデル一覧を更新しました: {len(self.model_types)} 件")
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