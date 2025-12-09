# ============================================================
# 软件名称：基于 SenseVoice 的 ASR 可视化操作系统
# 版本号：S-ASR-25.12
# 作者：Silas、Portnoy
# 立项日期：2025-11-25
#
# 版权所有 © 2025 Silas、Portnoy 保留所有权利
#
# 本软件为自主设计开发，包含：
# - 基于 SenseVoiceSmall 的本地语音识别核心
# - 实时流式 ASR 监听模块
# - VAD 声音活动检测模块
# - 文件识别、音频解码、噪声抑制模块
# - Flask 后端服务与前端交互接口
#
# 本软件的源代码、界面结构、交互逻辑、模块划分及实现方式
# 均属于作者原创设计，禁止未经授权的复制、修改与商用。
#
# 若需引用或二次开发，请保持本声明完整。
# ============================================================
import os
import time
import threading
import subprocess
import webbrowser
import io
import tempfile
import numpy as np
import torch
import pyaudio
import webrtcvad
import librosa
from flask import Flask, request, jsonify, send_from_directory
from funasr import AutoModel         
import config
from noise_reduction import denoise_audio

# ====== 全局路径与初始化 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "iic", "SenseVoiceSmall")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RATE = 16000
CHUNK_TIME = 0.02
CHUNK = int(RATE * CHUNK_TIME)

print(f"[INFO] 使用设备: {DEVICE}")
print(f"[INFO] 模型目录: {MODEL_PATH}")

# ====== 环境检测 ======
def check_environment():
    try:
        print(f"[INFO] PyTorch 版本: {torch.__version__}")
    except:
        print("[WARN] 未检测到有效的 PyTorch")

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            print(f"[INFO] FFmpeg: {result.stdout.splitlines()[0]}")
        else:
            print("[WARN] 未检测到 FFmpeg")
    except:
        print("[WARN] 未检测到 FFmpeg 可执行文件")

# ====== 加载模型（AutoModel） ======
print("[INFO] 正在加载本地 SenseVoiceSmall 模型…")

model = AutoModel(
    model=MODEL_PATH,
    device=DEVICE
)

print("[INFO] 模型加载完成")

# ====== Flask ======
app = Flask(__name__)

import logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# ====== 流式识别相关全局变量 ======
vad = webrtcvad.Vad(config.VAD_MODE)

p = None
stream = None
audio_buffer = []
silent_chunks = 0
listening_active = False
listening_thread = None
stream_results = []

buffer_lock = threading.Lock()


# ====== 保存配置 ======
def save_config(volume_threshold, silence_duration, vad_mode,
                nr_enabled=None, nr_method=None):
    """保存配置到 config.py，并同步到内存。

    nr_enabled / nr_method 如果未传入，则保留现有 config 中的值。
    """
    if nr_enabled is None:
        nr_enabled = getattr(config, "NR_ENABLED", False)
    if nr_method is None:
        nr_method = getattr(config, "NR_METHOD", "spectral")

    content = f"""# 自动生成的配置文件

VOLUME_THRESHOLD = {float(volume_threshold)}
SILENCE_DURATION = {float(silence_duration)}
VAD_MODE = {int(vad_mode)}
NR_ENABLED = {bool(nr_enabled)}
NR_METHOD = "{str(nr_method)}"
"""
    path = os.path.join(BASE_DIR, "config.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    # 同时更新内存中的值
    config.VOLUME_THRESHOLD = float(volume_threshold)
    config.SILENCE_DURATION = float(silence_duration)
    config.VAD_MODE = int(vad_mode)
    config.NR_ENABLED = bool(nr_enabled)
    config.NR_METHOD = str(nr_method)

    # 更新 VAD
    global vad
    vad = webrtcvad.Vad(int(config.VAD_MODE))

    print("[INFO] 配置已保存")


# ====== 文件转音频 ======
def load_audio_to_array(file_stream):
    try:
        data = file_stream.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(data)
            path = tmp.name

        audio, sr = librosa.load(path, sr=RATE, mono=True)
        audio = audio.astype(np.float32)

        os.remove(path)
        return audio, sr

    except Exception as e:
        raise RuntimeError(f"上传文件解码失败: {e}")


# ====== 推理 + 清洗 ======
def infer_text(audio_array):
    # 可选噪音抑制
    nr_enabled = getattr(config, "NR_ENABLED", False)
    nr_method = getattr(config, "NR_METHOD", "spectral")

    if nr_enabled:
        try:
            audio_array = denoise_audio(audio_array, RATE, method=nr_method)
        except Exception as e:
            print(f"[WARN] 噪音抑制失败: {e}")

    res = model.generate(
        input=audio_array,
        language="zh,en"
    )

    text = res[0]["text"].strip()

    # nospeech 过滤
    if "<|nospeech|>" in text:
        return ""

    # 过滤非中英文
    if text.startswith("<|") and not (
        text.startswith("<|zh|>") or text.startswith("<|en|>")
    ):
        return ""

    # 过滤英文单字母噪声
    clean = text.replace(" ", "").lower()
    if len(clean) == 1 and clean in ["a", "i", "o", "e", "u"]:
        return ""

    return text


# ====== 流式识别线程（移植 text.py） ======
def listening_loop():
    global p, stream, audio_buffer, silent_chunks, listening_active, stream_results

    print("[INFO] 启动实时监听线程…")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("[INFO] 麦克风已打开")

    audio_buffer = []
    silent_chunks = 0

    try:
        while listening_active:

            data = stream.read(CHUNK, exception_on_overflow=False)
            npdata = (
                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            is_voice = vad.is_speech(data, RATE)

            if is_voice:
                audio_buffer.append(npdata)

            if not is_voice:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks * CHUNK_TIME >= config.SILENCE_DURATION:

                if len(audio_buffer) > 0:
                    full_audio = np.concatenate(audio_buffer)
                    audio_buffer = []
                    silent_chunks = 0

                    if len(full_audio) < RATE * 0.5:
                        continue

                    print("[INFO] 检测到一句话，开始推理…")

                    text = infer_text(full_audio)

                    if text:
                        print(f"[ASR] {text}")

                        with buffer_lock:
                            stream_results.append(text)

    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()

        print("[INFO] 实时监听线程结束")


# ====== 路由 ======
@app.route("/")
def page_index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({
        "volume_threshold": config.VOLUME_THRESHOLD,
        "silence_duration": config.SILENCE_DURATION,
        "vad_mode": config.VAD_MODE,
        "nr_enabled": getattr(config, "NR_ENABLED", False),
        "nr_method": getattr(config, "NR_METHOD", "spectral"),
    })


@app.route("/save_config", methods=["POST"])
def route_save_config():
    data = request.get_json() or request.form

    vt = data.get("volume_threshold", config.VOLUME_THRESHOLD)
    sd = data.get("silence_duration", config.SILENCE_DURATION)
    vm = data.get("vad_mode", config.VAD_MODE)
    nr_enabled = data.get("nr_enabled", getattr(config, "NR_ENABLED", False))
    nr_method = data.get("nr_method", getattr(config, "NR_METHOD", "spectral"))

    try:
        save_config(vt, sd, vm, nr_enabled, nr_method)
        return jsonify({"ok": True, "message": "配置保存成功"})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/recognize_file", methods=["POST"])
def route_recognize_file():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"ok": False, "message": "未上传文件"}), 400

    f = request.files["file"]
    audio_array, _ = load_audio_to_array(f.stream)

    text = infer_text(audio_array)

    return jsonify({"ok": True, "text": text})


@app.route("/start_listen", methods=["POST"])
def route_start_listen():
    global listening_active, listening_thread, stream_results

    if listening_active:
        return jsonify({"ok": True, "message": "监听已在运行"})

    listening_active = True
    stream_results = []

    listening_thread = threading.Thread(
        target=listening_loop,
        daemon=True
    )
    listening_thread.start()

    return jsonify({"ok": True, "message": "监听已启动"})


@app.route("/stream_results", methods=["GET"])
def route_stream_results():
    global stream_results
    with buffer_lock:
        out = stream_results[:]
        stream_results = []
    return jsonify({"ok": True, "lines": out})


@app.route("/stop_listen", methods=["POST"])
def route_stop_listen():
    global listening_active
    listening_active = False
    return jsonify({"ok": True})


# ====== 退出 ======
def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()
    else:
        os._exit(0)


@app.route("/shutdown", methods=["POST"])
def route_shutdown():
    global listening_active
    listening_active = False
    shutdown_server()
    return jsonify({"ok": True})


# ====== 打开浏览器 ======
def open_browser():
    url = "http://127.0.0.1:1125/"
    time.sleep(1)
    webbrowser.open_new(url)


# ====== 主入口 ======
if __name__ == "__main__":
    check_environment()

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=1125, debug=False)
