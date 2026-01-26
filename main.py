#!/usr/bin/env python3
import asyncio
import queue
import ssl
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np
import sounddevice as sd
from aiohttp import web, WSMsgType

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 9000

TARGET_SR = 48000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(TARGET_SR * FRAME_MS / 1000)

DEFAULT_CHANNEL_GAIN = 1.0
DEFAULT_MASTER_GAIN = 1.0
DEFAULT_LIMITER_CEILING = 0.98  # 0..1 (float)
DEFAULT_GATE_THRESHOLD_RMS = 0.02  # 0..1, bem leve (opcional)
DEFAULT_GATE_ATTENUATION = 0.15  # quando abaixo do threshold, reduz para 15%

MAX_QUEUE_FRAMES = 200  # ~4s de áudio (200 * 20ms)

# =========================
# HTML (client + mixer)
# =========================

CLIENT_HTML = f"""<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Mic → Hub</title>
  <style>
    body {{ font-family: system-ui, sans-serif; padding: 16px; max-width: 720px; margin: auto; }}
    .row {{ margin: 12px 0; }}
    button, input {{ font-size: 16px; padding: 10px 12px; }}
    input {{ width: 100%; max-width: 420px; }}
    pre {{ background:#f4f4f4; padding:12px; border-radius:10px; }}
    .hint {{ color:#444; font-size: 14px; }}
  </style>
</head>
<body>
  <h2>Enviar microfone do celular para o Hub</h2>

  <div class="row">
    <label>Channel ID</label><br>
    <input id="ch" type="number" value="1" min="1" />
    <div class="hint">Dica: cada celular use um Channel ID diferente (1,2,3...).</div>
  </div>

  <div class="row">
    <button id="start">Start</button>
    <button id="stop" disabled>Stop</button>
  </div>

  <pre id="log">Pronto. (Abra em /mixer no PC para controlar os canais.)</pre>

<script>
(() => {{
  const logEl = document.getElementById("log");
  const startBtn = document.getElementById("start");
  const stopBtn = document.getElementById("stop");
  const chEl = document.getElementById("ch");

  let ws = null;
  let audioCtx = null;
  let processor = null;
  let source = null;
  let stream = null;

  const TARGET_SR = {TARGET_SR};
  const FRAME_MS = {FRAME_MS};
  const SAMPLES_PER_FRAME = Math.round(TARGET_SR * FRAME_MS / 1000);

  function log(s) {{ logEl.textContent = s; }}

  function floatToInt16PCM(float32) {{
    const buf = new ArrayBuffer(float32.length * 2);
    const view = new DataView(buf);
    for (let i = 0; i < float32.length; i++) {{
      let x = float32[i];
      x = Math.max(-1, Math.min(1, x));
      const s = x < 0 ? x * 0x8000 : x * 0x7FFF;
      view.setInt16(i * 2, s, true);
    }}
    return new Uint8Array(buf);
  }}

  async function start() {{
      const ch = parseInt(chEl.value || "1", 10);
      const proto = (location.protocol === "https:") ? "wss" : "ws";
      const wsUrl = `${{proto}}://${{location.host}}/ws_audio?ch=${{encodeURIComponent(ch)}}`;

      log("Pedindo permissão do microfone...");
      stream = await navigator.mediaDevices.getUserMedia({{
        audio: {{
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }}
      }});

      ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => log(`Conectado. Enviando áudio no canal ${{ch}}...`);
      ws.onclose = () => log("WebSocket fechado.");
      ws.onerror = () => log("Erro WebSocket.");

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate: TARGET_SR }});
    source = audioCtx.createMediaStreamSource(stream);

    const bufferSize = 1024; // MVP
    processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

    let stash = [];
    let stashLen = 0;

    processor.onaudioprocess = (ev) => {{
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const input = ev.inputBuffer.getChannelData(0);

      stash.push(input.slice());
      stashLen += input.length;

      while (stashLen >= SAMPLES_PER_FRAME) {{
        const frame = new Float32Array(SAMPLES_PER_FRAME);
        let filled = 0;

        while (filled < SAMPLES_PER_FRAME && stash.length) {{
          const head = stash[0];
          const need = SAMPLES_PER_FRAME - filled;

          if (head.length <= need) {{
            frame.set(head, filled);
            filled += head.length;
            stash.shift();
          }} else {{
            frame.set(head.subarray(0, need), filled);
            stash[0] = head.subarray(need);
            filled += need;
          }}
        }}

        stashLen -= SAMPLES_PER_FRAME;
        const pcmBytes = floatToInt16PCM(frame);
        ws.send(pcmBytes);
      }}
    }};

    source.connect(processor);
    processor.connect(audioCtx.destination);

    startBtn.disabled = true;
    stopBtn.disabled = false;
  }}

  async function stop() {{
    startBtn.disabled = false;
    stopBtn.disabled = true;

    if (processor) {{ processor.disconnect(); processor = null; }}
    if (source) {{ source.disconnect(); source = null; }}
    if (audioCtx) {{ await audioCtx.close(); audioCtx = null; }}
    if (ws) {{ ws.close(); ws = null; }}

    if (stream) {{
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }}
    log("Parado.");
  }}

  startBtn.onclick = () => start().catch(err => log("Erro: " + err));
  stopBtn.onclick = () => stop().catch(err => log("Erro: " + err));
}})();
</script>
</body>
</html>
"""

MIXER_HTML = """<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Mixer</title>
  <style>
    body { font-family: system-ui, sans-serif; padding: 16px; }
    .top { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-end; margin-bottom: 16px; }
    .card { border:1px solid #ddd; border-radius: 12px; padding: 12px; }
    .channels { display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end; }
    .ch { width: 180px; }
    .title { font-weight: 700; margin-bottom: 8px; }
    .row { margin: 8px 0; }
    input[type="range"] { width: 100%; }
    button { padding: 8px 10px; font-size: 14px; cursor:pointer; }
    .btns { display:flex; gap:8px; }
    .meter { height: 10px; background:#eee; border-radius: 10px; overflow:hidden; }
    .meter > div { height: 100%; width: 0%; background:#111; }
    .small { color:#444; font-size: 12px; }
    .warn { color:#b00; font-size: 12px; }
  </style>
</head>
<body>
  <h2>Mixer (Mesa de som do Hub)</h2>

  <div class="top">
    <div class="card">
      <div class="title">Master</div>
      <div class="row">
        <label>Master Gain: <span id="masterGainVal">1.00</span></label>
        <input id="masterGain" type="range" min="0" max="2" step="0.01" value="1.00">
      </div>
      <div class="row">
        <label>Limiter Ceiling: <span id="ceilingVal">0.98</span></label>
        <input id="ceiling" type="range" min="0.50" max="1.00" step="0.01" value="0.98">
      </div>
      <div class="row">
        <label>Gate Threshold (RMS): <span id="gateVal">0.02</span></label>
        <input id="gate" type="range" min="0" max="0.20" step="0.005" value="0.02">
        <div class="small">0 = gate desligado</div>
      </div>
      <div class="row btns">
        <button id="refresh">Atualizar</button>
      </div>
      <div class="row small" id="status">Conectando...</div>
    </div>

    <div class="card">
      <div class="title">Como usar</div>
      <div class="small">
        1) No celular, abra <b>/client</b> e escolha um Channel ID.<br>
        2) No PC, ajuste volume, mute/solo e veja os meters aqui.<br>
        3) Se ninguém fala e o ambiente fica “somando”, aumente um pouco o Gate (ou mute canais ociosos).
      </div>
    </div>
  </div>

  <div class="channels" id="channels"></div>

<script>
(() => {
  const proto = (location.protocol === "https:") ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws_ctrl`);

  const chWrap = document.getElementById("channels");
  const statusEl = document.getElementById("status");

  const masterGain = document.getElementById("masterGain");
  const ceiling = document.getElementById("ceiling");
  const gate = document.getElementById("gate");
  const masterGainVal = document.getElementById("masterGainVal");
  const ceilingVal = document.getElementById("ceilingVal");
  const gateVal = document.getElementById("gateVal");

  const refreshBtn = document.getElementById("refresh");

  const state = {
    channels: {}
  };

  function send(obj) {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function renderChannel(ch) {
    const id = ch.id;
    let el = document.getElementById(`ch_${id}`);
    if (!el) {
      el = document.createElement("div");
      el.className = "card ch";
      el.id = `ch_${id}`;
      el.innerHTML = `
        <div class="title">CH ${id} <span class="small" id="name_${id}"></span></div>

        <div class="row">
          <label>Gain: <span id="gainVal_${id}">1.00</span></label>
          <input id="gain_${id}" type="range" min="0" max="2" step="0.01" value="1.00">
        </div>

        <div class="row btns">
          <button id="mute_${id}">Mute</button>
          <button id="solo_${id}">Solo</button>
        </div>

        <div class="row">
          <div class="meter"><div id="meter_${id}"></div></div>
          <div class="small">Queue: <span id="q_${id}">0</span> | Drops: <span id="d_${id}">0</span></div>
          <div class="warn" id="warn_${id}"></div>
        </div>
      `;
      chWrap.appendChild(el);

      const gainEl = document.getElementById(`gain_${id}`);
      gainEl.oninput = () => {
        document.getElementById(`gainVal_${id}`).textContent = Number(gainEl.value).toFixed(2);
        send({ type: "set_channel", id, gain: Number(gainEl.value) });
      };

      document.getElementById(`mute_${id}`).onclick = () => {
        send({ type: "toggle_mute", id });
      };
      document.getElementById(`solo_${id}`).onclick = () => {
        send({ type: "toggle_solo", id });
      };
    }

    // update view
    document.getElementById(`gain_${id}`).value = ch.gain.toFixed(2);
    document.getElementById(`gainVal_${id}`).textContent = ch.gain.toFixed(2);
    document.getElementById(`mute_${id}`).textContent = ch.muted ? "Muted" : "Mute";
    document.getElementById(`solo_${id}`).textContent = ch.solo ? "Solo ON" : "Solo";
    document.getElementById(`q_${id}`).textContent = ch.queue_depth;
    document.getElementById(`d_${id}`).textContent = ch.drops;

    const meter = document.getElementById(`meter_${id}`);
    meter.style.width = `${Math.min(100, Math.round(ch.level * 100))}%`;

    const warn = document.getElementById(`warn_${id}`);
    warn.textContent = ch.queue_depth > 80 ? "buffer alto (latência)" : "";
  }

  function renderAll(payload) {
    statusEl.textContent = `Conectado. Canais: ${payload.channels.length}`;
    masterGain.value = payload.master.master_gain;
    ceiling.value = payload.master.limiter_ceiling;
    gate.value = payload.master.gate_threshold;

    masterGainVal.textContent = Number(masterGain.value).toFixed(2);
    ceilingVal.textContent = Number(ceiling.value).toFixed(2);
    gateVal.textContent = Number(gate.value).toFixed(3);

    // render channels
    payload.channels.sort((a,b) => a.id - b.id).forEach(renderChannel);
  }

  ws.onopen = () => {
    statusEl.textContent = "Conectado. Carregando...";
    send({ type: "get_state" });
  };
  ws.onclose = () => statusEl.textContent = "Desconectado.";
  ws.onerror = () => statusEl.textContent = "Erro no WebSocket.";

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") renderAll(msg.payload);
    if (msg.type === "tick") renderAll(msg.payload);
  };

  masterGain.oninput = () => {
    masterGainVal.textContent = Number(masterGain.value).toFixed(2);
    send({ type: "set_master", master_gain: Number(masterGain.value) });
  };
  ceiling.oninput = () => {
    ceilingVal.textContent = Number(ceiling.value).toFixed(2);
    send({ type: "set_master", limiter_ceiling: Number(ceiling.value) });
  };
  gate.oninput = () => {
    gateVal.textContent = Number(gate.value).toFixed(3);
    send({ type: "set_master", gate_threshold: Number(gate.value) });
  };

  refreshBtn.onclick = () => send({ type: "get_state" });
})();
</script>
</body>
</html>
"""


# =========================
# Mixer State
# =========================

@dataclass
class ChannelState:
    id: int
    gain: float = DEFAULT_CHANNEL_GAIN
    muted: bool = False
    solo: bool = False
    # buffer de frames int16 (mono)
    q: "queue.Queue[np.ndarray]" = field(default_factory=lambda: queue.Queue(maxsize=MAX_QUEUE_FRAMES))
    drops: int = 0
    level: float = 0.0  # 0..1 (RMS aproximado)
    last_seen: float = field(default_factory=time.time)


@dataclass
class MasterState:
    master_gain: float = DEFAULT_MASTER_GAIN
    limiter_ceiling: float = DEFAULT_LIMITER_CEILING
    gate_threshold: float = DEFAULT_GATE_THRESHOLD_RMS
    gate_attenuation: float = DEFAULT_GATE_ATTENUATION


class HubMixer:
    def __init__(self):
        self.channels: Dict[int, ChannelState] = {}
        self.master = MasterState()
        self._lock = asyncio.Lock()
        self._ctrl_clients: set[web.WebSocketResponse] = set()

        self._audio_stream: Optional[sd.OutputStream] = None
        self._running = False

    def get_or_create_channel(self, ch_id: int) -> ChannelState:
        ch = self.channels.get(ch_id)
        if ch is None:
            ch = ChannelState(id=ch_id)
            self.channels[ch_id] = ch
        return ch

    def push_audio_frame(self, ch_id: int, payload: bytes) -> None:
        # payload é PCM int16 LE mono, exatamente 20ms (ideal)
        ch = self.get_or_create_channel(ch_id)
        ch.last_seen = time.time()

        frame = np.frombuffer(payload, dtype=np.int16)

        if frame.size < SAMPLES_PER_FRAME:
            frame = np.pad(frame, (0, SAMPLES_PER_FRAME - frame.size))
        elif frame.size > SAMPLES_PER_FRAME:
            frame = frame[:SAMPLES_PER_FRAME]

        # nível RMS (para meter)
        f = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(f * f) + 1e-12))
        ch.level = min(1.0, rms)

        try:
            ch.q.put_nowait(frame)
        except queue.Full:
            ch.drops += 1
            # descarta um antigo pra não aumentar latência
            try:
                _ = ch.q.get_nowait()
            except queue.Empty:
                pass
            try:
                ch.q.put_nowait(frame)
            except queue.Full:
                pass

    def _mix_block(self, frames: int) -> np.ndarray:
        """
        Retorna float32 mono [-1..1] com frames amostras.
        """
        if not self.channels:
            return np.zeros((frames,), dtype=np.float32)

        # solo logic
        solo_any = any(ch.solo for ch in self.channels.values())
        active = []
        for ch in self.channels.values():
            if solo_any:
                if ch.solo and not ch.muted:
                    active.append(ch)
            else:
                if not ch.muted:
                    active.append(ch)

        if not active:
            return np.zeros((frames,), dtype=np.float32)

        mix = np.zeros((frames,), dtype=np.float32)

        for ch in active:
            try:
                frame_i16 = ch.q.get_nowait()
            except queue.Empty:
                frame_i16 = np.zeros((frames,), dtype=np.int16)

            if frame_i16.shape[0] != frames:
                if frame_i16.shape[0] < frames:
                    frame_i16 = np.pad(frame_i16, (0, frames - frame_i16.shape[0]))
                else:
                    frame_i16 = frame_i16[:frames]

            x = frame_i16.astype(np.float32) / 32768.0

            # gate simples (opcional)
            gt = self.master.gate_threshold
            if gt > 0:
                rms = float(np.sqrt(np.mean(x * x) + 1e-12))
                if rms < gt:
                    x *= self.master.gate_attenuation

            # ganho por canal
            x *= float(ch.gain)

            mix += x

        # master gain
        mix *= float(self.master.master_gain)

        # limiter simples (hard clip no ceiling)
        ceiling = float(self.master.limiter_ceiling)
        ceiling = max(0.05, min(1.0, ceiling))
        mix = np.clip(mix, -ceiling, ceiling)

        return mix

    def _audio_callback(self, outdata, frames, time_info, status):
        block = self._mix_block(frames)
        # sounddevice espera shape (frames, channels)
        outdata[:] = block.reshape(-1, 1)

    def start_audio(self):
        if self._audio_stream is not None:
            return

        self._audio_stream = sd.OutputStream(
            samplerate=TARGET_SR,
            channels=1,
            dtype="float32",
            blocksize=SAMPLES_PER_FRAME,  # 20ms
            callback=self._audio_callback,
        )
        self._audio_stream.start()
        self._running = True
        print("🎧 Audio playback iniciado (hub mixer).")

    def stop_audio(self):
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        self._running = False

    def state_payload(self) -> dict:
        chans = []
        for ch in self.channels.values():
            chans.append({
                "id": ch.id,
                "gain": float(ch.gain),
                "muted": bool(ch.muted),
                "solo": bool(ch.solo),
                "queue_depth": int(ch.q.qsize()),
                "drops": int(ch.drops),
                "level": float(ch.level),
            })
        return {
            "master": {
                "master_gain": float(self.master.master_gain),
                "limiter_ceiling": float(self.master.limiter_ceiling),
                "gate_threshold": float(self.master.gate_threshold),
            },
            "channels": chans,
        }

    async def broadcast_tick(self):
        # envia updates periódicos pros painéis conectados
        payload = self.state_payload()
        dead = []
        for ws in self._ctrl_clients:
            try:
                await ws.send_json({"type": "tick", "payload": payload})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ctrl_clients.discard(ws)


hub = HubMixer()


# =========================
# HTTP Handlers
# =========================

async def page_client(_request: web.Request):
    return web.Response(text=CLIENT_HTML, content_type="text/html")


async def page_mixer(_request: web.Request):
    return web.Response(text=MIXER_HTML, content_type="text/html")


async def ws_audio(request: web.Request):
    """
    WebSocket que recebe binário PCM int16 (mono) vindo do navegador do celular.
    URL: /ws_audio?ch=1
    """
    ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
    await ws.prepare(request)

    ch_id = int(request.query.get("ch", "1"))
    print(f"[AUDIO] conectado: ch={ch_id} from {request.remote}")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                hub.push_audio_frame(ch_id, msg.data)
            elif msg.type == WSMsgType.ERROR:
                print("[AUDIO] erro:", ws.exception())
                break
    finally:
        await ws.close()
        print(f"[AUDIO] desconectado: ch={ch_id}")

    return ws


async def ws_ctrl(request: web.Request):
    """
    WebSocket do painel /mixer para controlar canais.
    """
    ws = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024)
    await ws.prepare(request)

    hub._ctrl_clients.add(ws)
    # envia state inicial
    await ws.send_json({"type": "state", "payload": hub.state_payload()})

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data: dict[str, Any] = msg.json()
                except Exception:
                    continue

                t = data.get("type")
                if t == "get_state":
                    await ws.send_json({"type": "state", "payload": hub.state_payload()})

                elif t == "set_master":
                    if "master_gain" in data:
                        hub.master.master_gain = float(data["master_gain"])
                    if "limiter_ceiling" in data:
                        hub.master.limiter_ceiling = float(data["limiter_ceiling"])
                    if "gate_threshold" in data:
                        hub.master.gate_threshold = float(data["gate_threshold"])
                    await ws.send_json({"type": "state", "payload": hub.state_payload()})

                elif t == "set_channel":
                    ch_id = int(data.get("id", 0))
                    ch = hub.get_or_create_channel(ch_id)
                    if "gain" in data:
                        ch.gain = float(data["gain"])
                    await ws.send_json({"type": "state", "payload": hub.state_payload()})

                elif t == "toggle_mute":
                    ch_id = int(data.get("id", 0))
                    ch = hub.get_or_create_channel(ch_id)
                    ch.muted = not ch.muted
                    await ws.send_json({"type": "state", "payload": hub.state_payload()})

                elif t == "toggle_solo":
                    ch_id = int(data.get("id", 0))
                    ch = hub.get_or_create_channel(ch_id)
                    ch.solo = not ch.solo
                    await ws.send_json({"type": "state", "payload": hub.state_payload()})

            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        hub._ctrl_clients.discard(ws)
        await ws.close()

    return ws


async def ticker_task(app: web.Application):
    # start audio once server is up
    hub.start_audio()
    try:
        while True:
            await hub.broadcast_tick()
            await asyncio.sleep(0.20)  # 5x/s
    finally:
        hub.stop_audio()


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", page_mixer)
    app.router.add_get("/mixer", page_mixer)
    app.router.add_get("/client", page_client)
    app.router.add_get("/ws_audio", ws_audio)
    app.router.add_get("/ws_ctrl", ws_ctrl)

    async def on_startup(app: web.Application):
        app["ticker"] = asyncio.create_task(ticker_task(app))

    async def on_cleanup(app: web.Application):
        t = app.get("ticker")
        if t:
            t.cancel()
            try:
                await t
            except Exception:
                pass

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    print(f"✅ Hub Mixer rodando em https://{HTTP_HOST}:{HTTP_PORT}")
    print(f"   - Mixer (PC):   https://192.168.100.218:{HTTP_PORT}/mixer")
    print(f"   - Client (cel): https://192.168.100.218:{HTTP_PORT}/client")
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain("cert.pem", "key.pem")
    web.run_app(make_app(), host=HTTP_HOST, port=HTTP_PORT, ssl_context=ssl_ctx)