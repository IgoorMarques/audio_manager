#!/usr/bin/env python3
import asyncio
import queue
import ssl
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import av
import numpy as np
import sounddevice as sd
from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 9000

TARGET_SR = 48000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(TARGET_SR * FRAME_MS / 1000)

DEFAULT_CHANNEL_GAIN = 1.0
DEFAULT_MASTER_GAIN = 1.0
DEFAULT_LIMITER_CEILING = 0.98
DEFAULT_GATE_THRESHOLD_RMS = 0.00
DEFAULT_GATE_ATTENUATION = 0.15

MAX_QUEUE_FRAMES = 10

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
    pre {{ background:#f4f4f4; padding:12px; border-radius:10px; white-space: pre-wrap; }}
    .hint {{ color:#444; font-size: 14px; }}
  </style>
</head>
<body>
  <h2>Enviar microfone do celular para o Hub (WebRTC/Opus)</h2>

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

  let pc = null;
  let stream = null;

  function log(s) {{ logEl.textContent = s; }}

  async function start() {{
    const ch = parseInt(chEl.value || "1", 10);

    log("Pedindo permissão do microfone...");
    stream = await navigator.mediaDevices.getUserMedia({{
      audio: {{
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }},
      video: false
    }});

    // WebRTC peer connection
    pc = new RTCPeerConnection({{
      iceServers: [] // LAN: geralmente vazio é ok. Se for internet, adicione STUN/TURN.
    }});

    pc.onconnectionstatechange = () => {{
      log("Estado: " + pc.connectionState);
    }};

    // manda track de audio
    stream.getTracks().forEach(t => pc.addTrack(t, stream));

    const offer = await pc.createOffer({{ offerToReceiveAudio: false, offerToReceiveVideo: false }});
    await pc.setLocalDescription(offer);

    log("Enviando offer para o servidor...");
    const res = await fetch("/offer", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
        ch
      }})
    }});

    if (!res.ok) {{
      const t = await res.text();
      throw new Error("Falha no /offer: " + t);
    }}

    const answer = await res.json();
    await pc.setRemoteDescription(answer);

    log("Conectado. Transmitindo áudio no canal " + ch + ".");
    startBtn.disabled = true;
    stopBtn.disabled = false;
  }}

  async function stop() {{
    startBtn.disabled = false;
    stopBtn.disabled = true;

    if (pc) {{
      try {{ pc.getSenders().forEach(s => s.track && s.track.stop()); }} catch(e) {{}}
      try {{ await pc.close(); }} catch(e) {{}}
      pc = null;
    }}

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
  <h2>Mixer</h2>

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
        <div class="title">CH ${id}</div>

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

      document.getElementById(`mute_${id}`).onclick = () => send({ type: "toggle_mute", id });
      document.getElementById(`solo_${id}`).onclick = () => send({ type: "toggle_solo", id });
    }

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

    payload.channels.sort((a,b) => a.id - b.id).forEach(renderChannel);
  }

  ws.onopen = () => { statusEl.textContent = "Conectado. Carregando..."; send({ type: "get_state" }); };
  ws.onclose = () => statusEl.textContent = "Desconectado.";
  ws.onerror = () => statusEl.textContent = "Erro no WebSocket.";

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") renderAll(msg.payload);
    if (msg.type === "tick") renderAll(msg.payload);
  };

  masterGain.oninput = () => { masterGainVal.textContent = Number(masterGain.value).toFixed(2); send({ type: "set_master", master_gain: Number(masterGain.value) }); };
  ceiling.oninput = () => { ceilingVal.textContent = Number(ceiling.value).toFixed(2); send({ type: "set_master", limiter_ceiling: Number(ceiling.value) }); };
  gate.oninput = () => { gateVal.textContent = Number(gate.value).toFixed(3); send({ type: "set_master", gate_threshold: Number(gate.value) }); };

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
    q: "queue.Queue[np.ndarray]" = field(default_factory=lambda: queue.Queue(maxsize=MAX_QUEUE_FRAMES))
    drops: int = 0
    level: float = 0.0
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
        self._ctrl_clients: set[web.WebSocketResponse] = set()

        self._audio_stream: Optional[sd.OutputStream] = None
        self._running = False

        # peers webRTC
        self._pcs: set[RTCPeerConnection] = set()
        self._track_tasks: Dict[RTCPeerConnection, asyncio.Task] = {}

    def get_or_create_channel(self, ch_id: int) -> ChannelState:
        ch = self.channels.get(ch_id)
        if ch is None:
            ch = ChannelState(id=ch_id)
            self.channels[ch_id] = ch
        return ch

    def push_audio_i16_frame(self, ch_id: int, frame_i16: np.ndarray) -> None:
        """
        frame_i16: int16 mono com tamanho SAMPLES_PER_FRAME (20ms).
        """
        ch = self.get_or_create_channel(ch_id)
        ch.last_seen = time.time()

        if frame_i16.shape[0] < SAMPLES_PER_FRAME:
            frame_i16 = np.pad(frame_i16, (0, SAMPLES_PER_FRAME - frame_i16.shape[0]))
        elif frame_i16.shape[0] > SAMPLES_PER_FRAME:
            frame_i16 = frame_i16[:SAMPLES_PER_FRAME]

        # meter RMS
        f = frame_i16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(f * f) + 1e-12))
        ch.level = min(1.0, rms)

        try:
            ch.q.put_nowait(frame_i16)
        except queue.Full:
            ch.drops += 1
            try:
                _ = ch.q.get_nowait()
            except queue.Empty:
                pass
            try:
                ch.q.put_nowait(frame_i16)
            except queue.Full:
                pass

    def _mix_block(self, frames: int) -> np.ndarray:
        if not self.channels:
            return np.zeros((frames,), dtype=np.float32)

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

            gt = self.master.gate_threshold
            if gt > 0:
                rms = float(np.sqrt(np.mean(x * x) + 1e-12))
                if rms < gt:
                    x *= self.master.gate_attenuation

            x *= float(ch.gain)
            mix += x

        mix *= float(self.master.master_gain)

        ceiling = float(self.master.limiter_ceiling)
        ceiling = max(0.05, min(1.0, ceiling))
        mix = np.clip(mix, -ceiling, ceiling)

        return mix

    def _audio_callback(self, outdata, frames, time_info, status):
        block = self._mix_block(frames)
        outdata[:] = block.reshape(-1, 1)

    def start_audio(self):
        if self._audio_stream is not None:
            return
        self._audio_stream = sd.OutputStream(
            samplerate=TARGET_SR,
            channels=1,
            dtype="float32",
            blocksize=SAMPLES_PER_FRAME,
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
        payload = self.state_payload()
        dead = []
        for ws in self._ctrl_clients:
            try:
                await ws.send_json({"type": "tick", "payload": payload})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ctrl_clients.discard(ws)

    # =========================
    # WebRTC receive pipeline
    # =========================

    async def add_webrtc_peer(self, ch_id: int, offer: RTCSessionDescription) -> RTCSessionDescription:
        pc = RTCPeerConnection()
        self._pcs.add(pc)

        # resampler: qualquer coisa -> mono, 48k, s16
        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self._cleanup_pc(pc)

        @pc.on("track")
        def on_track(track):
            if track.kind != "audio":
                return

            print(f"[WebRTC] audio track recebida: ch={ch_id}")

            async def pump():
                # buffer pra formar frames fixos de 20ms
                stash = np.zeros((0,), dtype=np.int16)
                try:
                    while True:
                        frame = await track.recv()  # av.AudioFrame
                        # resample -> lista de frames (às vezes pode retornar 0..N)
                        for f in resampler.resample(frame):
                            # f é s16 mono
                            arr = f.to_ndarray()
                            # arr pode vir como (samples,) ou (1, samples); normaliza:
                            if arr.ndim == 2:
                                arr = arr[0]
                            arr = arr.astype(np.int16, copy=False)

                            if arr.size == 0:
                                continue

                            stash = np.concatenate([stash, arr])

                            while stash.size >= SAMPLES_PER_FRAME:
                                chunk = stash[:SAMPLES_PER_FRAME]
                                stash = stash[SAMPLES_PER_FRAME:]
                                self.push_audio_i16_frame(ch_id, chunk)
                except Exception as e:
                    # normalmente cai aqui quando peer fecha / track termina
                    print(f"[WebRTC] pump encerrado ch={ch_id}: {e!r}")

            task = asyncio.create_task(pump())
            self._track_tasks[pc] = task

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription

    async def _cleanup_pc(self, pc: RTCPeerConnection):
        if pc in self._track_tasks:
            t = self._track_tasks.pop(pc)
            t.cancel()
            try:
                await t
            except Exception:
                pass

        if pc in self._pcs:
            self._pcs.discard(pc)

        try:
            await pc.close()
        except Exception:
            pass

    def prune_inactive_channels(self, ttl_seconds: float = 3.0) -> None:
        now = time.time()
        dead = [ch_id for ch_id, ch in self.channels.items() if (now - ch.last_seen) > ttl_seconds]
        for ch_id in dead:
            del self.channels[ch_id]

hub = HubMixer()

# =========================
# HTTP Handlers
# =========================

async def page_client(_request: web.Request):
    return web.Response(text=CLIENT_HTML, content_type="text/html")

async def page_mixer(_request: web.Request):
    return web.Response(text=MIXER_HTML, content_type="text/html")

async def offer(request: web.Request):
    """
    Sinalização WebRTC:
    recebe {sdp, type, ch} e responde {sdp, type} (answer).
    """
    data = await request.json()
    ch_id = int(data.get("ch", 1))
    sdp = data["sdp"]
    typ = data["type"]

    offer = RTCSessionDescription(sdp=sdp, type=typ)
    answer = await hub.add_webrtc_peer(ch_id, offer)

    return web.json_response({"sdp": answer.sdp, "type": answer.type})

async def ws_ctrl(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024)
    await ws.prepare(request)

    hub._ctrl_clients.add(ws)
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
    hub.start_audio()
    try:
        while True:
            hub.prune_inactive_channels(ttl_seconds=3.0)
            await hub.broadcast_tick()
            await asyncio.sleep(0.20)
    finally:
        hub.stop_audio()

def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", page_mixer)
    app.router.add_get("/mixer", page_mixer)
    app.router.add_get("/client", page_client)
    app.router.add_post("/offer", offer)
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
