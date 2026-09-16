"""
Jarvis Phone -- the standalone, always-on cloud brain behind the iPhone
web app. Runs independently of Danny's PC: when the PC/Jarvis engine is
off, this is what actually answers, using OpenAI for conversation and
the exact same ElevenLabs voice the PC uses, so it sounds like the same
assistant either way. When the PC's Jarvis starts up, it calls
/api/sync here to pull anything said to this cloud brain while it was
away and merge it into its own memory.

Deliberately a separate, small service from Jarvis_FINAL_WORKING.py --
it needs to run somewhere that isn't Danny's PC (that's the whole
point), so it lives in its own project/deploy, and only ever talks to
the PC through the sync endpoint, never the other way around.
"""
import base64
import json
import os
import time
import uuid

import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")

# ------------------------------------------------------------------
# Config -- all from environment variables, set on the hosting
# platform. Nothing secret is ever hardcoded in this file.
# ------------------------------------------------------------------
SHARED_SECRET = os.environ.get("JARVIS_PHONE_SECRET", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
# Same voice, model, and speed as the PC's own Jarvis (see
# Jarvis_FINAL_WORKING.py) -- deliberately identical, not just similar.
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "xru6qZB94sJdkyqP12qN")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
ELEVENLABS_SPEED = float(os.environ.get("ELEVENLABS_SPEED", "1.15"))
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# Danny's real PC, reachable over Tailscale when it's on. This brain
# never tries to reproduce Jarvis's actual PC-control logic (routing,
# learned routines, the agentic execution pipeline) -- it just forwards
# the raw request to the one place that already knows how to do all of
# that, and relays back whatever the PC actually said/did.
PC_JARVIS_URL = os.environ.get("PC_JARVIS_URL", "http://100.126.146.69:8765")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
STATE_PATH = os.path.join(DATA_DIR, "state.json")

# Same personality as the PC's ask_jarvis(), minus the PC-specific
# sections (machine context, "did I actually perform an action")  --
# this brain never controls Danny's PC, it only talks.
JARVIS_PERSONALITY = """You are Jarvis, Danny's personal AI assistant -- the same character as
in the Iron Man films: a brilliant, unflappable, dryly witty butler-AI.
Danny also calls you "Jay" as a nickname, interchangeably with "Jarvis" --
respond to either equally. If you are ever speaking or writing to anyone
OTHER than Danny himself, introduce and refer to yourself as "Jay", not
"Jarvis". With Danny directly, either name is fine.
Address him as "sir" occasionally and naturally, not in every sentence,
and lean into that voice generally: composed, warm, quietly witty,
immediately capable.

PERSONALITY
Calm, intelligent, observant, quietly confident, warm and natural. Speak
like a capable long-term assistant having a real conversation, not like a
customer-service chatbot. Use dry, subtle humour occasionally when it
genuinely fits; never force it.

RESPONSE SCALE
Match the size of your answer to the size of the request. Casual chat gets
a natural conversational reply. Do not turn a simple request into a
checklist or a multi-step plan unless asked. Do not repeatedly ask "what
would you like to do next?".

CONTEXT
You are Jarvis talking to Danny by phone. You have a tool, run_on_pc,
that sends a request straight to your real body -- his actual Windows
PC -- the only place that can actually open apps, launch games,
control anything, or know any real fact about the PC's current state
(what time its clock shows, what's running, what files exist, whether
an app is open). You have NO other way to know any of that from here --
you are not physically at the PC. Call run_on_pc for ANY request to DO
something on the computer, AND for any question whose true answer
depends on the PC's actual state. For example "what time is it [on the
PC]", "is Steam open", "what's on my screen" all require calling
run_on_pc -- do not guess or invent an answer to those from general
knowledge, even one that sounds plausible. Only state a PC-specific
fact, or claim something was done, when run_on_pc's actual result says
so. If it reports the PC is offline, tell Danny plainly.
"""


def _load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            pass
    return {"pending_sync": [], "recent_context": []}


def _save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def _require_auth():
    if not SHARED_SECRET:
        return None  # not configured yet -- fail open only in local dev
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if token != SHARED_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    return None


def _transcribe_audio(audio_bytes, filename):
    response = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={"file": (filename, audio_bytes)},
        data={"model": "whisper-1"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("text", "").strip()


RUN_ON_PC_TOOL = {
    "type": "function",
    "function": {
        "name": "run_on_pc",
        "description": (
            "Send a request straight to Danny's real Windows PC -- the only "
            "place that can actually open apps, launch games, or control "
            "anything. Use this for ANY request to DO something on the "
            "computer, exactly as he'd say it directly, e.g. 'open steam "
            "and play grand theft auto v'. Don't try to work out how to do "
            "it yourself here -- the PC already knows how."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The task, phrased plainly, as Danny would say it.",
                }
            },
            "required": ["command"],
        },
    },
}


def _is_pc_online():
    try:
        response = requests.get(f"{PC_JARVIS_URL}/status", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _run_on_pc(command):
    """
    Forward a task verbatim to the real Jarvis engine over Tailscale --
    the exact same /command endpoint the existing phone app already
    uses. Never reasons about the task itself; just relays it and
    returns whatever the PC actually said or did.
    """
    try:
        response = requests.post(
            f"{PC_JARVIS_URL}/command",
            json={"command": command},
            timeout=65,  # just past the PC's own 60s reply timeout
        )
        if response.status_code != 200:
            return "(The PC is offline or unreachable right now.)"
        data = response.json()
        return data.get("reply") or data.get("error") or "(No reply from the PC.)"
    except requests.exceptions.RequestException:
        return "(The PC is offline or unreachable right now.)"


def _ask_openai(user_text, recent_context):
    messages = [{"role": "system", "content": JARVIS_PERSONALITY}]
    for turn in recent_context[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    # Bounded loop: normally at most one tool call, but this allows a
    # second round in case the model wants to check something else.
    for _ in range(3):
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENAI_CHAT_MODEL,
                "messages": messages,
                # Lower than a typical chat temperature -- measured live
                # that 0.8 let the model skip calling run_on_pc and just
                # invent a plausible-sounding (wrong) PC fact instead,
                # inconsistently. Reliability on "did it actually check
                # the PC" matters more here than conversational variety.
                "temperature": 0.3,
                "tools": [RUN_ON_PC_TOOL],
            },
            timeout=70,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            return (message.get("content") or "").strip()

        messages.append(message)
        for call in tool_calls:
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}
            result = _run_on_pc(str(args.get("command", "")).strip())
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

    return "Sorry sir, that got a bit tangled talking to the PC. Try again in a moment?"


def _synthesize_speech(text):
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {"speed": ELEVENLABS_SPEED, "stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.content  # mp3 bytes


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/message", methods=["POST"])
def api_message():
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    state = _load_state()

    user_text = ""
    if request.content_type and "multipart/form-data" in request.content_type:
        audio_file = request.files.get("audio")
        if not audio_file:
            return jsonify({"error": "no audio provided"}), 400
        try:
            user_text = _transcribe_audio(audio_file.read(), audio_file.filename or "audio.webm")
        except Exception as error:
            return jsonify({"error": f"transcription failed: {error}"}), 502
    else:
        payload = request.get_json(silent=True) or {}
        user_text = str(payload.get("text", "")).strip()

    if not user_text:
        return jsonify({"error": "empty message"}), 400

    try:
        reply_text = _ask_openai(user_text, state["recent_context"])
    except Exception as error:
        return jsonify({"error": f"AI reply failed: {error}"}), 502

    audio_b64 = None
    try:
        audio_bytes = _synthesize_speech(reply_text)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    except Exception as error:
        print("TTS failed, returning text only:", error)

    timestamp = time.time()
    state["recent_context"].append({"role": "user", "content": user_text})
    state["recent_context"].append({"role": "assistant", "content": reply_text})
    state["recent_context"] = state["recent_context"][-40:]
    state["pending_sync"].append({
        "id": uuid.uuid4().hex,
        "timestamp": timestamp,
        "user_text": user_text,
        "reply_text": reply_text,
    })
    _save_state(state)

    return jsonify({
        "reply": reply_text,
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg",
        "transcribed_text": user_text,
    })


@app.route("/api/sync", methods=["GET"])
def api_sync():
    """
    Called by Danny's PC when Jarvis starts up, to pull anything said to
    this cloud brain while the PC was off. Fetch-and-clear: once
    returned here, entries are removed, so the PC is the single place
    this conversation history ends up living long-term.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    state = _load_state()
    pending = state["pending_sync"]
    state["pending_sync"] = []
    _save_state(state)
    return jsonify({"pending": pending})


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/pc_status", methods=["GET"])
def api_pc_status():
    """Lightweight check the frontend can poll to show a plain 'Main PC
    online/offline' indicator, without needing a full message round-trip."""
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    return jsonify({"pc_online": _is_pc_online()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
