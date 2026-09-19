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
import re
import threading
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
# Danny's PC lives on a private Tailscale address that this backend,
# running on Render, has no route to -- Render was never on that
# tailnet, and Tailscale Funnel (which would fix that) turned out to
# need a paid plan. So this never calls the PC directly. Instead the
# PC and its launcher (jarvis_launcher.py) poll THIS backend every few
# seconds (see /api/pc_poll and /api/launcher_poll below), the same
# "poll and push" shape already used for pc_events. This brain still
# never reproduces Jarvis's actual PC-control logic; it just queues the
# raw request for the PC to run and relays back whatever it said/did.
PC_ONLINE_THRESHOLD_SECONDS = 12  # the PC polls roughly every 3s

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
    state = {
        "pending_sync": [],
        "recent_context": [],
        "pc_events": [],
        "pending_commands": [],
        "command_results": {},
        "pc_last_seen": 0,
        "start_requested": False,
        "mirror_events": [],
        "pending_mirror_messages": [],
        "mirror_reset_requested": False,
        # Reachable regardless of the PC's power state -- see
        # _pull_shared_memory_from_phone/_push_shared_memory_to_phone in
        # Jarvis_FINAL_WORKING.py. The PC's own jarvis_memory.json stays
        # the durable long-term archive; this is the always-reachable
        # mailbox/cache re-seeded from it, and the only copy this app can
        # read from when the PC is off.
        "shared_memory": {"facts": {}, "reminders": []},
    }
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as handle:
                state.update(json.load(handle))
        except Exception:
            pass
    return state


def _save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


# Guards read-modify-write access to state.json -- the PC can push a
# pc_said event at the same moment the phone is mid-conversation, and
# without this a fast enough race could silently drop one write.
_state_lock = threading.Lock()


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


# Checked BEFORE the AI model ever sees the message, same reasoning as
# the old code-mode check this replaced: confirmed live, repeatedly,
# that even an explicit system-prompt instruction doesn't reliably make
# a model call a tool for this instead of just answering conversationally
# with a convincing fake "sure, I'm on it" -- too important to a repair
# fallback to leave to a probabilistic judgment call.
_MIRROR_TRIGGERS = (
    "fix jarvis", "fix desktop jarvis", "repair jarvis",
    "switch to code", "lets switch to code", "let's switch to code",
    "do this in code", "do that in code", "lets do this in code", "let's do this in code",
)


def _looks_like_mirror_request(text):
    lowered = text.lower()
    return any(trigger in lowered for trigger in _MIRROR_TRIGGERS)


# Same reasoning, same model, as the PC's research_topic_online() in
# Jarvis_FINAL_WORKING.py: gpt-4o-mini via chat completions (what
# _ask_openai uses) has no web access at all, so a real research request
# needs OpenAI's Responses API and its hosted web_search tool instead.
# Only used when the PC is offline -- when it's online, _run_on_pc already
# reaches the PC's own (now web-search-capable) conversational brain.
OPENAI_RESEARCH_MODEL = os.environ.get("JARVIS_OPENAI_MODEL", "gpt-5-mini")

_RESEARCH_TRIGGERS = (
    "search for ", "can you search for ", "could you search for ",
    "find me ", "look up ", "can you look up ", "could you look up ",
    "search the internet for ", "search online for ", "search the web for ",
    "research ", "can you research ", "could you research ", "please research ",
    "look into ", "can you look into ", "could you look into ",
    "give me ideas for ", "give me some ideas for ", "give me some ideas about ",
    "what are some ideas for ", "what ideas do you have for ",
)


def _looks_like_research_request(text):
    lowered = text.lower().strip()
    return any(lowered.startswith(trigger) for trigger in _RESEARCH_TRIGGERS)


# Mirrors the PC's own "remember that X is Y" parsing in
# Jarvis_FINAL_WORKING.py (remember_fact and its prefixes), so a fact
# stated to the phone while the PC is offline is saved into
# shared_memory rather than lost -- the PC picks it up on its next
# startup pull, or immediately if it's already running (see
# _pull_shared_memory_from_phone / _push_shared_memory_to_phone). Only
# used when the PC is offline; when it's online the message is relayed
# there instead, and the PC's own remember_fact runs (see api_message).
_REMEMBER_PREFIXES = ("remember that ", "remember ", "jarvis remember that ", "jarvis remember ")


def _try_save_remembered_fact(user_text):
    lowered = user_text.strip().lower()
    for prefix in _REMEMBER_PREFIXES:
        if lowered.startswith(prefix):
            memory_text = user_text.strip()[len(prefix):].strip().rstrip(".?!")
            if not memory_text:
                return "What would you like me to remember, sir?"

            fact_match = re.match(r"(?:that\s+)?(?:my\s+)?(.+?)\s+is\s+(.+)$", memory_text, re.IGNORECASE)
            with _state_lock:
                state = _load_state()
                shared = state.setdefault("shared_memory", {"facts": {}, "reminders": []})
                facts = shared.setdefault("facts", {})
                if fact_match:
                    key = fact_match.group(1).strip().lower()
                    if key.startswith("my "):
                        key = key[3:].strip()
                    value = fact_match.group(2).strip()
                    if key and value:
                        facts[key] = value
                        _save_state(state)
                        return f"Got it. I'll remember that your {key} is {value}."
                key = f"memory_{len(facts) + 1}"
                facts[key] = memory_text
                _save_state(state)
                return "Got it. I'll remember that."
    return None


def _research_online(question):
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENAI_RESEARCH_MODEL,
                "instructions": (
                    "You are Jarvis, researching something for Danny using live "
                    "web search. Give a clear, conversational, spoken-style answer "
                    "-- not a report, no markdown, no headings or bullet lists. Get "
                    "to the actual answer or ideas quickly, mention anything "
                    "genuinely current or uncertain, and keep it under 200 words "
                    "unless the question clearly calls for more."
                ),
                "tools": [{"type": "web_search"}],
                "input": question,
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text = (content.get("text") or "").strip()
                        if text:
                            return text
        return None
    except Exception as error:
        print("Web research error:", error)
        return None


def _is_pc_online():
    with _state_lock:
        state = _load_state()
        last_seen = state.get("pc_last_seen", 0)
    return (time.time() - last_seen) < PC_ONLINE_THRESHOLD_SECONDS


def _run_on_pc(command):
    """
    Queue a task for the PC to pick up on its next poll (see
    /api/pc_poll) and wait here for the result to land in
    /api/pc_command_result. Never reasons about the task itself; just
    relays it and returns whatever the PC actually said or did.
    """
    if not _is_pc_online():
        return "(The PC is offline or unreachable right now.)"

    command_id = uuid.uuid4().hex
    with _state_lock:
        state = _load_state()
        state["pending_commands"].append({"id": command_id, "command": command})
        _save_state(state)

    deadline = time.time() + 45  # a bit under the phone's own 70s OpenAI timeout
    while time.time() < deadline:
        time.sleep(1)
        with _state_lock:
            state = _load_state()
            if command_id in state["command_results"]:
                reply = state["command_results"].pop(command_id)
                _save_state(state)
                return reply

    return "(The PC took too long to reply.)"


def _ask_openai(user_text, recent_context, shared_facts=None):
    system_content = JARVIS_PERSONALITY
    if shared_facts:
        memory_lines = "\n".join(f"{key}: {value}" for key, value in shared_facts.items())
        system_content += f"\n\nSAVED LONG-TERM MEMORY (recall these naturally when asked):\n{memory_lines}"
    messages = [{"role": "system", "content": system_content}]
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
            # Deliberately ignore whatever "command" argument the model
            # generated and send Danny's actual words instead -- found
            # live that for an unusual/compound instruction ("open code
            # and find out why we can't close jarvis remotely"), the
            # model doesn't always reproduce it verbatim like the tool
            # description asks; it can paraphrase or garble it (one
            # real case collapsed to just "open close"). The PC's own
            # command handling already parses natural phrasing fine, so
            # there's no reason to let the model rewrite it at all.
            result = _run_on_pc(user_text)
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

    # A request to fix/rebuild Jarvis needs the separate mirror page,
    # not a reply from either brain here -- it's a live mirror of the
    # actual dev conversation with Claude, independent of the PC
    # engine's health (which will likely be unstable during exactly
    # this conversation). See /api/mirror_message and static/mirror.html.
    if _looks_like_mirror_request(user_text):
        return jsonify({
            "reply": "Opening the code mirror for you, sir.",
            "open_mirror": True,
            "transcribed_text": user_text,
        })

    # Deterministic routing, not a model's judgment call -- confirmed live,
    # repeatedly (see _looks_like_mirror_request above), that an LLM given a
    # tool and told when to use it will sometimes just skip it and invent a
    # plausible-sounding answer instead. That's exactly what was causing the
    # missing replies / conflicting text: _ask_openai's model would
    # sometimes decide not to call run_on_pc even while the PC was right
    # there and online. Now the PC is simply always the source of truth
    # when it's reachable, with no judgment call involved.
    source = "cloud"
    if _is_pc_online():
        reply_text = _run_on_pc(user_text)
        source = "pc"
    else:
        remembered_reply = _try_save_remembered_fact(user_text)
        if remembered_reply:
            reply_text = remembered_reply
            source = "memory"
        elif _looks_like_research_request(user_text):
            answer = _research_online(user_text)
            reply_text = answer or "I couldn't find anything useful on that just now, sir."
            source = "research"
        else:
            try:
                shared_facts = state.get("shared_memory", {}).get("facts", {})
                reply_text = _ask_openai(user_text, state["recent_context"], shared_facts)
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
        "source": source,
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


@app.route("/api/memory", methods=["GET"])
def api_memory():
    """
    Read-only view of shared_memory -- called by the PC on startup (see
    _pull_shared_memory_from_phone) to merge in anything remembered via
    the phone while it was off, and used here in _ask_openai so this
    app's own fallback brain can recall the same facts when the PC is
    offline.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    state = _load_state()
    return jsonify(state.get("shared_memory", {"facts": {}, "reminders": []}))


@app.route("/api/memory_push", methods=["POST"])
def api_memory_push():
    """
    Called by the PC (see _push_shared_memory_to_phone) every time its
    own jarvis_memory.json is saved, so this backend always has a
    reasonably current copy reachable even while the PC is off. The PC's
    values win for any key it sends -- it's the durable long-term
    archive; this is just the always-reachable mirror of it.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    payload = request.get_json(silent=True) or {}
    with _state_lock:
        state = _load_state()
        shared = state.setdefault("shared_memory", {"facts": {}, "reminders": []})
        shared.setdefault("facts", {}).update(payload.get("facts") or {})
        if "reminders" in payload:
            shared["reminders"] = payload["reminders"]
        _save_state(state)
    return jsonify({"ok": True})


@app.route("/api/speak_text", methods=["POST"])
def api_speak_text():
    """
    Synthesize speech for a literal piece of text with no OpenAI call at
    all -- used for pc_events, which are already-final text Jarvis said
    on the PC. Kept separate from /api/message deliberately: those
    events shouldn't get "replied to" by the conversational brain, just
    spoken verbatim, and only ever synthesized on demand when a phone
    is actually there to hear it, not the instant the PC says it.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    try:
        audio_bytes = _synthesize_speech(text)
    except requests.exceptions.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        return jsonify({"error": f"TTS failed: {error}", "detail": detail}), 502
    except Exception as error:
        return jsonify({"error": f"TTS failed: {error}"}), 502

    return jsonify({
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "audio_mime": "audio/mpeg",
    })


@app.route("/api/pc_said", methods=["POST"])
def api_pc_said():
    """
    The PC pushes here every time Jarvis says ANYTHING (see the
    _relay_speech_to_phone hook on say() in Jarvis_FINAL_WORKING.py) --
    not just replies to something the phone asked. This is what lets
    the phone hear self-repair progress, reassurance check-ins, and
    routine completions that happen entirely on their own on the PC.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    with _state_lock:
        state = _load_state()
        state["pc_events"].append({
            "id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "text": text,
        })
        # Bounded so this can never grow unbounded if the phone app
        # isn't open to drain it for a long stretch.
        state["pc_events"] = state["pc_events"][-100:]
        _save_state(state)

    return jsonify({"status": "ok"})


@app.route("/api/pc_events", methods=["GET"])
def api_pc_events():
    """
    The phone polls this every couple of seconds. `since` is the id of
    the last event it already has (or empty, for a first call); returns
    only events after that point so nothing gets spoken twice.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    since = request.args.get("since", "")
    with _state_lock:
        state = _load_state()
        events = state["pc_events"]

    if not since:
        # First call from a fresh page load -- don't dump potentially
        # minutes of backlog at once, just start listening from now.
        return jsonify({"events": [], "last_id": events[-1]["id"] if events else ""})

    ids = [e["id"] for e in events]
    if since in ids:
        new_events = events[ids.index(since) + 1:]
    else:
        # The since id aged out of the last 100 -- just resume from now
        # rather than guessing how far back to go.
        new_events = []

    last_id = new_events[-1]["id"] if new_events else since
    return jsonify({"events": new_events, "last_id": last_id})


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


@app.route("/api/start_pc", methods=["POST"])
def api_start_pc():
    """
    Flags a start request for the PC's launcher (jarvis_launcher.py) to
    pick up on its next poll (see /api/launcher_poll) -- can't call the
    launcher directly for the same reason /api/pc_poll exists: this
    backend has no route to the PC's private Tailscale address. Only
    works if the PC itself is on/reachable; this does not power the
    machine on. For now it always starts the terminal version; the
    desktop HUD version will be added later.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    with _state_lock:
        state = _load_state()
        state["start_requested"] = True
        _save_state(state)

    return jsonify({"status": "requested"})


@app.route("/api/pc_poll", methods=["POST"])
def api_pc_poll():
    """
    Jarvis's own engine calls this every few seconds -- both a
    heartbeat (so /api/pc_status knows it's alive) and how it picks up
    anything queued for it by run_on_pc, since this backend can't reach
    the PC directly over its private Tailscale address. Fetch-and-clear:
    once handed out, a command isn't handed out again on the next poll.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    with _state_lock:
        state = _load_state()
        state["pc_last_seen"] = time.time()
        pending = state["pending_commands"]
        state["pending_commands"] = []
        _save_state(state)

    return jsonify({"pending_commands": pending})


@app.route("/api/pc_command_result", methods=["POST"])
def api_pc_command_result():
    """The PC posts back here once it's actually run a task /api/pc_poll handed it."""
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    payload = request.get_json(silent=True) or {}
    command_id = str(payload.get("id", "")).strip()
    if not command_id:
        return jsonify({"error": "missing id"}), 400

    with _state_lock:
        state = _load_state()
        state["command_results"][command_id] = str(payload.get("reply", ""))
        if len(state["command_results"]) > 200:
            state["command_results"] = dict(list(state["command_results"].items())[-200:])
        _save_state(state)

    return jsonify({"status": "ok"})


@app.route("/api/launcher_poll", methods=["POST"])
def api_launcher_poll():
    """
    jarvis_launcher.py (a separate always-on listener on the PC) polls
    this to see if a start was requested via /api/start_pc -- same
    reasoning as pc_poll, just for starting Jarvis instead of running a
    command on it. Fetch-and-clear so it's only acted on once.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    with _state_lock:
        state = _load_state()
        requested = state.get("start_requested", False)
        state["start_requested"] = False
        _save_state(state)

    return jsonify({"start_requested": requested})


@app.route("/api/mirror_message", methods=["POST"])
def api_mirror_message():
    """
    The mirror page (static/mirror.html) sends typed replies here.
    Queued for the launcher's own poll (jarvis_launcher.py, independent
    of the Jarvis engine) to pick up and run -- fire-and-forget, since a
    mirror reply can take a while (it forks the whole conversation's
    history) and arrives separately through /api/mirror_events as the
    model streams it, not as one bounded response to this request.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    with _state_lock:
        state = _load_state()
        state["pending_mirror_messages"].append(text)
        _save_state(state)

    return jsonify({"status": "queued"})


@app.route("/api/mirror_reset", methods=["POST"])
def api_mirror_reset():
    """
    Called when the mirror page loads fresh -- starts the next message
    from the designated live dev session again instead of continuing
    wherever a previous mirror conversation left off, so opening the
    page for a new problem doesn't drag in an old, unrelated one.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    with _state_lock:
        state = _load_state()
        state["mirror_reset_requested"] = True
        state["mirror_events"] = []
        _save_state(state)

    return jsonify({"status": "ok"})


@app.route("/api/mirror_poll", methods=["POST"])
def api_mirror_poll():
    """jarvis_launcher.py polls this every few seconds for pending mirror
    messages and a reset flag. Fetch-and-clear, same pattern as pc_poll."""
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    with _state_lock:
        state = _load_state()
        messages = state["pending_mirror_messages"]
        state["pending_mirror_messages"] = []
        reset = state.get("mirror_reset_requested", False)
        state["mirror_reset_requested"] = False
        _save_state(state)

    return jsonify({"messages": messages, "reset": reset})


@app.route("/api/mirror_said", methods=["POST"])
def api_mirror_said():
    """The launcher pushes the mirror conversation's text here -- tool-use
    markers and full replies. Never spoken; the mirror page displays it."""
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    with _state_lock:
        state = _load_state()
        state["mirror_events"].append({"id": uuid.uuid4().hex, "timestamp": time.time(), "text": text})
        state["mirror_events"] = state["mirror_events"][-300:]
        _save_state(state)

    return jsonify({"status": "ok"})


@app.route("/api/mirror_events", methods=["GET"])
def api_mirror_events():
    """Same incremental since-cursor polling as /api/pc_events, for the
    mirror conversation's text feed."""
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    since = request.args.get("since", "")
    with _state_lock:
        state = _load_state()
        events = state["mirror_events"]

    if not since:
        return jsonify({"events": [], "last_id": events[-1]["id"] if events else ""})

    ids = [e["id"] for e in events]
    if since in ids:
        new_events = events[ids.index(since) + 1:]
    else:
        new_events = []

    last_id = new_events[-1]["id"] if new_events else since
    return jsonify({"events": new_events, "last_id": last_id})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
