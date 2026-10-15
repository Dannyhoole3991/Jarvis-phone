"""
Jarvis Phone -- the standalone, always-on cloud brain behind the iPhone
web app. Runs independently of Danny's PC, with the exact same
ElevenLabs voice the PC uses, so it sounds like the same assistant.

The app has tabs:
  Chat  -- the quick brain (OpenAI), answers in a second or two and never
           waits on the PC. Knows the dashboard copy the PC pushes.
  PC    -- talks to the PC's own Jarvis (slower, but it can act): the
           request is queued for the PC's next poll and everything the PC
           says back (progress and answer) arrives through pc_events.
  Money, Store, Ads, Business, Trading, Usage, Needs you -- read-only
           views of that dashboard copy (/api/dashboard).
When the PC's Jarvis starts up, it calls /api/sync here to pull anything
said to the Chat brain and merge it into its own memory.

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

try:
    import tapo_plug
except Exception:
    # tapo_plug imports the `tapo` package, which isn't installed on
    # Render (only on the home server, which has real LAN access to the
    # plug) -- fails open with power-on simply unavailable there, rather
    # than crashing the whole app on an unrelated deploy.
    tapo_plug = None

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
# The Chat tab's brain. gpt-5-mini with minimal reasoning measured ~1s per reply
# (2026-10-15), faster than gpt-4o-mini's ~1.9s and sharper.
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5-mini")
OPENAI_CHAT_REASONING = os.environ.get("OPENAI_CHAT_REASONING", "minimal")
# How long the PC tab keeps a job open before treating it as abandoned.
PC_JOB_MAX_SECONDS = 15 * 60
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
You are Jarvis talking to Danny on his phone, in the app's Chat tab: the
quick brain, built for fast answers. Replies are spoken aloud, so keep them
to one to three short, natural sentences, with no markdown, lists or emoji.
Give the answer and stop: don't tack on offers or "would you like me to"
questions, and never offer anything you can't do from here (you can't set
reminders, schedule things, post, or change anything).

You can NOT control or check his PC from this tab: you can't open apps,
post videos, read his screen or see anything live on the computer. If he
asks for something like that, say in one short sentence that it needs the
PC tab (at the top of the app), where his PC does it directly. Never
pretend you did something on the PC.

You DO have a recent copy of his dashboard below (his store orders and
payouts, money, TikTok and adverts, business, trading, Claude usage and his
"needs you" jobs), sent over from the PC every few minutes. Answer from it
when he asks about those, and say how old it is if that matters (for
example "as of ten minutes ago"). Never invent a number that isn't in it.
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
        # PC tab jobs: {id: {"command", "queued_at", "done", "reply"}} -- see _queue_pc_job.
        "pc_jobs": {},
        # The desktop dashboard copy the PC pushes (see /api/dashboard_push).
        "dashboard": None,
        "dashboard_at": 0,
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
# Jarvis_FINAL_WORKING.py: the Chat brain (plain chat completions) has no
# web access at all, so a real research request needs OpenAI's Responses
# API and its hosted web_search tool instead. Used by the Chat tab for
# "search for / look up / research ..." requests.
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


# Same deterministic-phrase-before-model reasoning as the mirror/research
# checks above -- this one matters even more, since it's the one path
# that can actually act on real hardware (mains power to Danny's PC).
_POWER_ON_TRIGGERS = (
    "turn my pc on", "turn on my pc", "power on my pc", "power my pc on",
    "turn the pc on", "turn on the pc", "switch my pc on", "switch on my pc",
    "turn my computer on", "turn on my computer", "power on my computer",
    "boot my pc", "boot up my pc", "boot my pc up", "start my pc up",
    "wake my pc up", "wake up my pc",
)


def _looks_like_power_on_request(text):
    lowered = text.lower().strip().rstrip(".?!")
    return any(trigger in lowered for trigger in _POWER_ON_TRIGGERS)


def _handle_power_on_request():
    """
    Turns the PC's smart plug on (harmless no-op if it's already on) and
    queues a start request for jarvis_launcher.py to pick up once Windows
    has booted (see /api/launcher_poll) -- so "turn my PC on" also gets
    Jarvis itself running on it, not just the hardware powered up.
    """
    if tapo_plug is None:
        return "I can't reach the smart plug from here, sir -- that only works from the home server."
    try:
        status = tapo_plug.get_status()
        already_on = bool(status.get("device_on"))
        if not already_on:
            tapo_plug.turn_on()
    except Exception as error:
        return f"I couldn't reach the smart plug just now, sir: {error}"

    with _state_lock:
        state = _load_state()
        state["start_requested"] = True
        _save_state(state)

    if already_on:
        return "Your PC's already on, sir -- I've queued Jarvis to start up on it now."
    return "Turning your PC on now, sir. Give it a minute or two to boot, and Jarvis will start itself up on it."


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


def _queue_pc_job(command):
    """
    PC tab: hand the request to the PC (it picks it up on its next poll, about a
    second, see /api/pc_poll) and return at once. Everything the PC then says,
    progress and the final answer, reaches the phone through pc_events, and the
    phone asks /api/pc_job/<id> whether it has finished.
    """
    job_id = uuid.uuid4().hex
    now = time.time()
    with _state_lock:
        state = _load_state()
        state["pending_commands"].append({"id": job_id, "command": command})
        jobs = state.setdefault("pc_jobs", {})
        jobs[job_id] = {"command": command, "queued_at": now, "done": False, "reply": None}
        for old in [k for k, j in jobs.items() if now - j.get("queued_at", now) > PC_JOB_MAX_SECONDS]:
            jobs.pop(old, None)
        _save_state(state)
    return job_id


def _ago(seconds):
    seconds = max(0, int(seconds))
    if seconds < 90:
        return "just now"
    if seconds < 5400:
        return f"{seconds // 60} minutes ago"
    if seconds < 172800:
        return f"{seconds // 3600} hours ago"
    return f"{seconds // 86400} days ago"


def _dashboard_summary(dashboard, pushed_at, pc_online):
    """The PC's dashboard copy as compact text for the Chat brain. Every line is optional."""
    if not dashboard:
        return "No dashboard copy has arrived from the PC yet."
    out = [f"Dashboard copy from the PC, sent {_ago(time.time() - (pushed_at or 0))}"
           + ("" if pc_online else " (the PC is off right now, so this is the last copy)") + "."]

    def section(name, fn):
        try:
            lines = fn(dashboard.get(name) or {})
            if lines:
                out.extend(lines)
        except Exception:
            pass

    def store(s):
        lines = []
        orders = (s.get("orders") or {})
        listed = orders.get("orders") or []
        if orders.get("ok") or listed:
            lines.append(f"Shopify orders ({len(listed)}):" if listed else "Shopify orders: none yet.")
            for o in listed[:15]:
                lines.append(f"  {o.get('number')} {o.get('date')}, {o.get('customer')}, {o.get('total')}, "
                             f"{o.get('payment')}, fulfilment: {o.get('fulfilment') or 'not started'}")
        pay = orders.get("payouts") or {}
        if pay.get("pending"):
            dates = ", ".join(f"{t.get('order')} {t.get('net')} on {t.get('payout_date')}" for t in pay.get("transactions") or [])
            lines.append(f"Pending Shopify payout: {pay['pending']} ({dates}). Paid to his bank automatically.")
        products = s.get("products") or []
        if products:
            lines.append(f"Store: {len(products)} live products, {sum(1 for p in products if p.get('christmas'))} in the Christmas range.")
        return lines

    def money(m):
        return [f"Money check ({_ago(time.time() - m['at'])}): {m['text'][:700]}"] if m.get("text") and m.get("at") else []

    def ads(a):
        t = a.get("tiktok") or {}
        if not t.get("ok"):
            return []
        metrics = ", ".join(f"{k.replace('_', ' ')} {v.get('value')}" for k, v in (t.get("metrics") or {}).items() if isinstance(v, dict))
        top = sorted(t.get("videos") or [], key=lambda v: v.get("views") or 0, reverse=True)[:3]
        lines = [f"TikTok (last 7 days): {metrics}."]
        if top:
            lines.append("Top TikTok videos: " + "; ".join(f"{(v.get('caption') or '')[:50]} ({v.get('views')} views)" for v in top))
        return lines

    def posts(p):
        items = p.get("posts") or []
        if not items:
            return []
        ready = sum(1 for i in items if i.get("status") == "ready")
        posted = sum(1 for i in items if i.get("status") == "posted")
        return [f"TikTok posting list: {ready} ready to post, {posted} posted (Danny posts them from the desktop Advertising tab)."]

    def business(b):
        keys = [k for k in ("revenue", "cost", "profit", "target") if b.get(k) is not None]
        return ["Business engine: " + ", ".join(f"{k} {b[k]}" for k in keys)] if keys else []

    def trading(t):
        bits = [f"mode {t['mode']}" if t.get("mode") else "", f"{t['tested']} versions tested" if t.get("tested") else "",
                t.get("forward_note") or ""]
        bits = [b for b in bits if b]
        return ["Trading research lab (paper only, no live trading): " + "; ".join(bits)] if bits else []

    def usage(st):
        windows = (st.get("usage") or {}).get("windows") or {}
        names = {"five_hour": "5-hour session", "seven_day": "week"}
        parts = []
        for k, w in windows.items():
            u = w.get("utilization")
            if isinstance(u, (int, float)):
                parts.append(f"{names.get(k, k.replace('_', ' '))} {u * 100 if u <= 1 else u:.0f}% used")
        return ["Claude usage: " + ", ".join(parts)] if parts else []

    def needs(n):
        return [f"NEEDS DANNY (his to-do jobs):\n{n['text'][:1500]}"] if n.get("text") else []

    section("store", store)
    section("money", money)
    section("advertising", ads)
    section("posts", posts)
    section("business", business)
    section("trading", trading)
    section("state", usage)
    section("needs_you", needs)
    return "\n".join(out)


def _ask_chat_brain(user_text, recent_context, shared_facts=None, dashboard_text=""):
    """The Chat tab: one quick OpenAI call, no tools, never waits on the PC."""
    system_content = JARVIS_PERSONALITY
    if shared_facts:
        memory_lines = "\n".join(f"{key}: {value}" for key, value in shared_facts.items())
        system_content += f"\n\nSAVED LONG-TERM MEMORY (recall these naturally when asked):\n{memory_lines}"
    system_content += "\n\nNow: " + time.strftime("%A %d %B %Y, %H:%M") + " (UK time on the server may differ slightly)."
    if dashboard_text:
        system_content += "\n\n" + dashboard_text
    messages = [{"role": "system", "content": system_content}]
    for turn in recent_context[-12:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})
    body = {"model": OPENAI_CHAT_MODEL, "messages": messages}
    if OPENAI_CHAT_MODEL.startswith(("gpt-5", "o")):
        body["reasoning_effort"] = OPENAI_CHAT_REASONING
    else:
        body["temperature"] = 0.5
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=40,
    )
    response.raise_for_status()
    return (response.json()["choices"][0]["message"].get("content") or "").strip()


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

    # Which tab sent it. "pc" (the PC tab): hand it straight to the PC and return
    # at once -- the PC's progress and answer arrive through pc_events. Anything
    # else (the Chat tab): the quick cloud brain answers, never waiting on the PC.
    mode = (request.form.get("mode") if request.form else None) or (request.get_json(silent=True) or {}).get("mode") or "chat"

    if mode == "pc":
        if not _is_pc_online():
            return jsonify({
                "reply": "Your PC's offline, sir, so I can't reach it. Say \"turn my PC on\" in the Chat tab, or use the Start Jarvis button.",
                "transcribed_text": user_text, "source": "pc_offline",
            })
        return jsonify({"queued": _queue_pc_job(user_text), "transcribed_text": user_text, "source": "pc"})

    if _looks_like_power_on_request(user_text):
        reply_text = _handle_power_on_request()
        source = "power"
    elif (remembered_reply := _try_save_remembered_fact(user_text)):
        reply_text = remembered_reply
        source = "memory"
    elif _looks_like_research_request(user_text):
        answer = _research_online(user_text)
        reply_text = answer or "I couldn't find anything useful on that just now, sir."
        source = "research"
    else:
        try:
            dashboard_text = _dashboard_summary(state.get("dashboard"), state.get("dashboard_at"), _is_pc_online())
            reply_text = _ask_chat_brain(user_text, state["recent_context"], state.get("shared_memory", {}).get("facts", {}),
                                         dashboard_text)
            source = "cloud"
        except Exception as error:
            return jsonify({"error": f"AI reply failed: {error}"}), 502

    audio_b64 = None
    try:
        audio_bytes = _synthesize_speech(reply_text)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    except Exception as error:
        print("TTS failed, returning text only:", error)

    # Re-read under the lock and change only what this exchange adds: the state
    # loaded at the top is seconds old by now, and saving it whole would wipe
    # anything the PC wrote meanwhile (a pc_said event, its poll heartbeat).
    with _state_lock:
        fresh = _load_state()
        fresh["recent_context"] = (fresh["recent_context"] + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply_text},
        ])[-40:]
        # The PC gets these through /api/sync, so its own memory knows what
        # was said on the phone's Chat tab.
        fresh["pending_sync"].append({
            "id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "user_text": user_text,
            "reply_text": reply_text,
        })
        _save_state(fresh)

    return jsonify({
        "reply": reply_text,
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg",
        "transcribed_text": user_text,
        "source": source,
    })


@app.route("/api/pc_job/<job_id>", methods=["GET"])
def api_pc_job(job_id):
    """The PC tab asks this whether its request has finished (the answer itself
    arrives as a pc_event, so it's spoken in order with the PC's progress)."""
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    with _state_lock:
        job = _load_state().get("pc_jobs", {}).get(job_id)
    if not job:
        return jsonify({"done": True, "missing": True})
    expired = time.time() - job.get("queued_at", 0) > PC_JOB_MAX_SECONDS
    return jsonify({"done": bool(job.get("done")) or expired, "reply": job.get("reply"),
                    "seconds": int(time.time() - job.get("queued_at", time.time()))})


@app.route("/api/dashboard_push", methods=["POST"])
def api_dashboard_push():
    """The PC sends a copy of its desktop dashboard here every few minutes
    (jarvis_v2/phone_sync.py) for the phone's info tabs and the Chat brain."""
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    if (request.content_length or 0) > 1_500_000:
        return jsonify({"error": "too large"}), 413
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "expected a JSON object"}), 400
    with _state_lock:
        state = _load_state()
        state["dashboard"] = payload
        state["dashboard_at"] = time.time()
        _save_state(state)
    return jsonify({"status": "ok"})


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    with _state_lock:
        state = _load_state()
    return jsonify({"dashboard": state.get("dashboard"), "at": state.get("dashboard_at") or 0,
                    "pc_online": _is_pc_online(), "now": time.time()})


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
        # The PC posts a job's answer twice, as speech (here) and as the job's
        # result (/api/pc_command_result), in either order. Same text within
        # 20 seconds is the same message, so it's only queued once.
        if any(e.get("text") == text and time.time() - e.get("timestamp", 0) < 20 for e in state["pc_events"][-10:]):
            return jsonify({"status": "duplicate"})
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
        # Confirmed live: returning events:[] here while still advancing
        # last_id to the CURRENT latest event is a real race -- if no
        # event existed yet at this exact poll, last_id stays "", so the
        # client's cursor never actually moves off empty. The next poll
        # that happens to land the instant a real event appears then
        # hits this exact branch again, silently marking that event as
        # "already seen" without ever showing it. Simplest fix that
        # can't lose anything: just return whatever's already here. The
        # event lists are already capped (see below), so this is never
        # an unbounded backlog dump.
        return jsonify({"events": events, "last_id": events[-1]["id"] if events else ""})

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
    backend has no route to the PC's private Tailscale address.

    Also best-effort turns the PC's smart plug on first (see tapo_plug.py)
    -- harmless no-op if it's already on/reachable, and it's the only way
    "open Jarvis" can actually work when the PC is fully powered off
    rather than just idle: jarvis_launcher.py itself can't poll for this
    request at all until Windows has booted, so getting power to it is a
    prerequisite, not an alternative, to the flag below.
    """
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    if tapo_plug is not None:
        try:
            tapo_plug.turn_on()
        except Exception as error:
            print("start_pc: could not reach the smart plug (continuing anyway):", error)

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

    reply = str(payload.get("reply", ""))
    with _state_lock:
        state = _load_state()
        state["command_results"][command_id] = reply
        if len(state["command_results"]) > 200:
            state["command_results"] = dict(list(state["command_results"].items())[-200:])
        job = state.setdefault("pc_jobs", {}).get(command_id)
        if job is not None:
            job.update(done=True, reply=reply, finished_at=time.time())
            # The PC normally SAYS its answer, which already reached the phone as a
            # pc_event. If it didn't (e.g. "Jarvis took too long to reply."), add it,
            # so the PC tab always gets an answer -- and never the same one twice.
            said = {e.get("text", "").strip() for e in state["pc_events"] if e.get("timestamp", 0) >= job.get("queued_at", 0)}
            if reply.strip() and reply.strip() not in said:
                state["pc_events"].append({"id": uuid.uuid4().hex, "timestamp": time.time(), "text": reply.strip()})
                state["pc_events"] = state["pc_events"][-100:]
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
        # Same race as /api/pc_events -- see the comment there. Returning
        # events:[] while advancing last_id to whatever's currently
        # latest can silently eat the very first reply in a conversation
        # if it lands before the client's cursor has ever been set to a
        # real value. Confirmed live: this is why a mirror reply showed
        # nothing at all on the phone even though the launcher's side
        # completed normally.
        return jsonify({"events": events, "last_id": events[-1]["id"] if events else ""})

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
