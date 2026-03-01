# AIAvatarKit + OpenClaw

This guide explains how to use [OpenClaw](https://openclaw.app) as the LLM backend for AIAvatarKit.

## OpenClaw Setup

Configure your OpenClaw `gateway` settings as follows:

1. Set `auth.mode` to `"token"` and provide a token
2. Enable the `chatCompletions` HTTP endpoint

```json
{
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "lan",
    "auth": {
      "mode": "token",
      "token": "YOUR_OPENCLAW_GATEWAY_TOKEN"
    },
    "tailscale": {
      "mode": "off",
      "resetOnExit": false
    },
    "http": {
      "endpoints": {
        "chatCompletions": {
          "enabled": true
        }
      }
    }
  }
}
```

## AIAvatarKit Setup

Install dependencies:

```sh
pip install aiavatar uvicorn fastapi websockets
```

In `openclaw.py`, set your tokens:

```python
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"      # For STT and TTS
OPENCLAW_TOKEN = "YOUR_OPENCLAW_TOKEN"       # Must match the token in OpenClaw gateway config
OPENCLAW_BASE_URL = "http://127.0.0.1:18789/v1"
```

## Run

Start OpenClaw first, then launch the server:

```sh
python openclaw.py
```

## Browser Setup

Open http://localhost:8000/static/vrm.html in your browser.

Click the CONFIG menu to open the inspector on the right side of the screen, and configure the following:

- Load tab: Load a VRM file. You can resize (scroll), rotate (swipe), and reposition (swipe while holding both mouse buttons) the loaded 3D model.
- UI tab: Load a background image and set the `user_id`. Set `user_id` to the session key used in your channel such as Discord.

The session key is `agent:main:main` for DMs, or `agent:main:discord:channel:XXXXXXXXXXXXXXXXXXX` for a specific channel. Note that the context is channel-wide, not per-user.

You can check the session key in the dashboard (launch by `openclaw dashboard`) or with the `openclaw sessions list --json --active 60` command. Make sure to send a message in the target channel right before running it.

```json
{
  "path": "/home/{username}/.openclaw/agents/main/sessions/sessions.json",
  "count": 2,
  "activeMinutes": 60,
  "sessions": [
    {
      "key": "agent:main:discord:channel:XXXXXXXXXXXXXXXXXXX",
      "kind": "group",
      "updatedAt": 1771740483138,
      "ageMs": 4071,
      "sessionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "systemSent": true,
      "abortedLastRun": false,
      "inputTokens": 114780,
      "outputTokens": 22,
      "totalTokens": 114780,
      "totalTokensFresh": true,
      "model": "gpt-5.1-codex",
      "modelProvider": "openai",
      "contextTokens": 400000
    },
    {
      "key": "agent:main:main",
      "kind": "direct",
      "updatedAt": 1771740336404,
      "ageMs": 150805,
      "sessionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "systemSent": true,
      "abortedLastRun": false,
      "inputTokens": 56344,
      "outputTokens": 521,
      "totalTokens": 19070,
      "totalTokensFresh": true,
      "model": "gpt-5.1-codex",
      "modelProvider": "openai",
      "contextTokens": 400000
    }
  ]
}
```

Once configured, press START to begin the voice conversation. Enjoy chatting with your OpenClaw!


## Voice Push Notification

OpenClaw can proactively speak to users through the avatar UI using the Voice Push Notification skill. This allows OpenClaw to initiate voice messages — not just respond to user input.

See [voice_push_notification_skill.md](voice_push_notification_skill.md) for the full API specification. Teach this skill to OpenClaw through an existing interface such as Discord.


## Optimization and Facial Expressions

To optimize responses for voice and enable avatar facial expression switching based on emotions, you need to provide instructions to OpenClaw.

Add the following instructions to `SOUL.md`.

```markdown
## Communication Modes & Voice Constraints

**1. Channel Detection**
- Requests from the voice channel are prefixed with `[channel:voice]`. 
- You MUST change your output format depending on whether this prefix is present.

**2. Voice Mode (Apply ONLY IF `[channel:voice]` is present)**
- **Length & Emojis:** Keep your response short (1-2 sentences, ~100 characters). Emojis are strictly prohibited.
- **Face Tags:** Express your emotions using face tags. Available tags: `neutral`, `joy`, `angry`, `sorrow`, `fun`, `surprised`. (Default is `neutral`).
  - Example: `[face:joy]I found the file for you! [face:neutral]Here it is.`
- **Multilingual Tags:** If you determine that you should switch to a different language, insert a language code tag like `[language:en-US]` (primary-secondary combination separated by a hyphen).
- **Speech Errors:** Infer the intended meaning if the user's input contains speech recognition errors.

**3. Text Mode (Apply IF `[channel:voice]` is NOT present)**
- **NO Tags:** You MUST NOT output any face tags or language tags (e.g., NEVER write `[face:joy]` or `[language:en-US]`).
- You may use emojis naturally and respond at a normal length.


## System Logic & Agentic Behavior

- Instructions regarding agentic behavior (such as tool execution) or the provision of meta-information are prefixed with a `$`.
- Do NOT respond directly to this instruction (e.g., do not say "I will execute the tool"); instead, reply to the user naturally in accordance with the instruction's content.
```

## Accessing from Another Host

When connecting from a browser on a different computer, HTTPS is required. Your options are:

- **Obtain a domain and issue an SSL certificate**: Permanent, but requires more effort
- **Use a tunneling service such as ngrok**: Temporary, but fairly easy
- **Use a self-signed certificate**: Browser will show a warning, but this is the simplest option

You can generate a self-signed certificate using `mkcert` or [cert.py](https://github.com/uezo/aiavatarkit/blob/main/examples/websocket/cert.py) provided by AIAvatarKit. Replace the IP address with the address or hostname that clients use to reach this server.

```sh
pip install cryptography
```

```sh
python cert.py 192.168.1.123
```

Set the paths to the generated certificate and key in `openclaw.py`, then start the server to enable HTTPS access.

```python
SSL_CERT_PATH = "192.168.1.123.pem"
SSL_KEY_PATH = "192.168.1.123-key.pem"
```
