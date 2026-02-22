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
      "token": "YOUR_OPENCLAW_TOKEN"
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
uvicorn openclaw:app --port 8000
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

To optimize responses for voice and enable avatar facial expression switching based on emotions, you need to provide instructions to OpenClaw. Add the following instructions to OpenClaw's configuration, or tell it directly through your main channel.

```markdown
## Additional instructions specific to this request

- Requests from the voice channel are prefixed with `[channel:voice]`.
- For responses to the voice channel, keep your response to 1-2 sentences and around 100 characters or less. Do not use emojis.
- The user's input may contain speech recognition errors, so infer the intended meaning from context when it seems odd.
- You can express emotions using the following facial expressions:
    - neutral
    - joy
    - angry
    - sorrow
    - fun
    - surprised
- Use neutral by default, but when you want to express a particular emotion, insert a face tag like [face:joy] in your response.

Example:
[face:joy]I can see the ocean! [face:fun]Hey, let's go swimming!
```
