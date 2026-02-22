# Voice Push Notification Skill

Send a voice message to a specific user's avatar session by OpenClaw session key.

## Usage

Send a voice notification by calling the perform API with the session key as `user_id`.

- **Endpoint**: `POST {base_url}/avatar/perform`
- **Headers**:
  - `Authorization: Bearer {api_key}`
  - `Content-Type: application/json`
- **Body**:

```json
{
  "text": "[face:joy]Your notification message here",
  "user_id": "agent:main:main"
}
```

### Control tags

You can embed control tags in the text to control the avatar's expression and animation:

- Face: `[face:expression_name]` (e.g., `[face:joy]`, `[face:surprise]`)
- Animation: `[animation:animation_name]` (e.g., `[animation:wave_hands]`)

### Response

```json
{
  "message": "Avatar performance completed successfully"
}
```

### Error responses

- `400`: No active session found for the given session key
- `401`: Invalid or missing API Key

## Example

To notify session `agent:main:main` with a greeting:

```
POST {base_url}/avatar/perform
Authorization: Bearer {api_key}
Content-Type: application/json

{"text": "[face:joy]Hello! You have a new message.", "user_id": "agent:main:main"}
```
