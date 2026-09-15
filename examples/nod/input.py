"""Plain conversation input with explicit assistant-nod markers."""

from html import escape


def _line(text):
    # Only markers inserted by this module should be interpreted as nods.
    return escape(" ".join(text.split()), quote=False)


def annotated_text(text, nods, limit, *, text_offset=0):
    """Insert nods at original prefix offsets, retaining unplaceable ones separately.

    text_offset is the number of characters discarded before this retained text.
    Nod text_length values always refer to the original cumulative utterance.
    """
    if not isinstance(limit, int) or limit < 1 or not isinstance(text_offset, int) or text_offset < 0:
        raise ValueError("limit must be positive and text_offset nonnegative")
    start = max(0, len(text) - limit)
    positions, unplaced = [], []
    for event in nods:
        prefix, phrase = event["acknowledged_text"], event["phrase"]
        end = event.get("text_length", len(prefix)) - text_offset
        prefix_start = end - len(prefix)
        overlap = max(0, prefix_start)
        matched = (prefix and end + text_offset >= len(prefix) and 0 < end <= len(text)
                   and text[overlap:end] == prefix[overlap - prefix_start:])
        if matched and end >= start:
            positions.append((end, phrase))
        else:
            unplaced.append(event)
    parts, cursor = [], start
    for position, phrase in sorted(positions, key=lambda p: p[0]):
        parts.extend((_line(text[cursor:position]), f"<nod_assistant>{_line(phrase)}</nod_assistant>"))
        cursor = position
    parts.append(_line(text[cursor:]))
    for event in unplaced:
        parts.append(f"\nAI（送信済み相槌・位置不明）：{_line(event['phrase'])}"
                     f"（受け止めた内容：{_line(event['acknowledged_text'][-600:])}）")
    return "".join(parts)


def plain_input(history, text, nods=(), *, history_limit=10, text_offset=0):
    """Format one independent user message; assistant content must already be spoken text."""
    if not isinstance(history_limit, int) or not 1 <= history_limit <= 10:
        raise ValueError("history_limit must be between 1 and 10 messages")
    lines = ["【これまでの会話】"]
    for row in history[-history_limit:]:
        if row.get("role") == "user":
            lines.append("ユーザー：" + annotated_text(row["content"], row.get("nods", ()), 600,
                                                       text_offset=row.get("text_offset", 0)))
        elif row.get("role") == "assistant":
            lines.append("AI：" + _line(row["content"][-600:]))
    if len(lines) == 1:
        lines.append("（なし）")
    lines += ["", "【今回のユーザー発言】",
              "ユーザー：" + annotated_text(text, nods, 1200, text_offset=text_offset), "",
              "この流れで、今回の発言に相槌を入れる？",
              "候補の文言をnod_assistant要素1つで返す。入れない場合は空要素。"]
    return "\n".join(lines)
