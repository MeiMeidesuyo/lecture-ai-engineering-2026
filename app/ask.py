"""第1回: 素のLLMに質問し、会話履歴と要約を管理するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask
    uv run python -m app.ask --chat
    uv run python -m app.ask --full-history "質問"
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = PROJECT_ROOT / "history" / "conversation.json"
HISTORY_VERSION = 1

# Gemini系モデルの入力上限に余裕を持たせる保守的な設定。
# 正確な入力上限はモデルや将来の変更で変わるため、コードでは既定値を明示する。
MODEL_INPUT_LIMIT_TOKENS = 900_000
SUMMARY_THRESHOLD_TOKENS = 700_000
DEFAULT_KEEP_TURNS = 3

EXIT_COMMANDS = {"exit", "quit", "終了", "/exit", "/quit"}


class HistoryError(RuntimeError):
    pass


def estimate_input_tokens(text: str) -> int:
    """トークナイザなしで使用する概算値。英数字でも日本語でも安全側に4文字を1トークンとして扱う。"""
    return (len(text) + 3) // 4


def _validate_history(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("version") != HISTORY_VERSION:
        raise HistoryError("履歴ファイルの形式またはバージョンが不正です。")

    messages = data.get("messages")
    summary = data.get("summary")
    if not isinstance(messages, list) or not isinstance(summary, dict):
        raise HistoryError("履歴ファイルの形式が不正です。")

    for message in messages:
        if not isinstance(message, dict):
            raise HistoryError("履歴のメッセージ形式が不正です。")
        if message.get("role") not in {"user", "model"}:
            raise HistoryError("履歴の役割が不正です。")
        if not isinstance(message.get("content"), str):
            raise HistoryError("履歴の本文が不正です。")

    summarized_count = summary.get("summarized_message_count", 0)
    if not isinstance(summarized_count, int) or summarized_count < 0:
        raise HistoryError("要約済み範囲が不正です。")
    if summarized_count > len(messages):
        raise HistoryError("要約済み範囲が履歴の範囲を超えています。")
    if not isinstance(summary.get("text", ""), str):
        raise HistoryError("要約文の形式が不正です。")
    return data


def new_history() -> dict[str, Any]:
    return {
        "version": HISTORY_VERSION,
        "messages": [],
        "summary": {"text": "", "summarized_message_count": 0},
    }


def load_history(path: Path = HISTORY_PATH) -> dict[str, Any]:
    if not path.exists():
        return new_history()
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        raise HistoryError(f"履歴ファイルが壊れています: {path} ({e})") from e
    except OSError as e:
        raise HistoryError(f"履歴ファイルを読み込めませんでした: {path} ({e})") from e
    return _validate_history(data)


def save_history(history: dict[str, Any], path: Path = HISTORY_PATH) -> None:
    _validate_history(history)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary_path = file.name
            json.dump(history, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except OSError as e:
        if temporary_path:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError(f"履歴を保存できませんでした: {e}") from e


def messages_to_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{'ユーザー' if message['role'] == 'user' else 'AI'}: {message['content']}"
        for message in messages
    )


def build_full_contents(history: dict[str, Any], question: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in history["messages"]:
        role = "model" if message["role"] == "model" else "user"
        content = message["content"]
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["parts"][0]["text"] += "\n" + content
            continue
        normalized.append({"role": role, "parts": [{"text": content}]})

    while normalized and normalized[0]["role"] != "user":
        normalized.pop(0)

    if normalized and normalized[-1]["role"] == "user":
        normalized[-1]["parts"][0]["text"] += "\n" + question
    else:
        normalized.append({"role": "user", "parts": [{"text": question}]})
    return normalized


def build_summary_contents(history: dict[str, Any], question: str, keep_turns: int) -> tuple[list[dict[str, Any]], bool, int]:
    messages = history["messages"]
    summary_text = history["summary"]["text"]
    summarized_count = history["summary"]["summarized_message_count"]

    if not summary_text:
        return build_full_contents(history, question), False, 0

    recent_messages = messages[summarized_count:]

    context_parts = []
    if summary_text:
        context_parts.append(f"古い会話の要約:\n{summary_text}")
    if recent_messages:
        context_parts.append("直近の会話の原文:\n" + messages_to_text(recent_messages))
    context_parts.append("新しい質問:\n" + question)

    return [{"role": "user", "parts": [{"text": "\n\n".join(context_parts)}]}], True, summarized_count


def _print_usage(label: str, usage: Any) -> None:
    if not usage:
        return
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    print(f"[{label}] 入力: {input_tokens}, 出力: {output_tokens}")


def normalize_api_contents(contents: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """SDKへ渡す直前で、末尾が必ずuserになるようContent形式を正規化する。"""
    normalized: list[dict[str, Any]] = []
    for item in contents:
        role = "model" if item.get("role") == "model" else "user"
        parts = item.get("parts") or [{"text": ""}]
        text = "\n".join(str(part.get("text", "")) for part in parts)
        if not text:
            continue
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["parts"][0]["text"] += "\n" + text
        else:
            normalized.append({"role": role, "parts": [{"text": text}]})

    while normalized and normalized[0]["role"] != "user":
        normalized.pop(0)

    if not normalized or normalized[-1]["role"] != "user":
        normalized.append({"role": "user", "parts": [{"text": question}]})
    return normalized


def ask(question: str, contents: list[dict[str, Any]] | None = None, client: Any | None = None) -> str:
    if client is None:
        client = genai.Client(api_key=get_api_key())
    api_contents = normalize_api_contents(contents, question) if contents is not None else question
    response = client.models.generate_content(model=MODEL, contents=api_contents)
    _print_usage("tokens", response.usage_metadata)
    return response.text


def _summarize_new_turns(client: Any, history: dict[str, Any], keep_turns: int) -> tuple[dict[str, Any] | None, Any]:
    messages = history["messages"]
    old_summary = history["summary"]
    summarized_count = old_summary["summarized_message_count"]
    summarized_end = max(0, len(messages) - keep_turns * 2)

    if summarized_end <= summarized_count:
        return None, None

    newly_targeted = messages[summarized_count:summarized_end]
    previous_summary = old_summary["text"]
    prompt = (
        "以下の会話を要約してください。目的、確定事項、ユーザーの希望、数値、固有名詞、"
        "制約、未解決事項を優先し、ユーザーの発言とAIの提案・推測を区別し、訂正後の内容を優先してください。\n\n"
        + ("前回の要約:\n" + previous_summary + "\n\n" if previous_summary else "")
        + "新たに要約する会話:\n" + messages_to_text(newly_targeted)
    )
    response = client.models.generate_content(
        model=MODEL,
        contents={"role": "user", "parts": [{"text": prompt}]},
    )
    updated = {
        "text": response.text.strip(),
        "summarized_message_count": summarized_end,
    }
    return updated, response.usage_metadata


def _print_over_limit(text: str, estimated_tokens: int, limit: int) -> bool:
    print("入力が上限を超える可能性があります。")
    print(f"推定入力トークン数: {estimated_tokens} / 上限: {limit}")
    print("質問を短くするか、履歴を整理して再実行してください。会話は変更されません。")
    return False


def answer_one_question(client: Any, history: dict[str, Any], question: str, keep_turns: int, full_history_mode: bool) -> bool:
    staged_summary: dict[str, Any] | None = None
    if full_history_mode:
        contents = build_full_contents(history, question)
        estimated_tokens = estimate_input_tokens(messages_to_text(history["messages"] + [{"role": "user", "content": question}]))
        if estimated_tokens > MODEL_INPUT_LIMIT_TOKENS:
            return _print_over_limit("", estimated_tokens, MODEL_INPUT_LIMIT_TOKENS)
    else:
        estimated_tokens = estimate_input_tokens(messages_to_text(history["messages"] + [{"role": "user", "content": question}]))
        if estimated_tokens > SUMMARY_THRESHOLD_TOKENS:
            try:
                staged_summary, summary_usage = _summarize_new_turns(client, history, keep_turns)
                if staged_summary:
                    print("[summary] 会話を自動要約しました。")
                    _print_usage("summary tokens", summary_usage)
                    summary_history = {
                        "messages": history["messages"],
                        "summary": staged_summary,
                    }
                    contents, _, _ = build_summary_contents(summary_history, question, keep_turns)
                else:
                    contents = build_full_contents(history, question)
            except (errors.APIError, Exception) as e:
                print("[summary error] 要約に失敗しました。既存の要約と要約済み範囲は変更せず、全文を原文として保持します。")
                if isinstance(e, errors.APIError):
                    print(api_error_message(e.code, e.message))
                else:
                    print(str(e))
                contents = build_full_contents(history, question)

            contents_text = "\n".join(part["text"] for item in contents for part in item["parts"])
            estimated_tokens = estimate_input_tokens(contents_text)

        else:
            contents = build_full_contents(history, question)

        if estimated_tokens > MODEL_INPUT_LIMIT_TOKENS:
            return _print_over_limit("", estimated_tokens, MODEL_INPUT_LIMIT_TOKENS)

    try:
        answer = ask(question, contents=contents, client=client)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        return False
    except Exception as e:
        print(f"API呼び出しに失敗しました: {e}")
        return False

    old_summary = history["summary"]
    history["messages"].append({"role": "user", "content": question})
    history["messages"].append({"role": "model", "content": answer})
    if staged_summary is not None:
        history["summary"] = staged_summary
    try:
        save_history(history)
    except RuntimeError as e:
        history["messages"] = history["messages"][:-2]
        history["summary"] = old_summary
        print(str(e))
        return False
    print(answer)
    return True


def _chat_loop(args: argparse.Namespace, client: Any, history: dict[str, Any]) -> None:
    question: str | None = args.question
    if question is None:
        question = DEFAULT_QUESTION

    while True:
        print(f"[質問] {question}\n")
        if not answer_one_question(client, history, question, args.keep_turns, args.full_history):
            if question != "":
                print("未回答の質問は履歴に追加されませんでした。")
        try:
            question = input("あなた: ").strip()
        except EOFError:
            print("\n会話を終了します。")
            return
        if not question:
            print("会話を終了します。")
            return
        if question.lower() in EXIT_COMMANDS:
            print("会話を終了します。")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="履歴と要約機能付きのLLM CLI")
    parser.add_argument("question", nargs="?", help="質問（省略時は講座用の既定質問）")
    parser.add_argument("--chat", action="store_true", help="回答後も質問を続ける対話モード")
    parser.add_argument("--keep-turns", type=int, default=DEFAULT_KEEP_TURNS, help="要約しない直近の往復数（既定: 3）")
    parser.add_argument("--full-history", action="store_true", help="自動要約を使わず全文を送る（上限超過時は送信しない）")
    args = parser.parse_args()

    if args.keep_turns < 0:
        parser.error("--keep-turns は0以上にしてください")

    print(f"[model] {MODEL}")
    print(f"[history] {HISTORY_PATH}")
    try:
        history = load_history()
    except HistoryError as e:
        print(str(e))
        raise SystemExit(1)

    try:
        client = genai.Client(api_key=get_api_key())
    except SystemExit as e:
        raise e

    if args.chat:
        _chat_loop(args, client, history)
        return

    question = args.question if args.question else DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    if not answer_one_question(client, history, question, args.keep_turns, args.full_history):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
