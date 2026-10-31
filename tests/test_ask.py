import json
from types import SimpleNamespace

import pytest

from app import ask
from argparse import Namespace


class MockModels:
    def __init__(self, answer="回答です"):
        self.answer = answer
        self.calls = []

    def generate_content(self, model, contents):
        self.calls.append(contents)
        return SimpleNamespace(
            text=self.answer,
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        )


class MockClient:
    def __init__(self, answer="回答です"):
        self.models = MockModels(answer)


def test_history_round_trip_preserves_full_messages(tmp_path):
    path = tmp_path / "history" / "conversation.json"
    history = ask.new_history()
    history["messages"] = [
        {"role": "user", "content": "質問1"},
        {"role": "model", "content": "回答1"},
    ]

    ask.save_history(history, path)

    restored = ask.load_history(path)
    assert restored == history
    assert restored["messages"][0]["role"] == "user"


def test_broken_history_is_not_overwritten(tmp_path):
    path = tmp_path / "conversation.json"
    path.write_text("{ broken json", encoding="utf-8")

    with pytest.raises(ask.HistoryError, match="壊れています"):
        ask.load_history(path)

    assert path.read_text(encoding="utf-8") == "{ broken json"


def test_short_conversation_sends_full_history_and_saves_answer(tmp_path, monkeypatch):
    real_save_history = ask.save_history
    monkeypatch.setattr(ask, "HISTORY_PATH", tmp_path / "conversation.json")
    monkeypatch.setattr(ask, "save_history", lambda history, path=None: real_save_history(history, path or tmp_path / "conversation.json"))
    history = ask.new_history()
    client = MockClient()

    assert ask.answer_one_question(client, history, "次の質問", keep_turns=3, full_history_mode=False)

    assert history["messages"] == [
        {"role": "user", "content": "次の質問"},
        {"role": "model", "content": "回答です"},
    ]
    assert client.models.calls[0][0]["role"] == "user"


def test_normalize_api_contents_never_ends_with_model():
    contents = [
        {"role": "user", "parts": [{"text": "質問1"}]},
        {"role": "model", "parts": [{"text": "回答1"}]},
        {"role": "model", "parts": [{"text": "回答2"}]},
    ]

    normalized = ask.normalize_api_contents(contents, "新しい質問")

    assert normalized[-1]["role"] == "user"
    assert normalized[-1]["parts"] == [{"text": "新しい質問"}]
    assert all(normalized[index]["role"] != normalized[index + 1]["role"] for index in range(len(normalized) - 1))


def test_full_contents_normalizes_roles_and_always_ends_with_user():
    history = ask.new_history()
    history["messages"] = [
        {"role": "model", "content": "先頭の不要な回答"},
        {"role": "user", "content": "質問1"},
        {"role": "user", "content": "質問2"},
        {"role": "model", "content": "回答1"},
    ]

    contents = ask.build_full_contents(history, "新しい質問")

    assert contents[-1]["role"] == "user"
    assert contents[-1]["parts"] == [{"text": "新しい質問"}]
    assert all(contents[index]["role"] != contents[index + 1]["role"] for index in range(len(contents) - 1))


def test_chat_loop_empty_enter_ends_without_second_question(monkeypatch, capsys):
    args = Namespace(question="最初の質問", keep_turns=3, full_history=False)
    answered = []

    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr(
        ask,
        "answer_one_question",
        lambda client, history, question, keep_turns, full_history: answered.append(question) or True,
    )

    ask._chat_loop(args, MockClient(), ask.new_history())

    assert answered == ["最初の質問"]
    assert "会話を終了します。" in capsys.readouterr().out


def test_full_history_mode_over_limit_does_not_append(tmp_path, monkeypatch):
    monkeypatch.setattr(ask, "MODEL_INPUT_LIMIT_TOKENS", 1)
    history = ask.new_history()
    history["messages"] = [{"role": "user", "content": "非常に長い会話です"}] * 2

    result = ask.answer_one_question(MockClient(), history, "質問", keep_turns=3, full_history_mode=True)

    assert result is False
    assert len(history["messages"]) == 2


def test_re_summary_prompt_includes_previous_summary_and_only_new_range():
    history = ask.new_history()
    history["summary"] = {"text": "前回の要約", "summarized_message_count": 4}
    history["messages"] = [
        {"role": "user", "content": "古い1"}, {"role": "model", "content": "回答1"},
        {"role": "user", "content": "古い2"}, {"role": "model", "content": "回答2"},
        {"role": "user", "content": "新しい1"}, {"role": "model", "content": "回答3"},
        {"role": "user", "content": "新しい2"}, {"role": "model", "content": "回答4"},
        {"role": "user", "content": "直近1"}, {"role": "model", "content": "直近回答1"},
    ]
    client = MockClient(answer="統合要約")

    updated, _ = ask._summarize_new_turns(client, history, keep_turns=2)

    prompt = client.models.calls[0]["parts"][0]["text"]
    assert "前回の要約" in prompt
    assert "前回の要約" in prompt
    assert "新しい1" in prompt
    assert "直近1" not in prompt
    assert updated["text"] == "統合要約"
    assert updated["summarized_message_count"] == 6
