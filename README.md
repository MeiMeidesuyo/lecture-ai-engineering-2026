# AIE2026 演習・宿題リポジトリ

AIエンジニアリング実践講座（AIE2026）の演習と宿題で使うリポジトリです。
講座を通して、**AIエンジニアリングの学習を支援するエージェント**を少しずつ育てていきます。

## 想定する環境

演習は **ブラウザだけで** 進められるように作っています。自分の PC への導入は不要です。

| 使うもの | 用途 | 注意 |
|---|---|---|
| GitHub の **個人** アカウント | このリポジトリを fork し、作業ブランチと PR で提出する | 会社や学校が管理するアカウントでは Codespaces・Copilot が制限されることがあります |
| GitHub Codespaces | 第1・3・5回などの開発環境。`.devcontainer/` の設定で Python 3.12・uv・Node.js が入ったコンテナが自動で用意されます | 個人アカウントの無料枠は月 120 コア時間（2 コアで約 60 時間）。**使い終わったら停止**してください |
| GitHub Copilot（Free でよい） | コードの補完とチャット。第5回の発展課題では MCP クライアントとしても使います | Free は補完・チャットの回数に月の上限があります |
| Gemini API（Google AI Studio の無料枠） | LLM の呼び出し。**個人**の Google アカウントでキーを発行し、Codespaces secret `GEMINI_API_KEY` で渡します | 無料枠には 1 分あたり・1 日あたりの上限があります（下の「前提と制約」） |
| Google Colab（T4 GPU、無料枠） | 第2回の LoRA 学習だけに使います | GPU の利用時間に上限があります。終わったらランタイムを削除してください |

自分の PC で動かす場合は、Python 3.12 と [uv](https://docs.astral.sh/uv/) を入れて `uv sync --frozen` を実行し、環境変数 `GEMINI_API_KEY` を設定してください。

## はじめに（第1回）

1. このリポジトリを自分のアカウントに **fork** する
2. [Google AI Studio](https://aistudio.google.com/apikey) で Gemini API キーを作る
3. GitHub の **Settings → Codespaces → Secrets** に `GEMINI_API_KEY` を登録し、使う repository に自分の fork を選ぶ
4. 自分の fork で **Code → Codespaces → Create codespace on main**
5. ターミナルで動作確認

```bash
uv run pytest -q
uv run python -m app.ask "LoRAとは何ですか？"
uv run python -m app.ask              # 質問省略時は講座用の既定質問
uv run python -m app.ask --chat       # 回答後も質問を続ける
uv run python -m app.ask --chat --keep-turns 3
uv run python -m app.ask --full-history "全文を送って確認したい質問"
```

`--chat` では `exit`、`quit`、`終了`、`/exit`、`/quit`、または空の入力で終了・無視できます。

### 会話履歴と要約

会話履歴は、実行場所に関係なくプロジェクト直下の `history/conversation.json` に保存されます。このファイルは `.gitignore` の `/history/` で Git の管理対象外です。

- **保存範囲**: ユーザーとAIの発言を `role` と `content` を持つ `messages` 全文で保存します。要約文と、要約済みメッセージ数も保存します。要約しても全文は削除・置換しません。
- **APIへ送る範囲**: 通常は短い会話なら `messages` の全文と新しい質問を送ります。長い会話では「古い会話の要約＋直近3往復の原文＋新しい質問」を送ります。`--keep-turns` で直近往復数を変更できます。
- **要約するタイミング**: 保存済み履歴と新しい質問の概算入力が 700,000 トークンを超えたときに自動要約します。既定の入力上限に余裕を持たせるため、モデル側の入力上限 900,000 トークンより低く設定しています。この概算は厳密なトークナイザではなく、4文字を1トークンとして日本語にも安全側に寄せた簡易判定です。

要約は細部を失う可能性があります。`history/conversation.json` に全文を保存していても、要約モードでは全文はモデルに送られず、要約と直近原文しか見えません。原文を確認したいときは `--full-history` を使います。ただし推定入力が 900,000 トークンを超える場合は送信せず、質問を短くするなどの案内を表示します。

詳しい手順は [lectures/01/problem.md](lectures/01/problem.md) を見てください。

## 構成

```
.devcontainer/   Codespaces の環境定義（Python 3.12、uv、Node.js。起動時に uv sync --frozen）
.github/workflows/test.yml  PR ごとにテストを実行（fork では Actions タブで有効化が必要）
history/        会話履歴（Git管理外。会話ごとの全文・要約を保存）
app/
  config.py      モデル名とAPIキーの読み込み
  ask.py         第1回：LLMに質問する
  tools.py       第3回：エージェントのツール（用語集の検索など）
  agent.py       第3回：Pydantic AI の学習支援エージェント（第5回：--mcp で MCP 経由）
  mcp_server.py  第5回：用語集を公開する MCP サーバー
  mcp_servers.json 第5回：使う MCP サーバーの設定（mcpServers 形式）
data/
  glossary.json  講座の用語集
  lectures/      各回の要点
lectures/NN/
  problem.md     その回の演習と宿題
  answer.md      あなたが記入して提出するファイル
notebooks/       第2回：Google Colab で動かす LoRA のノートブック（lora_format_tuning.ipynb）
scripts/         補助スクリプト
tests/           テスト（APIキー不要）
pyproject.toml / uv.lock  依存パッケージとその固定版
```

## 提出のしかた

1. 回ごとに作業ブランチを作る（例：`git switch -c lec01`）
2. 演習・宿題を行い、`lectures/NN/answer.md` に記入する
3. push して、**自分の fork の中で** 作業ブランチ → `main` の PR を作る

宿題は「基礎・実装・発展」の3レベルがあり、どれか1つに取り組めば完了です。

## 前提と制約

- **依存パッケージの版は固定**しています（`uv.lock`。Python 3.12、pydantic-ai-slim 2.57.0、google-genai 2.31.0）。講座の期間中は上げません。`uv sync --frozen` でそのまま入れてください。
- **既定のモデルは `gemini-3.5-flash-lite`** です。環境変数 `AIE_MODEL` で切り替えられます（下の「モデルの切り替え」）。ほかのモデルや自前契約の API を使ってもかまいません
- **Gemini API の無料枠にはレート制限**があります。エージェント（第3回以降）は 1 つの質問で LLM を複数回呼ぶので、第1回より早く上限に達します。`レート制限に達しました` と出たら 1 分ほど待ってください。`モデルが混み合っています（503）` が続くときは `AIE_MODEL=gemini-3.1-flash-lite` を試してください
- **API キーは環境変数 `GEMINI_API_KEY` だけ**で渡します。コード・コミット・チャットに貼らないでください。漏れたら AI Studio で削除して作り直します
- Codespaces からインターネットへは出られますが、外から Codespaces の中へは接続できません（ポート転送は既定で非公開）。第5回の MCP サーバーは Codespaces の中で動かします
- テスト（`uv run pytest -q`）は API キーなしで動きます（LLM を呼ばない TestModel を使用）

## 守ってほしいこと

- **APIキーをコードやコミットに書かない。** Codespaces secret で渡します
- **使い終わったら Codespaces を停止する。** 無料枠を使い切ると起動できなくなります
- AIツール（Copilot など）は使ってかまいません。提案されたコードは、読んで理解してから採用してください

## モデルの切り替え

既定モデルは `gemini-3.5-flash-lite` です。環境変数 `AIE_MODEL` で切り替えられます。

```bash
AIE_MODEL=gemini-3.5-flash uv run python -m app.ask "LoRAとは何ですか？"
```

無料枠には1分あたり・1日あたりのリクエスト上限があります。上限は [Gemini API のレート制限のページ](https://ai.google.dev/gemini-api/docs/rate-limits) で確認してください。
