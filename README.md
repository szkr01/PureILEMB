# PureILEMB

PureILEMBは、AIを活用したローカル画像検索・管理バックエンドシステムです。 ローカルにある画像フォルダを監視し、自動的に解析・タグ付け・ベクトル化を行うことで、高速な類似画像検索やタグ検索を実現します。

## 主な機能

- **リアルタイム監視**: 指定したフォルダに追加された画像を `watchdog` で自動検知し、インデックスに追加・更新します。
- **AIタグ付け・ベクトル化**: `wd-eva02-large-tagger-v3` などの最新のAIモデルを使用して、画像の特徴量抽出と自動タグ付けを行います。
- **高速ベクトル検索**: Facebook AI Researchの `Faiss` ライブラリを使用し、大量の画像から類似画像を瞬時に検索します。
- **ハイブリッド検索**: 画像そのものによる検索（Image-to-Image）や、テキストタグによる検索に対応しています。
- **モバイル対応Web UI**: スマートフォンからのアクセスを自動検出し、モバイル最適化されたビュー (`index_mobile.html`) を提供します。
- **効率的なデータ管理**: 画像のメタデータやパス情報はSQLiteで、ベクトルデータはFaissで効率的に管理します。また、インデックスの永続化にも対応しています。

## 必要条件

- **OS**: Windows (推奨), Linux, macOS
- **Python**: 3.12 以上
- **パッケージマネージャ**: `uv` (推奨) または `pip`
- **ハードウェア**: CUDA対応GPU (NVIDIA製) の使用を強く推奨します（CPUでも動作しますが、処理速度が大幅に低下します）。

## インストール方法

### 1. リポジトリのクローン

### 2. 依存関係のインストール

`uv` を使用する場合（推奨）:
```bash
uv sync
```

`pip` を使用する場合:
```bash
pip install -r requirements.txt
```

> **注意**: PyTorchはシステムのCUDAバージョンに合わせて適切にインストールしてください。 `requirements.txt` にはCPU版の `faiss-cpu` が記載されていますが、GPUを使用する場合は `faiss-gpu` のインストールを検討してください。

## 設定

プロジェクトルートディレクトリに `config.json` を作成することで、動作設定をカスタマイズできます。ファイルが存在しない場合はデフォルト値が使用されます。

**`config.json` の作成例:**

```json
{
  "watch_dirs": [
    "C:/Users/Public/Pictures/Sample Pictures",
    "./watched_images"
  ],
  "model_repo": "SmilingWolf/wd-eva02-large-tagger-v3",
  "db_path": "data/images.db",
  "faiss_index_path": "data/faiss.index"
}
```

### 設定項目説明

| キー | デフォルト値 | 説明 |
| --- | --- | --- |
| `watch_dirs` | `["./watched_images"]` | 監視対象とするディレクトリパスのリスト。ここにある画像がインデックスされます。 |
| `model_repo` | `SmilingWolf/wd-eva02-large-tagger-v3` | 特徴抽出に使用するHugging Faceモデルのリポジトリ名。 |
| `db_path` | `data/images.db` | メタデータを保存するSQLiteデータベースのパス。 |
| `faiss_index_path` | `data/faiss.index` | ベクトルインデックスを保存するファイルのパス。 |

## 実行方法

以下のコマンドでサーバーを起動します。

```bash
uv run python -m app.main
```
(または `python -m app.main`)

サーバーが起動すると、デフォルトでポート `8002` で待機します。

### ブラウザでのアクセス

- **Web UI**: [http://localhost:8002/app/](http://localhost:8002/app/)
- **API ドキュメント (Swagger UI)**: [http://localhost:8002/docs](http://localhost:8002/docs)

## API エンドポイント概要

- **GET /app/**: メインのWebインターフェースを提供します。User-Agentに応じてモバイル版/PC版を切り替えます。
- **POST /API/search**: 検索API。テキストクエリ (`q`) または画像ファイル (`image`) を受け取り、類似画像を返します。
- **GET /API/media/{entry_id}**: 画像データを返します。`size=300x300` のようにクエリパラメータでリサイズや切り抜きが可能です。
- **GET /API/tags**: 利用可能なタグの一覧を返します。
- **GET /API/tags_from_id/{entry_id}**: 特定の画像IDに関連付けられたタグと確率を取得します。

## プロジェクト構造

- `app/`: バックエンドのソースコード
    - `main.py`: FastAPIアプリケーションのエントリーポイント
    - `watcher.py`: ディレクトリ監視サービス
    - `indexer.py`: ベクトルインデックス管理
    - `database.py`: SQLiteデータベース操作
    - `services/`: 検索ロジックなどのサービス層
- `web_local/`: Webフロントエンド (HTML/JS/CSS)
- `data/`: データベースファイルやインデックスの保存先 (自動生成)

## ライセンス

[MIT License](LICENSE)
