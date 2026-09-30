# Krea2 AFM (experimental)

ComfyUI native Krea2 向けの独立した診断用カスタムノードです。
**現在は計画書の Task 0–1。Attention の編集機能は未実装です。**
Anima AFM や Krea2T Enhancer をインストールする必要はありません。

モデルを使った実機検証は今回の対象外です。CPU テストは API 契約・
状態分離・layout 検証の証拠であり、GPU 対応・高速化・画質改善を示しません。

## 対応の前提

- ComfyUI のネイティブ `comfy.ldm.krea2.model.SingleStreamDiT`。
- V3 ノード API、`OUTER_SAMPLE` / `DIFFUSION_MODEL` wrapper、`post_input`。
- ソース確認基準: ComfyUI `9d80841aa1990305cc7a280c5ea505f317efcc2d`。
  これは実機確認済み最低版ではありません。実行時にも必要 API を検査します。
- 通常の4D latent、時間軸が1の5D latent。reference / 複数フレームは対象外。
- 他の Attention override、post_input、QKV、Attention 出力パッチは競合エラー。

## インストールとテスト

このフォルダーを ComfyUI の `custom_nodes/AFM-for-Krea2-experimental` に配置し、
ComfyUI を再起動します。既存の ComfyUI 環境の PyTorch を使用します。
このパッケージのために PyTorch や GPU ドライバーを更新する必要はありません。

CPU テストは PyTorch と pytest が入った環境で次のように実行します。

```console
python -m pytest tests -q
python -m compileall -q afm nodes.py __init__.py
```

ノード、ログ、後日の実機確認については [検証手順](docs/validation.md) を参照。
[実装状況](docs/implementation-status.md) と [第三者資料](NOTICE.md) も同梱しています。

## 使用方法

モデルローダーの MODEL と KSampler の MODEL の間に **Krea2 AFM Model Patch**
を接続します。最初は `discover`、次に `observe` / `block_scope=0-3`。

| モード | 動作 |
| --- | --- |
| `off` | clone を返すだけ。診断 hook / Attention override を追加しない |
| `discover` | 全本体 block を記録。画像・テキスト出力は元 backend の戻り値そのもの |
| `observe` | `block_scope` に指定した本体 block を記録。同じく出力を変更しない |

`branch_mode` と進行区間は、今後の編集対象となる行・区間をログへ示すための
設定です。診断版では選択・非選択を含む Attention 出力全体を変更しません。
`spectral_diag` は off 固定です。`edit` は API 経由で指定しても明示エラーになります。

既定のログ先は ComfyUI の `output/krea2_afm/diagnostics.jsonl`。各 run に固有の
`run_id` が付き、`run_start`、`attention`、`run_summary` を追記します。
`jsonl_path` を空にするとファイル出力を無効化します。`debug_level=summary` は
開始・集計をコンソールにも表示し、`verbose` は各呼び出しも表示します。

`fail_mode=raise` を推奨します。`fallback_original` は対象外 layout の診断を
スキップして元の経路を使い、理由を記録します。元 backend 自体の例外は再試行せず
伝播します。`edited_calls` はこの版では常に0です。

## API ワークフロー

- [baseline](examples/krea2_afm_baseline_api.json): AFM ノードなし。
- [discover](examples/krea2_afm_discover_api.json): 全本体 block の診断。
- [observe](examples/krea2_afm_observe_api.json): block 0–3 の診断。

これらは **API prompt JSON** です。ブラウザーへドラッグする UI workflow 形式では
ありません。公式 Turbo ワークフローの標準ノード構成を参考に、512×512、固定seed、
8 steps、CFG=1、Euler/simple としています。任意の LoRA / prompt enhancement は含みません。
モデル名は例であり、インストール済みファイル名へ合わせる必要があります。

モデルやサーバーを使わずに JSON の参照関係を確認できます。

```console
python scripts/validate_workflow.py examples/krea2_afm_discover_api.json
```

保存済みの実サーバー `/object_info` があれば、ノード・入力・socket・モデル選択肢も
確認できます。このコマンドはネットワーク通信・生成・モデル取得を行いません。

```console
python scripts/validate_workflow.py examples/krea2_afm_observe_api.json --object-info object_info.json
```
