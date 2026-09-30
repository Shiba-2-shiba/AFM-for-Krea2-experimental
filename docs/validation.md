# 検証範囲

この版は Task 0–1 の診断版です。今回、実モデルの推論は実施しません。

| 項目 | 根拠 / 状態 |
| --- | --- |
| Python / PyTorch | ローカル Python 3.10.11 / PyTorch 2.10.0+cpu |
| Krea2 API | 固定 ComfyUI commit のソース確認と最小 CPU スタブ |
| Turbo / Raw 推論 | 未実施 |
| FP8 / INT8 / LoRA / img2img | 未検証 |
| CUDA backend / VRAM / 時間 | 未測定 |
| AFM manual / augmented | 未実装 |
| auto 採用プロファイル | 未登録 |

ソース基準:
[Krea2](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/ldm/krea2/model.py)、
[Attention dispatch](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/ldm/modules/attention.py)、
[wrappers](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/patcher_extension.py)、
[samplers](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/samplers.py)。

例の workflow は後日のサーバー確認用です。実サーバーの `/object_info` で
loader / sampler / AFM ノードの入力を照合し、インストール済みモデル名へ置換して
初めて実行条件が揃います。静的 JSON 確認を推論成功と扱わないでください。

## 後日行う実機確認（今回は実施しない）

1. 実サーバーの ComfyUI commit、PyTorch/CUDA、GPU、モデルファイル名、
   量子化形式、起動時 backend オプションを記録する。
2. `/object_info` を保存し、`scripts/validate_workflow.py --object-info` で
   API 例を照合する。ファイル名と解像度以外の比較条件を固定する。
3. baseline を同じ seed で2回実行し、元 backend 自体の再現性を把握する。
4. discover を実行する。本体の `block_type=single` に相当する呼び出しのみ、
   `block_index`、`grid`、`text_tokens`、`image_tokens`、`img_slice`、
   `baseline_backend`、`preferred_backend`、`activation_dtype` を確認する。
5. observe を実行して baseline の変動範囲と比較する。`observed_calls > 0`、
   `edited_calls=0`、`status=ok` を確認。全fallbackは診断成功と扱わない。
6. 768正方形と長方形、Raw の CFG>1、連続run、通常LoRAなどを別条件として検証する。
   `[0] / [1] / [0,1] / [1,0]` の CFG 配置と selected batch rows を記録する。

API JSON を `/prompt` に渡す場合は `{"prompt": <API JSON>}` の形で包みます。
ブラウザー UI 用workflowのロード・実行確認は Task 5 に残しています。

## ログの読み方と限界

- `schema_version=4`, `model_family=krea2`, runごとに新しい `run_id`。
- `model_eval_index` は同じstep内のforwardごとに増加。CFGが別forwardの場合も区別する。
- `progress` は実際の `sample_sigmas` の評価点。対応しないsigmaを最近傍へ丸めない。
  未対応の場合は `unsupported_progress` と null を記録して観測を続ける。
- `cond_or_uncond` が欠ける場合、positiveと推測せず選択行を空にする。
  型が不正な場合も選択せず、値そのものをダンプせずnullと理由を記録する。
- 現行container API / 従来Tensor APIの両方を扱い、`peek()` による形状検査後に
  baseline dispatchが一度だけ消費する。observeの戻り値はbaselineと同じTensor。
- `max_logits_delta`, 周波数指標、編集backendなどの未測定値は null。
  診断版はlogits/FFTを計算しない。
- `wall_seconds` はCPU側の非同期診断区間。性能benchmarkやGPU時間ではない。
  VRAM peakは未測定としてnullを記録する。Task 5でCUDA同期と測定条件を追加する。
- モデルのファイル名はModelPatcherから確実に取得できないため `unknown` とする。
  API promptを実行記録と一緒に保存する。ComfyUI git情報が取得できない場合もunknown。
- マルチGPU、torch.compile、reference、複数フレーム、他Attentionパッチとの併用を
  保証しない。検出可能な条件は拒否する。
  AFM設置後に追加されたpost_input/QKV/outputパッチもforward前に拒否する。
- sampling例外とsummary保存失敗が同時に発生した場合、元のsampling例外を維持し、
  二次的なログ失敗をコンソールへ記録する。summaryの試行は1回。
- ノードからのJSONL保存先はComfyUI output内の `krea2_afm/` に限定する。
  絶対パス・traversal・解決後の範囲外パスを拒否する。内部Python APIの
  `AFMConfig.jsonl_path` は信頼された呼び出し元向けで、ノード入力用ではない。

既知の小課題: `validate_workflow.py --object-info` で、リンク先ノードがdictではない
壊れたJSONを渡すと、検証エラー一覧の代わりにAttributeErrorで終了する場合があります。
同梱の3つのAPI例はこの条件に該当しません。

## ソースだけで行う上流契約テスト

ComfyUIソースがある場合、環境変数 `KREA2_COMFY_SOURCE_ROOT` をそのルートへ設定し、
次を実行できます。通常のテストではこの3件をスキップします。

```console
python -m pytest tests/test_source_contract.py -q
```

ASTで読み出した上流の `AttentionTensorContainer`, `wrap_attn`, `WrapperExecutor`,
optionsコピー関数のみを実行します。ComfyUIのモデル・サーバーやGPU依存moduleは
ロードしません。固定revisionでは3件が通過しています。別版を指定した結果は、
その版のdispatcher契約に限った証拠です。

参照した公式workflow:
[image_krea2_turbo_t2i.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_krea2_turbo_t2i.json)。
2026-09-30取得版の SHA-256:
`8688a4fe39292924761b184d9bc1ad6a62817c55fb0de704becca041150fac69`。
