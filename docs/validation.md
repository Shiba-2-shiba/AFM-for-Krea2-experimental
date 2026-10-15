# 検証と実機確認への引き継ぎ

> 2026-10-15 refactor: ユーザー指定により今回の差分は未検証です。以前のPASS記録と区別し、[今回の実装引き継ぎ](refactoring-implementation-2026-10-15.md)を参照してください。

今回は実モデル推論・GPU測定を実施していません。Task 2–4の編集コードと、
Task 5のローカル検証/引き継ぎツールを実装しています。

| 項目 | 現時点の証拠 |
| --- | --- |
| manual / augmented / dense reference | 合成TensorのCPU FP32比較 |
| DC・長方形・線形性・明示scale・head復元 | CPU回帰テスト |
| CFG・container・例外時atomic fallback | 最小hostスタブと上流dispatcher契約テスト |
| FP32を指定した計算 | CPU BF16 autocast下でもFP32 referenceと比較 |
| メモリ | Tensor生存期間を含む見積もりとchunk縮小テスト。実測VRAMではない |
| V3/API/UI workflow | schema・JSON・リンク/設定の静的確認。実サーバーロード未確認 |
| Turbo / Raw、量子化、LoRA、img2img | 実機未検証 |
| 速度・peak VRAM・生成画像の品質 | 未測定 |
| auto採用profile | 空。未確認環境はmanual |

## ローカル確認

Python 3.10.11、PyTorch 2.10.0+cpu、pytest 9.0.2で確認します。
2026-09-30の再開後検証は、上流ソース契約を含めて241件PASS、GPU1件除外。
構文チェック、API graph検証、7条件dry-runのログ対応確認も成功しています。
専用lint/typecheckツールは未導入です。

```console
python -m pytest tests -q
python -m compileall -q afm nodes.py __init__.py scripts
python scripts/validate_workflow.py examples/krea2_afm_edit_api.json
```

通常実行はGPU markerを除外します。ComfyUIソースがあれば環境変数
`KREA2_COMFY_SOURCE_ROOT` にルートを設定して、`tests/test_source_contract.py` の3件も
実行できます。上流のcontainer / dispatcher / wrapper定義だけをASTで読み出す方式で、
モデルやGPU依存moduleはロードしません。ソースがなければ、この3件はスキップします。

## 診断の意味

- `schema_version=4`, `model_family=krea2`。各runの開始とsummaryを `run_id` で結びます。
- `model_eval_index` は同じstep内のforwardごとに増加。CFGが別forwardの場合も区別します。
- `progress` は実際の `sample_sigmas` の評価点。未知のsigmaを最近傍へ丸めません。
- CFG metadataが欠落/不正ならpositiveと仮定しません。編集はraiseまたは元出力へのfallback。
- 編集は画像Query→テキストKeyのlogitsだけに適用しますが、正規化は全Keyで行います。
- 元backendは1回だけ呼び、containerは1回だけ消費します。失敗時は同じbaselineへ戻ります。
- sampled spectral診断の `rho_text_conditional` は、テキスト内softmaxのtop-K平均集中度から
  空間平均を引き、そのスペクトルのHF比を測るものです。画像解像度/追従性の指標ではありません。
- `text_attention_mass` は全Keyで正規化した確率のうちテキストKeyに割り当てられた量です。
- diagnostics off時のaugmented経路は観測用logitsを再生成しません。
- kernel比較は同じQKVの選択head/画像行で実行。manual referenceは同じQKVのFP32 castです。
  別runの最終画像だけでkernelの等価性を判定しません。
- 未計測値はnull。モデルファイル名はModelPatcherから確実に取得できないのでunknown。
  実際のAPI promptをmanifestと一緒に保存してください。

ログ内の `branch` は比較対象のbatch行番号です。`attention` イベントには
`cond_or_uncond` と `selected_batch_rows` も記録するため、CFG配置と併せて解釈します。
run間で比較する際は同じCFG配置・batch条件を維持してください。

## 後日行う実機確認（今回は実行しない）

1. ComfyUI commit、PyTorch/CUDA、GPU、モデルファイル名/量子化、起動backendを記録します。
2. 実サーバー `/object_info` を保存し、例のモデル名を置換して照合します。
3. true baselineを同一seedで2回実行し、元backend自体の変動を確認します。
4. discover/observeでtxtfusion除外、block、grid、img_slice、preferred backendを確認します。
5. zero original、manual zero、augmented zeroを分けて測定します。
6. manual/augmentedの .05/.1/.2 と同一QKV比較を、まず小解像度、768正方形、長方形で行います。
7. Raw CFG>1、通常LoRA、img2img、量子化/backend形式を別条件として確認します。
8. 各方式1回warm-up後、順序を交互にして5回ずつ測定します。

```console
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 --matrix examples/validation_matrix.json --out-dir outputs/krea2_afm
```

このコマンドはdry-runです。後日の生成時だけ `--execute` を付けます。
`--unet-name`, `--clip-name`, `--vae-name` で実際のファイル名を指定できます。
API JSONを自分で送信する場合は `{"prompt": <API JSON>}` の形で `/prompt` に渡します。
生成成功と編集適用成功を混同せず、serverのJSONL summaryも回収してください。

manifestの `invocation_id` はrunner呼び出しごとに変わります。各条件の
`server_log_relative` とpromptの `jsonl_path` は
`krea2_afm/<条件ID>-<invocation_id>.jsonl` で一致します。
`--server-log-dir` はサーバーの `output/krea2_afm` をクライアント側へ
マウント/同期したディレクトリです。今回のファイルがなければ
`server_metrics=unverified` とし、以前のログで代用しません。

`benchmark=true` のときだけ、server wrapperでCUDA同期とpeak計測を行います。
CPU/defaultでは `cuda_synchronized=false`、VRAMはnullです。clientプロセスのCUDA統計を
serverの値として使いません。coldとwarmは分け、benchmark中は比較/spectral診断をoffにします。
全fallback、OOM/NaN、未測定、効果判定不能の条件をautoへ採用しません。

数値の閾値と性能条件は計画書6.2/6.3の通りです。
`afm/qualified_profiles.json` は自動更新せず空のままにしてあります。
profileを将来追加するには、同じ環境/精度/形状/chunk/フィルタ設定の数値reportと
server benchmarkを揃え、閾値を満たす必要があります。FP32だけの合格をBF16/FP16へ転用しません。
単一profileの数値reportはactivation dtypeを統一し、そのdtypeでstrength
`.05/.1/.2` の証拠を揃えます。dtypeの別名は正規化しますが、FP16とBF16の
混在reportは拒否します。

## 出典

[Krea2モデル](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/ldm/krea2/model.py)、
[Attention API](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/ldm/modules/attention.py)、
[wrapper](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/patcher_extension.py)、
[sampler](https://github.com/Comfy-Org/ComfyUI/blob/9d80841aa1990305cc7a280c5ea505f317efcc2d/comfy/samplers.py)。

[PyTorch 2.10 SDPA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)の
明示scale、dropout=0、math経路のFP32中間計算を基準にしています。SDPA呼び出し成功だけで
FlashAttention使用や高速化を主張しません。

[公式Turbo workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_krea2_turbo_t2i.json)
の2026-09-30取得版SHA-256:
`8688a4fe39292924761b184d9bc1ad6a62817c55fb0de704becca041150fac69`。
例では任意LoRAとprompt enhancementを省き、標準loader/conditioning/sampler構成を使っています。

## 2026-10-15の変更を検証する際の注意

今回のリファクタリング差分は未検証です。以前の241件PASSを新方式の証拠に使用しないでください。schema 5のraw/centered/radial-binned指標、保存性・物理head診断、旧schema 4のlegacy reader、総注意量保存manual、cache拒否、manifest 2/paired集計を追加しています。手順は[実装引き継ぎ](refactoring-implementation-2026-10-15.md)と[計画書](superpowers/plans/2026-10-15-krea2-afm-refactoring.md)を参照してください。テストコードは整備していますが、今回は実行していません。
