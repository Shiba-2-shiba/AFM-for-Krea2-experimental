> 最新: 2026-10-15の実機検証を末尾に追記。22生成条件・7拒否試験と人物26枚を確認。回帰342成功/19失敗のため全体未合格。以下の旧記録の「未実施」は実装当時の状態。

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

## 2026-10-15 起動中ComfyUIでの検証結果

**判定: 実機の限定構成は確認できたが、回帰suiteは未合格。T0–T5の総合合格・T6全構成完了・T7の画質改善・T8のprofile採用は表明しない。** 製品コードとテストの修正は行っていない。

対象HEADは `31df9b01b9a1fee6c3cadc3c6274d5eb90b262e2`。
`.omx/plans/2026-10-15-krea2-afm-refactoring.md` は `docs/superpowers/plans/2026-10-15-krea2-afm-refactoring.md` を正本として参照しており、両者の範囲を照合した。

### 実環境と再現用成果物

- 起動済みサーバー: http://127.0.0.1:8000（発見時localhost:8000）。ComfyUI 0.37.0。
- 当日ログで確認したsource: `C:/Users/inott/ComfyUI-Installs/ComfyUI/ComfyUI`、git HEAD `15ef24d1c0333a3eba56c5cd153d8db65363ff8f`。
- 実行環境: Python 3.12.11、PyTorch 2.12.0+rocm7.14.0、AMD Radeon AI PRO R9700、約32 GiB、pytorch attention。
- ユーザー指定model: `D:/StabilityMatrix-win-x64/Data/Models/DiffusionModels/intorealismAsian_k2JAVFLASHV1.safetensors`。headerはINT8 convrot、元dtype BF16。実attentionはBF16。重み量子化とattention dtypeを区別する。
- Encoder: `qwen3vl_4b_fp8_scaled.safetensors`、VAE: `qwen_image_vae.safetensors`。いずれもlive loader choicesから確認。
- 既存API例のコピーを使用。seed=20260930、Euler/simple、8 steps、512×512、CFG=1を基本とし、640×512とCFG=2のpositive/negative/bothを追加。
- 新しい依存・モデルdownload・ComfyUI再起動は行っていない。

証拠はローカルの [outputs/validation-20261015](../outputs/validation-20261015/) に保存。主なファイル:

- [evidence.json](../outputs/validation-20261015/evidence.json): 22条件のsummary、保存誤差、coverage、画像hash/差分、8項目のassertion結果。
- [direct-live/manifest.json](../outputs/validation-20261015/direct-live/manifest.json): 最初の13条件のprompt、prompt ID、history、画像、今回のrun_idと対応するserver report。
- [extended/manifest.json](../outputs/validation-20261015/extended/manifest.json): 追加9条件。prompt、history、画像とserver log先。
- [server-logs](../outputs/validation-20261015/server-logs/): 今回の固有runに対応するschema 5 JSONLのコピー。
- [negative/results.json](../outputs/validation-20261015/negative/results.json): 想定拒否7件のエラー履歴。
- [comfy-pytest.log](../outputs/validation-20261015/comfy-pytest.log)、[pytest.log](../outputs/validation-20261015/pytest.log): 2環境の回帰結果。
- [paired-analysis.json](../outputs/validation-20261015/paired-analysis.json)、[CSV](../outputs/validation-20261015/paired-analysis.csv): 既存集計CLIの出力。1 prompt/seedのsmokeであり、効果評価の証拠にしない。
- [analyze_validation.py](../outputs/validation-20261015/analyze_validation.py): 収集済みログ・画像のオフライン再集計とassertion。実行時のHTTP代替transportはmanifestに明記。

### 回帰・静的確認

通常Python（3.12.10 / torch 2.9.1+rocm7.2.1）で **342 passed, 19 failed, 1 deselected / 4.27秒**。
ComfyUIのPython/PyTorchでも **342 passed, 19 failed, 1 deselected / 4.73秒** と同じ19件を再現。pytestは既存の通常Pythonのsite-packagesを末尾に追加して利用し、ComfyUIのtorchを維持した。追加インストールなし。
両実行でKREA2_COMFY_SOURCE_ROOTを当日のsourceに設定。source-contract 3件はskipせず実行。旧参照commit 9d80841の別treeは今回再実行していない。
最初の実行の一時フォルダーPermissionErrorは、リポジトリ内の新規basetempを指定して解消した。以後の19件にはfixture setup errorは含まれない。
`compileall` と `git diff --check` は成功。専用ruff/mypy/pyrightは当該ComfyUI環境に存在せず未実施。

19件を19個の製品不具合とは数えない。以下の契約不一致とテスト側の問題を分けて修正する必要がある。

| 分類 | 失敗数 | 観測と対応箇所 |
|---|---:|---|
| 共通入力検証 | 6 | shape例外の文言不一致2件、QKV dtype不一致の事前拒否欠落2件、scale=0/負値を拒否しない2件。`afm/attention_math.py:10–29` とjoint/augmentedのテスト。dtype不一致でmanualは黙ってcastし、augmentedは後段SDPAでRuntimeErrorになる。 |
| 保存helperの例外文言 | 2 | 実装は正しくFP32以外を拒否するが、testのregex float32に対しメッセージがFP32。`afm/attention_math.py:63`、`tests/test_mass_preservation.py:73`。 |
| auto選択と理由 | 2 | preserve + autoにnormalization_requires_manualを返す分岐がない。`afm/selection.py:66–86`。合成のjoint signature/profileを渡すtestではaugmentedを選ぶ。実環境の空profileではmanualだが理由はprofile_not_qualified。実runtimeのfilter signatureはmodeを含み、今回未補正augmentedの誤実行は観測していない。 |
| spectral期待値 | 4 | FP32作業結果とFP64 oracleのraw差約9.66e-10に対し3件の許容差が1e-10。checkerboardは実測9.5934e-7に対し期待3.83e-6。入力が0/1の振幅である点を含め独立期待値を見直す。`afm/observation.py:34–40`、`tests/test_observation.py:83–117`。 |
| runner mockのschema | 2 | diagnostic_mode=offが3入力を加えるがmockのAFM object_infoに定義がなく、unknown inputで送信前に拒否。`scripts/run_afm_matrix.py:210–225`、`tests/test_matrix_runner.py:201,380`。実サーバーのschemaには存在し、全prompt照合を通過。 |
| 診断workspaceの期待 | 1 | 実装は物理headを逐次測定してpeakを1head分とするが、testは選択head数とともに見積もりが増えることを要求。`afm/memory.py:69–70`、`tests/test_memory.py:191`。本結果だけで過小見積もりとは断定しない。 |
| legacy benchmarkの期待 | 1 | 実経路情報のないschema 4 fixtureについて、新readerのmanual配列が空。`scripts/summarize_krea2_afm.py:390–420`、`tests/test_reports.py:244`。表示互換と資格判定の契約を整合させる必要がある。 |
| legacy byte hash | 1 | Windows checkoutのCRLFによりbytes hashが不一致。5 fixtureすべて、LFへ正規化したhashは固定値と一致した。`tests/test_workflows.py:13–18,45–48`。意味上のworkflow破損ではないがbyte固定のgateは未合格。 |

### 実機で確認できた範囲

生成22条件はすべてterminal success、画像を回収。AFMの成功runはfallback/NaN/OOMがすべて0。
最初のbaselineは直前の疎通生成のcacheを利用したため、それ自体を独立反復と数えていない。編集条件の後のbaselineではKSamplerが再実行され、同じ画像になった。

- baselineとoff / zero original / discover / observe / spectral observeはRGB画素単位で一致。
- discoverは8 model evaluations × 28本体block = 224 attention callsを記録。observeのscope 0–3は32 calls、block 0だけの編集は8 calls。
- 拡張条件ではblock 0,14,27、物理head 0,16,32,47、step 0–7を照合。通常96件、CFG bothは192件のspectral semantic keyが一意に揃い、対応attentionイベント・edit_appliedと一致。
- 以下は**同じcall内**のFP32作業値に対する保存誤差。各値は測定したhead/queryの最大。全head・全modelへの上限保証ではない。

| 条件 | edited calls | spectral events | text mass max abs | image probability max abs |
|---|---:|---:|---:|---:|
| multi-preserve | 24 | 96 | 7.749e-7 | 2.682e-7 |
| multi-preserve-auto | 24 | 96 | 7.749e-7 | 2.682e-7 |
| rectangle-preserve | 24 | 96 | 1.013e-6 | 4.172e-7 |
| cfg-positive | 24 | 96 | 7.749e-7 | 3.576e-7 |
| cfg-negative | 24 | 96 | 4.768e-7 | 8.941e-8 |
| cfg-both | 24 | 192 | 9.537e-7 | 2.980e-7 |

いずれも計画の1e-5以内。text massは0だけでなく最大約0.9947まで含む。対照のmulti-jointではtext mass最大変化0.9834、image probability最大変化0.7776であり、保存方式との差を実測できた。jointがmassを変えること自体は既定設計どおり。

- preserveの診断ON/OFF、manual/autoは画像が画素単位で一致。測定イベントの未選択branch/text行保持はtrue。
- BF16の同一QKV manual FP32対augmented比較は32件。最大relL2=0.003743、scaledMax=0.010542で暫定BF16閾値0.02/0.05以内。ただしhead 0 / block 0のみでeffect_resolved=falseの条件を含み、strength .1/.2や性能資格の証拠がないためprofileへ採用しない。
- manual zero / augmented zeroはbaselineと画素一致しない（平均絶対RGB差1.578 / 1.410、最大117 / 135）。仕様どおり別kernelの数値対照であり、zero originalと混同しない。これらの画像差をAFM効果とは呼ばない。
- preserve+augmented、preserve+compare_kernels、範囲外head48を想定どおり拒否。EasyCacheとLazyCacheはAFMの前後の両順序、計4件すべてunsupported_model_skip_cacheで拒否。
- 想定拒否7件後もbaselineを再実行して成功。prompt IDは `8ec59d86-c9d4-4c58-b723-43a59fd0b196`。検証終了時のqueue_running / queue_pendingはともに0件。

### 未解決・未実施

付属runnerはlocalhostと127.0.0.1の両方で、urllibによる全object_info取得がWinError 10054で失敗した。system_statsとNode fetchの全catalog取得は成功したため、Node fetchで同じ準備済みpromptを順次送信・terminal historyを確認し、画像をviewから回収した。原因の確定とrunner自身での一貫した実機完走は未完了。失敗manifestはlive / live-ipv4、成功manifestはdirect-live / extendedへ分離した。

centered metricはheadによってundefinedとなる。拡張positive系は96件中8件、negative-onlyは96件すべてが未定義。構造coverageとmetric定義状態を分離し、未定義値を0や有意なHF改善として集計しない。旧smokeのhead0はtext mass=0があり、保存誤差0だけでは十分な検証にならなかったため複数head/blockを追加した。

paired集計CLIは完走したが、smokeは1 prompt/seedしかなく、rawの差は約1e-17、centeredはmetric_undefinedで除外される。単一promptのbootstrap CIから一般的改善は結論しない。またsummary-only discover/observeはspectral keyがないため同CLIのcoverage=failとなるが、生成失敗や観測hook未実行とは区別する。

768²/768×512、奇数latent/padding、5D、batch>1、CFG別forward、Raw/Base別model、LoRA、img2img、追加sampler、FP16、他量子化/backendは実機未確認。今回のCFG=2は指定FLASHモデルの接続試験でありRawモデル検証の代用ではない。ブラウザーでのUI保存・再ロードは未実施。
同期benchmark/peak VRAM/反復warm測定は実施していない。ログのwall_secondsはunsynchronizedかつ診断込みで、速度比較の証拠にしない。qualified_profiles.jsonは空を維持。

独立code-reviewer/architectは、設定されたgpt-5.4がアカウント非対応というエラーで両方とも起動できなかった（independent review unavailable）。本検証は独立レビュー承認やmerge-ready判定ではない。

## 人物・高強度の追加比較（2026-10-15）

ユーザーの追加依頼により、指定INT8モデルと起動中ComfyUIで顔中心・手を含む人物の2構図を生成。各構図でseedを固定し、manual・全28block、強度0/.05/.2/.5/1/2をjointとpreserve_text_massで比較した。元経路baseline各1枚を含め26枚すべて成功。各編集runは224 edited calls、fallback/NaN/OOMは0。構図ごとにchunk設定を統一（顔head=2、人物・手head=8、query=128、workspace=512MiB）。

[人物比較ページ](../outputs/portrait-evaluation-20261015/review/index.html)、[顔の比較一覧](../outputs/portrait-evaluation-20261015/review/00_portrait_start.png)、[顔の拡大](../outputs/portrait-evaluation-20261015/review/01_face_closeup.png)、[手の比較](../outputs/portrait-evaluation-20261015/review/04_hands_start.png)を用意。元PNG26枚はhash一致のコピー、固定領域の拡大は通常の補間のみ。20組の0対各強度比較は、strengthとログ/保存先を除くprompt一致を検査した。

画像の変化を目視し、ユーザーの主観評価も取得した（下記）。2 prompt・各1 seedの探索であり一般的改善や最適値の判定ではない。高強度の画像も選別せず残した。既存の回帰19失敗・独立レビュー未実施という判定は変更していない。プロンプト、画像、JSONL、全設定・検証結果はoutputs/portrait-evaluation-20261015に保存した。

## 人物比較の観察と現時点の結論

ユーザーの観察は「総注意量保存の方が良い印象で、その場合は強度は2ぐらい必要。従来方式だと構図が変わり、強度2だと崩れが発生している」。人物・手のjoint強度2ではカップが増える例も確認した。これはユーザーの定性的評価として記録し、数値スコアや複数seedでの一般的改善へ読み替えない。

現時点の有力候補は `normalization_mode=preserve_text_mass, implementation=manual, strength=2.0, block_scope=all`。検証した2構図・各1 seed・指定INT8モデル・8 steps/CFG 1の範囲に限る。注意量保存は同じcall内の性質であり、最終構図の不変を保証しない。ノードの既定値、qualified_profiles.json、自動採用条件は変更していない。

次の検証案は保存方式の強度1.0/1.5/2.0を複数seedで比較すること。1.5の生成や複数seed比較は今回未実施。画質の一般的改善、最適強度、性能資格を確定する前の探索結果である。

## Gitに保存した検証資料

`outputs/` は既存の.gitignoreで追跡外のため、上記outputsリンクは検証したPC上のローカル資料を指す。リモートからも確認できる資料を [validation-results/2026-10-15](validation-results/2026-10-15/) に保存した。

- [機械可読の結果要約](validation-results/2026-10-15/summary.json): 環境、22生成条件、人物26条件、編集回数、保存誤差、画像hash、既知の未合格事項。
- [ユーザーの観察](validation-results/2026-10-15/user-observation.json): 保存方式2.0を候補とする理由と未実施の次検証案。
- [顔の比較](validation-results/2026-10-15/comparisons/portrait-strength-comparison.png) / [顔の拡大](validation-results/2026-10-15/comparisons/portrait-face-crops.png)。
- [人物・手の比較](validation-results/2026-10-15/comparisons/hands-strength-comparison.png) / [手の拡大](validation-results/2026-10-15/comparisons/hands-crops.png)。各一覧は上段joint、下段保存方式、左から強度0/0.5/2.0。
- [初期smokeの対照比較](validation-results/2026-10-15/comparisons/smoke-controls.png)。
- [smoke 22条件の実行prompt](validation-results/2026-10-15/prompts/smoke/) / [人物26条件の実行prompt](validation-results/2026-10-15/prompts/portrait/): 送信時のJSON。モデル名・seed・全AFM入力を再現できる。
- [コミット前の回帰ログ](validation-results/2026-10-15/logs/precommit-pytest.log): ComfyUI Python/PyTorchで再実行し **342 passed, 19 failed, 1 deselected / 6.50秒**。既知の19件が再現し、修正は行っていない。
- [成果物のSHA256一覧](validation-results/2026-10-15/artifacts.json)。元PNG・全JSONL/history・評価HTML・残りの比較画像はローカルoutputsに保持する。元PNGのhashは要約に記録した。

製品コードとテストは変更せず、今回は検証記録と共有用証拠を追加した。全体の回帰gateは未合格、独立レビューは環境制約で未実施、GPU性能と一般的画質改善は未確定のままである。

## 修正後の実機検証計画（2026-10-15）

次の実機検証は [AFM修正後のCodex実機検証計画](hardware-validation-plan-2026-10-15.md) を参照。baselineとsame-kernel zero、4 prompt×4 seed×5条件の80画像Pilot、盲検評価、任意の16画像frequency-neutral対照、同期性能測定を段階的に行う計画であり、今回はGPU実行しない。修正後CPU検証と新しい実機証拠を上記の履歴から分離し、既存の失敗ログ・テスト数・画像を上書きしない。
