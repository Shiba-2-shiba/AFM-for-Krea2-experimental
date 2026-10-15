# AFM修正後のCodex実機検証計画

作成日: 2026-10-15。対象: 修正後のソースと、後日明示的に選定するComfyUI実機。

**この文書は実行計画であり、実機合格の記録ではない。今回の作業ではGPU推論・モデル取得・サーバー送信を行わない。** runnerの既定はdry-runを維持する。`--execute`、GPU testのopt-in、サーバー再起動は、この計画を読んだだけでは実行しない。

## 1. 今回確認することと、過去の証拠

- 修正後のCPU契約を固定し、実際にロードされたソースを確認してから、同じモデル・backendでbaselineとsame-kernel zeroを取り直す。
- 主実験は **4 prompt × 4 seed × 5条件 = 80画像**。original、preserve/manual zero、preserve強度1.0/1.5/2.0を比較する。人による画像評価とattention数値集計は別の結果として提出する。
- 任意のfrequency-neutral対照は別matrixの16画像。追加する場合のみ合計96画像とする。80画像の既定計画へ自動追加しない。
- 互換性、画質、同期性能、auto/profile資格は別gate。CPUテストの修正・成功をGPU資格や画質改善と呼ばない。

[従来の検証記録](validation.md)では、HEAD `31df9b01b9a1fee6c3cadc3c6274d5eb90b262e2` に対し、22生成条件、7拒否試験、人物26画像を確認した。一方、2つのPython環境で342成功・19失敗・GPU 1件除外を再現した。19失敗には入力検証、preserve/autoの選択理由、FP32結果とFP64期待値、runnerのmock schema、逐次head診断のworkspace、legacy表示、CRLF byte hashの契約不一致が含まれる。これらは19個の独立した実機障害を意味しないが、当時の回帰gateは未合格である。

同日の人物比較は2 prompt・各1 seedの探索であり、ユーザーの「保存方式は強度2程度が良い」という観察を今回の候補選定に使う。元の画像、失敗ログ、過去のテスト数は書き換えない。以前の実機ログを修正後の実行証拠へ付け替えず、新しい日付・出力ディレクトリに保存する。

今回の修正対象は、QKV dtypeの厳密一致、正かつ有限の実数scale（bool/string/tensorはbackend入口でも拒否）、preserve+autoのmanual固定、独立spectral oracleの期待値、逐次head診断のpeak見積もりテスト、legacy実測値の未帰属表示、CRLFを正規化したfixture hashである。productionのspectral指標や許容差をテスト通過目的で緩めない。legacy値の表示互換をprofile資格に使わず、既定値・既存profileは変更しない。

## 2. 実行前に揃えるもの

1. 修正後checkout、git HEAD、dirty diff、CPU検証ログ。変更が残る場合はHEADだけで版を特定しない。
2. 使用を承認された実機、ComfyUIの絶対パスとPython実行ファイル、URL、出力ディレクトリ。以前のホストはWindows / ComfyUI 0.37.0 / AMD Radeon AI PRO R9700 / PyTorch 2.12.0+rocm7.14.0 / pytorch attentionだったが、現在値を再取得する。
3. 実機に既存の `intorealismAsian_k2JAVFLASHV1.safetensors`、`qwen3vl_4b_fp8_scaled.safetensors`、`qwen_image_vae.safetensors`。実際のloader choiceを確認して指定する。名前が異なれば無言で別モデルへ替えず、matrixの新revisionとして固定する。
4. ComfyUIの `output/krea2_afm` をクライアントから読めるマウントまたは同期先。`--server-log-dir` はこのディレクトリそのものを指定する。
5. Python 3.10以上、当該環境のtorchとpytest。依存がなければ不足を報告する。テストのためにComfyUIのtorchを置換しない。GPU testは通常のpytestから除外されている。
6. 利用者の他のジョブがなく、連続実行を許可された時間帯。既存queueを勝手に削除しない。必要な再起動・キャッシュ設定変更は先に了承を取る。

以下のPowerShell例はリポジトリrootで実行する。パスは当該ホストで確認した値へ置き換える。`python.exe` の所在やURLを過去の記録だけから推定しない。

```powershell
$Python = '<確認したComfyUI環境のpython.exe絶対パス>'
$ComfyRoot = '<確認したComfyUIソースの絶対パス>'
$ComfyUrl = '<確認したURL。以前のホストはhttp://127.0.0.1:8000>'
$ServerLogDir = '<クライアントから読めるoutput/krea2_afm絶対パス>'
$Out = Join-Path (Get-Location).Path 'outputs/afm-pilot-corrected-YYYYMMDD-HHMMSS'
New-Item -ItemType Directory -Force "$Out/preflight" | Out-Null
$env:KREA2_COMFY_SOURCE_ROOT = $ComfyRoot
```

## 3. Gate A: CPU・静的検査・送信なしの準備

各コマンドの終了コードを確認し、失敗があれば実機へ進まない。修正した対象テストだけでなく全suiteを実行する。ComfyUIの実行中revisionと、旧設計の参照revision `9d80841aa1990305cc7a280c5ea505f317efcc2d` のソース契約は別に記録する。以前の実機sourceは `15ef24d1c0333a3eba56c5cd153d8db65363ff8f` であり、2つのrevisionを同一の検証結果にしない。どのtreeを使ったCPUログかを残し、参照treeがなければ取得済みと装わず未確認とする。

```powershell
& $Python -m pytest tests -q --basetemp "$Out/pytest-tmp" 2>&1 |
  Tee-Object "$Out/preflight/pytest.log"
if ($LASTEXITCODE -ne 0) { throw 'CPU regression failed' }
& $Python -m compileall -q afm nodes.py __init__.py scripts
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
git diff --check
if ($LASTEXITCODE -ne 0) { throw 'Diff check failed' }
& $Python scripts/validate_workflow.py examples/krea2_afm_experiment_api.json
& $Python scripts/run_afm_matrix.py --help
& $Python scripts/analyze_afm_experiment.py --help
& $Python scripts/run_afm_matrix.py --comfy-url $ComfyUrl `
  --matrix examples/validation_matrix_afm_pilot.json `
  --out-dir "$Out/pilot-planned" `
  --unet-name intorealismAsian_k2JAVFLASHV1.safetensors `
  --clip-name qwen3vl_4b_fp8_scaled.safetensors `
  --vae-name qwen_image_vae.safetensors
```

このrunnerコマンドには `--execute` がないため、URLを引数にしていても通信・送信しない。matrix入力はschema 1、生成manifestはschema 2。静的検査とdry-runはGPU検証ではない。`ruff` / `mypy` / `pyright` が既存環境にある場合のみ該当範囲で実行し、なければ未実施と記す。

dry-run成果物を読み、以下をassertする。

- `mode=dry_run`、80件、全件 `status=not_submitted`、`server_metrics=unverified`。`actual_implementation=unverified` と `runtime_run_id=null` を成功値に書き換えない。
- 16個のprompt/seed pairに5条件がちょうど1件ずつあり、条件以外のモデル・prompt・seed・sampling・shape・chunkが一致する。
- 各promptの `jsonl_path` とmanifestの `server_log_relative` が一致し、呼び出し固有の `invocation_id` を含む。別実行の既存JSONLを割り当てない。
- `expected_coverage` は予定であり実測ではない。original/offや診断offの空集合を「spectral coverage合格」に変換しない。

## 4. Gate B: 配備・runtime来歴と接続

GPU実行を開始する前に、次の内容を `preflight/` に保存する。runnerが自動収集していない項目は別ファイルとして収集し、未収集を明記する。

| ファイル | 必須内容 |
|---|---|
| `source.json` / `source.patch` | AFM HEAD、branch、git status、dirty diffのSHA-256、`nodes.py`・`__init__.py`・`afm/*.py`・`scripts/*.py`・profile JSONの相対パスとbytes SHA-256 |
| `runtime.json` / `runtime-files.json` | server PID・起動時刻・Python実行ファイル、実ロード先custom-nodeパス、配備した実行対象ファイルのSHA-256、ComfyUI HEAD/dirty状態、torch・CUDA/ROCm・driver・GPU・起動backend・精度設定 |
| `model-files.json` | diffusion/encoder/VAEの実ファイル名・サイズ・SHA-256。重みのINT8/FP8とattentionのBF16/FP16を別欄に記録 |
| `object_info.json` / `system_stats.json` | 今回接続したサーバーの応答、取得時刻、URL |
| `artifact-hashes.json` | matrix、解決済みprompt、manifest、history、元PNG、JSONL、集計JSON/CSV、ratings/keyのSHA-256 |

bytes hashの例は `Get-FileHash -Algorithm SHA256 <実ファイル>`。`resolved_workflow_sha256` はrunnerが解決後graphをcanonical JSON化したhashであり、ファイルbytes hashやモデル重みhashの代わりではない。モデル名だけで重み同一とは判定しない。

**配備ファイルが一致していても、古いserver processが旧moduleを保持していればruntime一致ではない。** 更新後に当該processが対象ソースをロードした証拠を起動ログ・ロード先と結び付ける。確認できなければ停止し、必要な再起動の了承を得る。ディスク上のhashを「実行中コードのhash」と偽装しない。

接続確認を始める承認後、読み取りのみを実施する。

```powershell
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$ObjectInfo = Invoke-RestMethod "$ComfyUrl/object_info" | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText("$Out/preflight/object_info.json", $ObjectInfo, $Utf8NoBom)
$SystemStats = Invoke-RestMethod "$ComfyUrl/system_stats" | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText("$Out/preflight/system_stats.json", $SystemStats, $Utf8NoBom)
Get-ChildItem "$Out/pilot-planned/prompts/*.json" | ForEach-Object {
  & $Python scripts/validate_workflow.py $_.FullName `
    --object-info "$Out/preflight/object_info.json"
  if ($LASTEXITCODE -ne 0) { throw "Workflow validation failed: $($_.Name)" }
}
```

Python側がUTF-8として読むため、上記はPowerShellのversionによるBOM差を避けて保存する。UIでは旧API/旧UI workflowをロードし、追加optional入力・既存widget順・保存後再ロードを確認する。UI保存成功と推論成功は分ける。

以前はrunnerのurllibによる `/object_info` 取得がWinError 10054で失敗し、別transportで生成した。今回はまずrunner自身で小さいgateを完走させる。同じ障害があれば失敗manifestを保持し、transportを調査する。別transportを使う場合は明示的に別の実行系列・hash・prompt IDを記録し、「付属runner完走」とは報告しない。

## 5. Gate C: baselineとsame-kernel zeroの取り直し

主実験より前に、1 prompt・1 seed・512×512・batch 1・Euler/simple・8 steps・CFG 1で以下を順に確認する。基準例は `examples/krea2_afm_baseline_api.json` と `examples/krea2_afm_experiment_api.json`。必要なgate matrixを出力先に準備し、workflow参照を実在する絶対パスに固定する。

| 条件 | 設定・確認点 |
|---|---|
| true baseline × 2 | AFM nodeなし。独立にKSamplerが実行されたことを確認し、同seedのnative変動を測る |
| off / zero original | `mode=off`、および `mode=edit,strength=0,zero_strength_mode=original`。native出力との差を測る |
| observe | 編集なし。spectral ON/OFFを分け、nativeとの出力同等性を確認 |
| preserve manual zero | `mode=edit,normalization_mode=preserve_text_mass,implementation=manual,strength=0,zero_strength_mode=manual`。主実験と同一kernel・scope・chunk |
| preserve 1 / 1.5 / 2 | `schedule=curve,preserve_dc=true`、manual、`block_scope=all`。同じcall内の保存性とscopeを確認 |
| 回復baseline | 上記または想定例外後に再実行し、状態漏れがないことを確認 |

編集scopeは全body block、診断は `diagnostic_block_scope=0,13,27`、物理head `0,16,32,47`、全stepを使う。これらのindexは以前の28 block/48 headモデルに基づく。今回のdiscover/runtimeで寸法が一致しなければ送信せず、予定集合を再固定する。headを結果が良かったものへ後から差し替えない。初回Pilotも同じ512²に固定する。後日768²へ拡張する場合は、baseline・manual zero・保存方式の候補からshape別gateを取り直す。

`strength=0` でも `zero_strength_mode=manual` はmanual計算を通る。originalとmanual zeroの画像差はkernel/丸めの差を含み、AFMの非zero効果ではない。効果の主対照は **同じpreserve/manual経路のzero** とする。joint zeroやaugmented zeroで代用しない。preserve+augmentedおよびpreserve+compare_kernelsは対応対象外として拒否を確認する。

baselineの画像が同じでもcache hitだけでは反復証拠にならない。historyの `execution_cached`、サーバーログ、実際のsampler実行を照合する。cache回避のためにseedやpromptを変えて同seed反復と数えない。再計算を確認できない場合は `not_independent` とし、了承のあるcache設定で取り直す。

### 送信方法と停止単位

現runnerは全matrix内の失敗後も残り条件へ進むため、停止gateとしてそのまま80件を一括送信しない。凍結matrixから1件matrixを作り、元のrun ID・全設定・絶対workflow参照を維持する。下記は**後日実行許可を得てから**使うCLIであり、準備段階では末尾の `--execute` を付けない。

```powershell
& $Python scripts/run_afm_matrix.py --comfy-url $ComfyUrl `
  --matrix "$Out/one-run/<確認済みrun-id>.json" `
  --out-dir "$Out/actual/<確認済みrun-id>" `
  --server-log-dir $ServerLogDir `
  --unet-name intorealismAsian_k2JAVFLASHV1.safetensors `
  --clip-name qwen3vl_4b_fp8_scaled.safetensors `
  --vae-name qwen_image_vae.safetensors --timeout 1800 --execute
```

各件でterminal history、画像、今回のJSONL、gateのassertを確認してから次を送る。timeout/応答切断は未送信と決めつけず、prompt ID/history/queueを調べ、重複送信を避ける。合格条件は次の全て。

- runtime statusがok、fallback/NaN/OOMが0。保存画像だけで成功としない。
- 予定step/block/branch/headに対応する実測eventが一意に存在し、`expected_edit`・normalization・実経路と一致する。8×28=224 edited callsは対象構成での補助確認であり、総数だけをcoverage証拠にしない。
- 保存方式の同一call内FP32作業値でtext massおよび各image probabilityの最大絶対誤差が `1e-5` 以下。text massが0のheadだけではなく非退化headを含む。
- text行・未選択branchの保持、scope外・編集区間外の非編集を確認。保存性は後続block/stepや最終構図の不変を意味しない。
- nativeの反復変動と、observe/off/zero originalとの差を分離する。予期しない差があれば原因を確認するまで主実験を保留する。

## 6. 80画像Pilotの固定と収集

入力: [validation_matrix_afm_pilot.json](../examples/validation_matrix_afm_pilot.json)。80件すべてを事前登録し、結果を見てpromptやseedを交換しない。prompt IDは `face_portrait`、`person_hands`、`detailed_still_life`、`spatial_objects`。seedは `20261015,20261016,20261017,20261018`。**512×512**、batch 1、Euler/simple、8 steps、CFG 1、head chunk 2、query chunk 128、workspace 512 MiBを全条件で固定する。診断はblock 0/13/27、head 0/16/32/47、全step、top-K 8、radial bins 32。exact text、全入力、実行順はmatrixと解決後promptのhashを正本とする。

顔・手のpromptは今回新しく固定した文面で、seedも過去の人物比較の再現用ではない。過去の26画像を同条件で再生成する実験とは区別する。低解像度Pilotの結果から768²など高解像度の画質・細部改善を確定せず、候補を絞った後の独立確認・性能stageへ残す。

| 条件 | 画像数 | 用途 |
|---|---:|---|
| original | 16 | native/offの見た目の基準。spectral baselineを装わない |
| preserve/manual zero | 16 | same-kernel zero、強度効果の主対照 |
| preserve/manual 1.0 | 16 | 候補1 |
| preserve/manual 1.5 | 16 | 候補2 |
| preserve/manual 2.0 | 16 | 以前の探索から得た候補3 |

baseline反復、接続smoke、性能warm-up/反復はこの80枚に含めず別枠で数える。warm-up画像や再試行を都合のよいPilot画像へ差し替えない。失敗・未提出・重複・再試行を明記し、必要な再試行では元runとの関係を残す。

予定manifestを保持し、実行結果は別の `pilot-actual/manifest.json` に収集する。1件実行manifestをまとめる際は、各source manifestのパス/hash/`invocation_id`、元run ID、prompt ID、`runtime_run_id`、固有JSONLへの対応を失わない。集約ファイルの単一invocationで実行したことにしない。予定80件に対するcompleted/failed/not-submitted、回収PNG、runtime証拠、採点完了を別々に数える。original/offの16件はJSONLと `runtime_run_id` がN/Aであり、history・元PNG・実行process来歴で確認する。JSONLを要求するのは64件のmanual条件で、originalに架空ログを作らない。

### 人による盲検画像評価

個別画像の採点テンプレートは [afm_pilot_ratings.csv](../examples/afm_pilot_ratings.csv)、対応表は [afm_pilot_rating_key.csv](../examples/afm_pilot_rating_key.csv)。これは予定IDのテンプレートであり、画像生成・採点済みの証拠ではない。このCSV単独ではpaired選好の勝敗/tieを収集・計算しない。

1. 全80元PNGを無選別で保存し、bytes SHA-256と復号後RGB pixel hashを記録する。盲検用コピーはランダム化したblind IDで表示し、ファイル名・PNG metadata・見出しに条件/強度を出さない。metadata除去でbytesが変わる場合は元とのRGB一致を確認する。
2. keyに元run ID、実行manifest、実PNGとhashを結び付け、採点者にはkeyを渡さない。表示順・左右順・ランダム化seedを固定する。cropは全条件で同一座標・倍率とし、元画像も閲覧できるようにする。
3. promptを示した上で、構図/指示への忠実さ、全体品質、顔・手などの破綻、物体の増減・重複を同一rubricで採点する。適用不能は空欄またはNAとし、0点にしない。5段階欄は1=明確に悪い、3=許容、5=良い、`artifact_severity_0_to_4` は0=なし、4=重度という定義を採点前に固定する。
4. 選好を報告する前に別の `paired-preference.csv` を準備し、同prompt/seedのblind image A/B、表示順、A/B/tie、破綻の理由、採点者ID、rubric version、時刻を収集する。既存テンプレートへ条件名を露出させず、両CSVを凍結・hash保存してからkeyを開示する。paired sheetがなければ個別得点の差だけを報告し、「何勝何敗」と推測しない。1人の評価を複数評価者の一致と呼ばない。
5. primaryは各非zero条件対manual zero、secondaryは対original。prompt内でseedのpaired差を平均し、そのprompt平均を等重み集約する。4 promptの探索なので有意差・一般的優越・最適強度を確定しない。3強度から選んだ候補は別の未使用prompt/seedで確認する。

### attention解析は別の出力

```powershell
& $Python scripts/analyze_afm_experiment.py `
  --manifest "$Out/pilot-actual/manifest.json" `
  --out-json "$Out/analysis/attention.json" `
  --out-csv "$Out/analysis/attention.csv" `
  --bootstrap-seed 20261015 --bootstrap-samples 10000
```

CLIのschema-5 coverage/paired差は画像採点を行わない。PNGが存在してもspectral PASSにはならず、original/offはattentionイベントのない画像対照として扱う。旧baseline/joint/preserveの3条件比較を埋めるためにmanual zeroをjointへ偽装しない。same-kernel zeroとの対応を明示した比較結果を使い、比較不能・complete pair不足はそのまま報告する。

`rho_text_raw`、centered、radial-binned、同一call内direct deltaとrun間trajectory差を混ぜない。終盤 `progress>=0.8` は8 stepでは原則index 6/7。head→block→stepの予定等重みを守り、undefinedを0や改善にしない。centeredがundefinedでもrawが有効な場合は別判定とする。期待集合を観測されたeventから逆算しない。bootstrap CIはprompt単位で解釈し、4×4を16独立promptとして扱わない。LPIPS等が未導入なら未測定とし、画像変化量を画質改善へ読み替えない。

## 7. 任意のfrequency-neutral対照

別入力: [validation_matrix_afm_pilot_neutral.json](../examples/validation_matrix_afm_pilot_neutral.json)。4 prompt × 4 seed × 1条件 = 16画像。既定80画像に含めない。採用を判断した時点で別matrix/hashとして固定する。盲検用の [neutral ratings](../examples/afm_pilot_neutral_ratings.csv) と [別key](../examples/afm_pilot_neutral_rating_key.csv) も主実験と分ける。

`schedule=frequency_neutral` は全progressで **alphaLF = alphaHF = 1 + strength/2**。強度2なら双方2。比較対象のcurveと同じstrength、`preserve_dc=true`、preserve/manual、LSE補正、scope・chunk・seed・promptを使用する。DCは保持され、非DC成分に一様なgainがかかる。

この対照で切り分けるのはLF→HFの時間・周波数依存scheduleと一様非DC gainの違いである。**総編集量が同じ、同じ画像変化量、単純なtemperature変更と同等、という対照ではない。** LSE補正後の最終logit DCをFFT段のDC保持と混同しない。既定scheduleはcurveのままにする。

```powershell
& $Python scripts/run_afm_matrix.py --comfy-url $ComfyUrl `
  --matrix examples/validation_matrix_afm_pilot_neutral.json `
  --out-dir "$Out/neutral-planned" `
  --unet-name intorealismAsian_k2JAVFLASHV1.safetensors `
  --clip-name qwen3vl_4b_fp8_scaled.safetensors `
  --vae-name qwen_image_vae.safetensors
```

これもdry-runのみ。後日許可されて実行する場合は、neutral対curve強度2の16組を独立の探索比較として追加し、80画像の主結果と混ぜない。両matrix完了時だけ「96画像」と数える。

## 8. 性能stage: 診断off・warm-up後に5回以上

画質・保存性gateの後に別matrixを作る。まず512²で同じpreserve/manualのzero・1・1.5・2を対象とし、モデル、prompt/seed、shape、dtype、scope、chunk、実行中の他負荷を固定する。768²と768×512は後日、各shapeの接続・保存性gateを通した別系列で測定し、512²の値と混ぜない。

- 各条件で `benchmark=true, debug_level=off, spectral_diag=off, compare_kernels=false`。matrixの `diagnostic_mode=off` も明示する。監査用JSONLのrun identity/実経路/summaryは保持する。
- 初回ロードを含むcold runを別保存し、**各条件1回以上warm-up、その後各条件5回以上**測る。順序はroundごとに正順/逆順を交互にするなど事前固定し、同条件をまとめて測らない。warm-upと測定をphase/repeat indexで区別する。
- server内で同期された `wall_seconds`、`edit_seconds`、peak allocated/reservedを採取し、`measurement_source=server`、`cuda_synchronized=true`、`diagnostics_enabled=false` を確認する。ROCmでも実際のPyTorch API動作と `torch.version.hip` を記録し、CUDAハードウェアと誤記しない。
- 現runtimeの `timing_scope=outer_sample_cuda_synchronized` はsampling区間であり、VAE decode/PNG保存を含む全生成時間ではない。全生成時間を測るなら別途server内の測定区間を定義する。HTTP待ち時間やclient側VRAMを代用しない。
- original/offはAFM wrapperを通らないため、その設定だけでは同期AFM benchmarkログを得られない。nativeの性能比較には別の検証済みserver計測が必要であり、欠損を0にしない。
- cache hitをwarm測定に含めず、各回の実sampler実行を確認する。各条件のn、中央値、範囲・ばらつき、実chunk、shape、dtype、実装経路を報告する。source/runtimeが途中で変われば新系列にする。

preserve/manualのコスト測定で高速化を保証しない。preserve+augmentedは今回対象外である。将来のaugmented profile資格は同一normalization/versionに対する数値基準と、manual比でwarm時間・peak allocated中央値が共に非悪化かつ片方5%以上改善、差が変動幅を超える証拠が必要。CPU FP32合格をBF16/FP16へ転用しない。`qualified_profiles.json` は自動更新しない。

## 9. 完了・停止・保留の判定

| Gate | 完了に必要な証拠 | 停止または保留 |
|---|---|---|
| CPU | 修正後全suite、source-contract、compile、diff、dry-runを現在sourceへ結び付ける | 失敗・必要なsource-contract skip・未検査差分 |
| runtime/接続 | 配備とロード済み版が一致、live schema照合、対象model/backendのbaseline/zero/保存性 | stale process、hash不一致、未知backend、runner transport未解決 |
| Pilot収集 | 予定80件と実際の80画像・history、manual64件のJSONL（original16件はN/A）、全失敗・再試行を追跡 | OOM/NaN/fallback、unexpected edit、保存誤差超過、missing/duplicate必須event |
| 画像評価 | 全条件の盲検採点とkey、破綻例を含む無選別の結果 | 欠損/非盲検/事後選別は探索的と明記。無理に優越判定しない |
| attention | 対応するcomplete pairと定義済みmetric、欠損理由、正しいcoverage | 未対応protocol/undefinedはそのmetricをunverified。originalに架空PASSを付けない |
| 性能 | 同期server測定、各条件warm-up＋5回、診断off、cold分離 | cache hit、未同期、他負荷、VRAM null、ばらつきが差以上 |

技術的な停止条件が出たら新規送信を止め、失敗prompt/history/ログを保存する。無限再試行、閾値の事後緩和、失敗画像の除外、別モデルへの無言変更は行わない。再開には原因と修正sourceを記録し、関連gateを取り直す。画像Pilotの完了だけでは既定値変更・auto採用・一般的画質改善を宣言しない。

## 10. 今回後回しにする互換性と提出物

別stratumとして残す: 768²/768×512および他の未測定shape、奇数latent/padding、単一frame5D（主実験は4D）、batch>1、CFG別forward、Raw/Base/Turbo別モデル、CFG negative/both、LoRA、img2img/denoise<1、追加sampler・未知sigma、FP16、他量子化・backend、DC/soft-mask/cutoff/K/bin感度、全scope/head保証。768²は初回Pilotの条件ではなく、確認実験・性能測定まで未検証とする。指定FLASHモデルのCFG=2をRawモデル資格へ流用しない。

reference連結、複数frame、競合attention/QKV/output patch、EasyCache/LazyCache、multi-GPU、torch.compileは、対応範囲の拡張を主張せず既存の拒否/対象外契約を確認する。例外後回復・clone/run分離とUI保存再ロードは接続gateの別項目として残す。

Codexの最終報告には、実行/未実行/失敗を分けたgate一覧、source/runtime/model hash、予定/実測manifest、元PNGと盲検画像の対応、採点CSVとkey、JSONL/history、attention解析、同期性能表、全失敗例、未対応構成を含める。portableな要約とhash一覧をGit管理下へ追加する場合も、過去の `docs/validation-results/2026-10-15` を上書きしない。push・公開・profile追加はこの検証計画に含まない。

参照: [従来の検証記録](validation.md)、[リファクタリング計画 §7–10](superpowers/plans/2026-10-15-krea2-afm-refactoring.md)、[旧実装計画 §6.2–6.3](../Krea2_AFM_Codex_Implementation_Plan_2026-09-30.md)。本書の80画像Pilotは直近の高強度探索を受けた限定計画であり、旧計画の128画像Pilot/240画像確認実験を実行済みとは扱わない。
