> 後続検証: 起動中ComfyUIと指定モデルで実機試験を実施しました。初期22条件・人物26枚を確認し、ユーザーは保存方式の強度2.0を好印象と評価しています。回帰342成功・19失敗で全体未合格です。[検証結果](validation.md)を参照。以下は検証前の実装引き継ぎ記録です。

# 2026-10-15 リファクタリング実装・検証引き継ぎ

計画: [リファクタリング・追加検証計画書](superpowers/plans/2026-10-15-krea2-afm-refactoring.md)

## 今回の状態

ユーザー指定により実装とテストコードの整備まで行い、**検証は実行していない**。
テスト・構文チェック・lint・型検査・workflow dry-run・実サーバー/UI・GPU推論・性能測定・独立コードレビューはいずれも未実施。
従来の241件PASSは旧HEAD `944e507` に対する履歴で、今回の差分の証拠ではない。

計画T0–T5の実装作業を対象とする。検証を要求するチェック項目は未完了のまま残し、T6–T8は別の検証・効果評価・高速化段階とする。

## 実装内容

- 共用の入力/finite/option検査を `afm/attention_math.py` へ移動。
- `normalization_mode=preserve_text_mass` のmanual kernelを追加。既定はjoint。
- scale適用済みtext logitsの元logsumexpを保存し、FFT後にLSE差を加算する。
- preserve modeのaugmented/augmented zero control/kernel比較を事前拒否。autoはmanualを選択。
- 新入力normalization_mode、diagnostic_head_indices、diagnostic_radial_binsは末尾のoptional widget。旧APIで省略可能。
- JSONL schema 5。raw/centered/radial-binnedのHF比を区別し、metric別の未定義をnullで表す。
- 物理headごとにbounded診断。同一callのtext mass/各image probabilityの最大変化を記録。
- 正規化mode/versionをfilter signatureへ含め、旧profileや異なる方式の証拠を分離。
- 既知EasyCache/LazyCacheの省略wrapperはactive AFMとの併用を拒否。
- edit/observe共通経路で観測回数を数えるよう整理し、unsupported理由の二重計数と選択理由のsummary欠落に対する回帰ケースを追加。
- 実験runner/report reader/paired analysisをschema 5・manifest 2へ拡張。
- 独立FP64 oracle、保存性・legacy workflow・selection・診断・cache・集計の回帰テストを追加。**テストは未実行**。

## 利用上の契約

総注意量保存は同一QKVの同じattention call内の性質であり、後続block・次step・生成画像の不変を意味しない。
preserve_dcはFFT段のDC保持で、LSE補正後の最終logitのDC保持とは異なる。
保存性の診断基準は同一QKVから得たFP32作業logit。native backendの丸め・量子化との差はzero controlで別途確認する必要がある。

`implementation=augmented` を新方式に使用する対応は初版に含まれない。
`compare_kernels=true` もjoint方式だけのmanual/augmented比較である。

`diagnostic_top_k` の旧既定4は維持し、論文比較用workflowは8を明示する。
`diagnostic_head_indices` は例 `0,16,32,47`。空欄は先頭headをlimitで選ぶ。
`diagnostic_radial_bins` は32が既定で、計画の16/32/64感度比較に使用できる。

## 別途実行する確認

プロジェクトrootを作業ディレクトリとして、まず依存追加なしのCPU側確認から始める。

~~~powershell
$env:KREA2_COMFY_SOURCE_ROOT = (Resolve-Path '../ComfyUI').Path
python -m pytest tests -q
python -m compileall -q afm nodes.py __init__.py scripts
git diff --check
python scripts/validate_workflow.py examples/krea2_afm_edit_api.json
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 --matrix examples/validation_matrix_afm_refactor.json --out-dir outputs/krea2_afm_refactor/dry-run
~~~

この一覧は**未実行の引き継ぎ手順**。dry-runでも今回は実行していない。
実モデル生成へ進む前にobject_info/旧workflow/UI再保存を確認する。
GPUでは画像生成成功と、予定step/block/CFG/headへの編集適用を別々に確認する。
効果評価はtrue baseline/observeの同等性確認後に、計画の探索・Pilot・独立確認実験へ進む。

保存済みmanifestのオフライン集計例（未実行）:

```powershell
python scripts/analyze_afm_experiment.py --manifest outputs/krea2_afm_refactor/manifest.json --out-json outputs/krea2_afm_refactor/analysis.json --out-csv outputs/krea2_afm_refactor/analysis.csv
```

集計はmodel/sampling/resolution/metric定義/観測座標/比較設定でstratumを分け、3条件共通のcomplete pairだけを使います。baselineのrepeat 1以降はprimary比較に混ぜず、変動の記述統計として残します。

実装取り込み完了: T0–T5の製品コード・テストコード・workflow/matrix・集計ツールを整備しました。初回の自動coverage生成はbatch=1/positive_only/Euler/simpleを対象とし、negative_only/both等の未確立条件はunverifiedです。benchmarkの値は設定値autoではなく実選択経路へ帰属させます。いずれも未検証です。
