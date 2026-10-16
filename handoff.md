# Krea2 AFM 実装の引き継ぎ

**最新: 2026-10-16 — 検証・実装方針の切り替え**

今後は[効果確認・層／編集区間比較の計画](docs/superpowers/plans/2026-10-16-afm-scope-window-validation.md)を正本とします。ユーザー指定の `Krea2_AFM_Validation_Implementation_Plan_2026-10-16.md` を内容変更なしでリポジトリ内へ格納しました。以下の旧引き継ぎ・実績は履歴として残します。

- 適用時に確認したHEADは `3c1adb2994f957082afcad3d57377f0e007da8e1`、branchは `feat/krea2-afm-edit`。更新前の作業ツリーはcleanで、計画書の基準と一致します。
- 方針は現行80画像Pilot、明示選択のneutral対照16画像、層×編集区間の200画像探索、独立確認の順です。変更量制限を追加せず、既存AFM演算・CFG=1・全8ステップ・元のsigma列とcurveを維持します。各editの主対照は同じscope／windowのmanual zeroです。
- 今回実施したのは計画書格納とREADME・旧実機計画・本引き継ぎの参照更新です。T0–T6は未着手で、新CLI・matrix・盲検評価・比較集計の追加は未実施です。チェックボックスは更新していません。
- 次の実装はT0から開始し、HEAD／dirty diff／Python・torchの再取得、分離環境の準備、既存CPU suite・compile、scope／window契約の回帰化を行います。その後T1–T6へ進め、CPU／mock・送信なしdry-runまでを検証します。
- 今回はCPUテストを再実行していません。GPU実験、モデル取得、サーバー再起動、既定値・profile更新、commit・pushは行っていません。GPU段階の配備・来歴・接続gateは[従来の実機検証手順](docs/hardware-validation-plan-2026-10-15.md)を継承します。

**2026-10-15時点の引き継ぎ** — リファクタリング実装とテストコードを追加し、ユーザー指定により検証は未実施です。[今回の実装引き継ぎ](docs/refactoring-implementation-2026-10-15.md)を先に参照してください。以下のPASS・レビュー記録は2026-09-30の旧実装に対する履歴です。


更新日: 2026-09-30。再開作業を完了。Task 0–4とTask 5のオフラインツールは実装済みです。
最新の検証記録は `docs/implementation-status.md` と `docs/validation.md` を参照。
停止前の引き継ぎ原文はローカル `outputs/handoff-before-resume.md` に保存しています。

## 制約

- 実モデル検証は今回の対象外。モデル取得、ComfyUIへの生成投入、GPU生成・性能測定は行っていません。
- CPUテストの成功を実機/GPU/画質の保証としない。
- 作業は `feat/krea2-afm-edit` にローカルで保持。外部push・PR・merge・公開は不要。
- 作業ツリーや `.reference/`、`.superpowers/`、`outputs/` を初期化しない。

## 今回完了した残作業

1. `scripts/run_afm_matrix.py` にrunner呼び出しごとのUUID `invocation_id` を追加。
   prompt、manifestの `server_log_relative`、mapped logの参照先を
   `krea2_afm/<条件ID>-<invocation_id>.jsonl` に統一しました。
   `node_inputs` ではこのログ名を上書きできず、各条件は自分のログだけを参照します。
   今回のログがなければ `server_metrics=unverified`。過去ログでは代用しません。
   `--server-log-dir` はサーバーの `output/krea2_afm` をマウント/同期したディレクトリです。
2. `scripts/summarize_krea2_afm.py` でdtype aliasを正規化。
   単一profileのreportはactivation dtypeを統一し、そのdtypeでstrength `.05/.1/.2`
   を揃える必要があります。FP16/BF16混在で証拠を合算できません。
3. 計画書・README・validation・status・パッケージ説明を現在の編集実装へ整合。

停止時に追加されていた回帰テストをRED→GREENで完了し、複数条件のログ対応と
FP16/BF16 aliasの回帰テストも追加しました。
独立した最終レビューでは今回のrunner/dtype修正に未解決の指摘はありません。

## 最終検証

- Python 3.10.11 / PyTorch 2.10.0+cpu / pytest 9.0.2。
- 上流ソース契約を含むCPU suite: **241 PASS、GPU1件deselected**。
- runner suite 11 PASS、report suite 15 PASS。
- `compileall` / `git diff --check` / 編集API graph validation成功。
- 7条件dry-run: すべて `not_submitted` / `unverified`。7つの固有ログ名と
  6つのAFM promptのログ入力がmanifestに一致（残り1つはtrue baseline）。
- 専用lint/typecheckツールは未導入。実 `/object_info`・ブラウザー読込・推論は未検証。

安全な再検証コマンド:

```powershell
$env:KREA2_COMFY_SOURCE_ROOT = (Resolve-Path '.reference/upstream-9d80841').Path
python -m pytest tests -q
python -m compileall -q afm nodes.py __init__.py scripts
git diff --check
python scripts/validate_workflow.py examples/krea2_afm_edit_api.json
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 --matrix examples/validation_matrix.json --out-dir outputs/krea2_afm_dry_run
```

`--execute` や `KREA2_AFM_RUN_REAL_MODEL=1` を追加しないこと。
`.reference/upstream-9d80841/` はモデルではなく、固定revisionのソースのみです。

## 残課題

- `afm/backend.py` の不正branch/maskの理由を1callで2回数える場合があります。
- `profile_not_qualified` はattention metadataに記録されますが、`RunState.reasons` の
  集計には入りません。編集結果・edited_calls・fallback_callsには影響しません。
- これら2件は以前のkernel/runtimeレビューからのMinorとして保留。
- 実機のTurbo/Raw、CFG、量子化/backend、LoRA、img2img、数値/速度/VRAM/画質評価は未実施。
- `afm/qualified_profiles.json` は `[]`。autoは未検証ならmanual。

次の作業でGPU/実モデル確認を行う場合は、今回の除外条件を変更する明示指示が必要です。
この計画の実機項目は未完了のため、ローカルのSDD作業記録は保持しています。
