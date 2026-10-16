# Krea2 AFM (experimental)

**2026-10-16 今後の検証・実装方針:** [効果確認・層／編集区間比較の計画](docs/superpowers/plans/2026-10-16-afm-scope-window-validation.md)を今後の方針の正本とします。既存80画像Pilot（neutral対照16画像は明示選択）で効果を確認し、層×編集区間の200画像探索、独立確認へ段階的に進めます。AFM演算とCFG=1・全8ステップを維持し、変更量制限は追加しません。

今回の更新は計画書の格納と参照先の切り替えです。新しいmatrix生成・盲検評価・比較集計ツールは未実装で、CPU検証の再実行・GPU実験は未実施です。次の実装は新計画のT0から開始します。配備・来歴・CPU／接続gate・停止条件は[従来の実機検証手順](docs/hardware-validation-plan-2026-10-15.md)を継承します。

**2026-10-15 修正後:** CPU回帰は470成功・GPU 1件除外。入力契約と検証ツールを修正しました。過去のGPU記録は修正前の証拠であり、この差分のGPU再検証は未実施です。[現在の実装・CPU検証](docs/implementation-status-2026-10-15.md)と[従来の実機検証手順](docs/hardware-validation-plan-2026-10-15.md)を参照してください。

**2026-10-15 修正前の検証記録:** 起動中ComfyUIと指定INT8モデルで、22生成条件・7拒否試験と人物26枚の強度比較を確認しました。回帰テストは342成功・19失敗のため、全体合格には未達です。確認範囲・失敗原因・証拠は[検証結果](docs/validation.md)を参照してください。

ComfyUIのネイティブKrea2向けに、画像Query → テキストKeyのAttention logitsを
空間周波数で編集するカスタムノードです。画像・テキストを合わせた全Keyで正規化します。
Anima AFM / Krea2T Enhancerのインストールは不要です。

**編集機能を実装済みです。指定INT8モデルで生成・編集適用・総注意量保存を確認しました。GPU性能・一般的な画質改善は未検証です。**
CPUの小規模Tensorで、直接計算・分割manual・Q/K拡張augmentedの数値比較と
CFG行保持・fallbackを検証しています。CPUでの一致は実モデル/GPUの保証ではありません。

## インストール

このフォルダーをComfyUIの `custom_nodes/AFM-for-Krea2-experimental` に配置し、
ComfyUIを再起動します。既存環境のPyTorchを使用します。追加ライブラリは不要です。

ソース確認基準はComfyUI `9d80841aa1990305cc7a280c5ea505f317efcc2d`。
V3ノード、OUTER_SAMPLE / DIFFUSION_MODEL wrapper、post_input、およびnative
`SingleStreamDiT` を機能検出します。実機確認済みの最低版という意味ではありません。

## 使い方

モデルローダーのMODELとKSamplerのMODELの間に **Krea2 AFM Model Patch** を接続します。
通常LoRAがある場合は、そのMODEL出力の後へ接続します。

| モード | 動作 |
| --- | --- |
| `off` | cloneのみ返し、AFMのhookを追加しない |
| `discover` | txtfusion後の全本体blockのlayout/backendを記録。出力を変更しない |
| `observe` | 指定blockを観測。出力を変更しない |
| `edit` | 選択したCFG branchの画像行をAFMで置換する |

ノード既定は `observe`。編集を試す設定例は `mode=edit`, `implementation=manual`,
`strength=0.05`, `block_scope=0`, `branch_mode=positive_only` です。
最適設定や画質改善を確認した値ではありません。

人物2構図・各1 seedの目視評価では、ユーザーは `preserve_text_mass / manual / strength=2.0 / block_scope=all` を好印象と評価しました。従来のjoint方式は構図が変わり、強度2.0では物体重複などの破綻が見られました。これは限定条件の探索結果で、既定値の変更や一般的改善の確認ではありません。[人物比較画像と検証記録](docs/validation.md#人物比較の観察と現時点の結論)を参照してください。

| 設定 | 意味 |
| --- | --- |
| `normalization_mode=joint` | 従来方式。テキストlogitsをFFT編集して全Key softmax |
| `normalization_mode=preserve_text_mass` | 同一callのtext総注意量を保つLSE補正。manualのみ、autoもmanual |
| `diagnostic_head_indices / diagnostic_radial_bins` | 物理headの指定とradial bin数。空欄headは従来の先頭選択 |
| `implementation=manual` | 全Key正規化を保つFP32 logits/softmaxの分割基準実装 |
| `implementation=augmented` | Q/K/V拡張と分割SDPAを使う実験経路。元のAttention scaleを維持 |
| `implementation=auto` | 未検証環境ではmanual。出荷時の採用リストは空 |
| `strength` | 0なら既定では元経路。正負の値で周波数gainを変更 |
| `schedule=curve` | 進行度uに対してLF=1+s(1-u)、HF=1+su |
| `cutoff / soft_mask / mask_width` | 正規化半径周波数の境界とcosine遷移幅 |
| `preserve_dc` | 空間平均に対応するDC成分を維持 |
| `block_scope` | `0-3`, `0,2`, `all` など。本体の実際のblock index |
| `start_percent / end_percent` | 実際に渡されたsigma schedule上の編集区間 |
| `head_chunk_size / query_chunk_size` | 分割上限。既定2 / 128。FFTは常に画像全grid |
| `max_workspace_mib` | 追加・延長生存Tensorの保守的な見積もり予算。既定512 MiB |
| `fail_mode` | `raise` は停止、`fallback_original` はその呼び出しの元出力へ戻す |
| `augmented_precision` | SDPAにactivation dtypeを使うか比較用FP32を使うか |

予算不足ならhead/query chunkを縮小し、それでも不足なら `memory_guard`。
見積もりは総VRAM上限ではありません。baseline自体が動く構成が前提です。
途中のOOM・非有限値・例外では部分編集を返さず、別kernelへ自動再試行もしません。

`strength=0, zero_strength_mode=original` はAFM hookを追加しません。
`zero_strength_mode=manual / augmented` は指定方式の数値対照を強制し、
`implementation` より優先します。元backendとbitwise同一のbaselineではありません。

## ログと比較診断

既定のログ先はComfyUIの `output/krea2_afm/diagnostics.jsonl`。
ノード入力には `krea2_afm/任意名.jsonl` を指定します。空欄でファイル出力を無効化。
絶対パス、`..`、専用フォルダー外へ解決されるパスは拒否します。

各runには固有の `run_id` が付き、`run_start`, `attention`, `run_summary` を記録します。
未計測値はnull。`edited_calls=0` や全fallbackを編集成功と扱いません。
テキスト行・未選択branchは同一callのbaselineをそのまま保持しますが、編集が後続blockへ
伝わるため、最終的なnegative予測全体が不変になるという意味ではありません。

通常は `spectral_diag=off`, `compare_kernels=false`。必要時だけ有効化します。

- `spectral_diag=sampled`: 指定step/block/headで、テキスト内softmaxのtop-K集中度の
  HF比をraw/centered/radial-binnedの別名で記録し、joint softmaxの `text_attention_mass` と保存誤差を別に記録。schema 5の未定義指標はnull。
- `compare_kernels=true`: 同じQKVでmanual FP32とaugmented、および各zero対照を比較。
  `effect_resolved=false` は効果を数値ノイズから分離できないことを示します。
- `diagnostic_block_scope` は編集scopeとは独立。head数・query chunk・step間隔を制限できます。
- `benchmark=true`: 将来のGPU検証時に、サーバープロセス内でCUDA同期・peak allocated/reserved・
  生成時間と編集時間を測定。比較診断とは同時に有効化できません。CPUではCUDA値を記録しません。

## ワークフローと検証ツール

- [編集用UI workflow](examples/krea2_afm_edit_ui.json): ブラウザーへロードする形式。
- API prompt JSON: [baseline](examples/krea2_afm_baseline_api.json) /
  [discover](examples/krea2_afm_discover_api.json) / [observe](examples/krea2_afm_observe_api.json) /
  [edit](examples/krea2_afm_edit_api.json)。API JSONはUI形式ではありません。

公式Turbo構成を基に、512×512、固定seed、8 steps、CFG=1、Euler/simpleで用意しました。
モデル名を実際のファイル名に合わせてください。指定モデルに置換したAPIの実サーバー生成を確認しました。ブラウザーでのUIロード・再保存は未検証です。

```console
python scripts/validate_workflow.py examples/krea2_afm_edit_api.json
python scripts/validate_workflow.py examples/krea2_afm_edit_api.json --object-info object_info.json
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 --matrix examples/validation_matrix.json --out-dir outputs/krea2_afm
```

最後のコマンドも既定はdry-runで、prompt/manifestを保存するだけです。
生成するには `--execute` を追加します。今回の環境では付属runnerのHTTP取得が失敗し、別transportで同じpromptを実行しました。詳細は検証結果を参照してください。

runnerは呼び出しごとの `invocation_id` をmanifestへ記録し、AFMログを
`krea2_afm/<条件ID>-<invocation_id>.jsonl` に保存するようpromptへ設定します。
`--server-log-dir` にはサーバーの `output/krea2_afm` をクライアントへ
マウントまたは同期したディレクトリを指定します。今回のログがなければ
`unverified` のままにし、過去の同名条件のログを証拠として使いません。

## テストと対応範囲

```console
python -m pytest tests -q
python -m compileall -q afm nodes.py __init__.py scripts
```

4D latent・単一フレーム5D・長方形を扱います。reference連結、複数フレーム、他Attention
 override/post_input/QKV/outputパッチは拒否します。複数GPUとtorch.compileは対象外です。
今回確認したのは指定INT8モデルとBF16 attentionの限定構成です。他の量子化・Turbo/Rawモデル・LoRA・img2imgの実機動作は未確認です。

詳しい手順は [検証文書](docs/validation.md)、実装記録は
[実装状況](docs/implementation-status.md)、出典とライセンスは [NOTICE](NOTICE.md) を参照してください。
