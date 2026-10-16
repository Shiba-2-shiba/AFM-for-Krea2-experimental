# Krea2 AFM：効果確認・層／編集区間比較 Implementation Plan

> **For agentic workers:** 実装時は `superpowers:executing-plans` または `superpowers:subagent-driven-development` を用い、タスクごとに回帰テストとレビューを行う。チェックボックスは実施・確認後にのみ更新する。本書の作成は、製品コード変更、GPU実験、commit、pushを実行したことを意味しない。

**Goal:** Krea2のCFG=1・8ステップ推論を維持し、現行AFMの効果確認と、適用層・編集区間の比較を再現可能にする。変更量制限は実装しない。

**Architecture:** 既存のAFM演算、`block_scope`、`start_percent`／`end_percent`、manual zero、実験runnerを再利用する。追加の中心は、実験matrix生成、実行来歴の引き継ぎ、条件間の明示的な比較、盲検画像評価である。推論経路は変更せず、既存解析の厳格な照合を維持したまま、新しい比較経路を追加する。

**Tech Stack:** Python 3.10以上、標準ライブラリ、既存PyTorch／pytest、ComfyUI。画像の盲検コピーと画素hashには、評価環境に既存のPillowを使用する。新しいモデル、評価モデル、学習、実行時依存の自動追加は不要。

**Spec:** 本書§2–§8を自己完結した実装仕様とする。直前の合意「第一段階の効果確認、第二段階の層・生成段階選別、変更量制限なし」を具体化したもの。文献の新規実装や状態依存制御の実装計画ではない。

**作成日:** 2026-10-16（JST）  
**対象:** `Shiba-2-shiba/AFM-for-Krea2-experimental`  
**確認した基準commit:** `3c1adb2994f957082afcad3d57377f0e007da8e1`  
**現在の基準branch:** `feat/krea2-afm-edit`  
**実装用branch案:** `feat/afm-scope-window-validation`  
**リポジトリへ格納する場合の推奨先:** `docs/superpowers/plans/2026-10-16-afm-scope-window-validation.md`  
**本書の状態:** 計画書。実装、CPUテスト再実行、GPU検証、リポジトリ更新はいずれも本書作成作業の対象外。

## Global Constraints

- **変更量制限を追加しない。** 出力差分／logit差分のclip、変更量を理由にしたstrengthの自動減衰、norm上限、許容編集量の探索、変更量が大きいという理由だけのfallbackを導入しない。
- ユーザーが指定した非zeroの有限strengthを、そのままmatrixへ記録する。探索候補の1.0／1.5／2.0は現行Pilot由来であり、strengthの製品上限ではない。
- `preserve_text_mass`、DC保持、FP32 FFT、joint softmax、`curve`、`frequency_neutral`の演算を変更しない。
- 層・区間の違いに応じてstrengthを自動補正しない。適用回数が減っても、同じ選択strengthを使う。
- CFG=1、Euler／simple、8 steps、denoise=1、batch=1を初回protocolとして固定する。条件なし分岐、CoG、追加モデル推論を導入しない。
- **全8ステップを実行する。** 編集区間を絞ってもsigma列は切り詰めず、区間内の進行度を0–1へ再正規化しない。
- AFMノードの新widget、層別strength、head別編集、複数の非連続編集区間、状態依存gain、RoPE変更は今回対象外。
- numerical finite検査、入力契約、workspace管理、例外時の既存fallback、数値実装間の誤差検証は維持する。これらは不要とされた「画像への変更量制限」と別物である。
- 古いAPI/UI workflow、JSONL schema 5、manifest schema 2、既存CLIの既定動作を維持する。追加manifest項目は任意の名前空間付きメタデータとする。
- `qualified_profiles.json`、既定strength、既定normalization、既定block_scopeを自動変更しない。
- runnerはdry-runが既定。計画書の受領やCPU実装完了を、GPU実行・サーバー再起動・モデル取得・pushの許可に読み替えない。
- 比較用の旧画像、旧検証記録、既存80画像／16画像matrixを上書きしない。実行していない計画を実績と記載しない。

## Review Focus

1. **局所zeroの取り違え:** 前半／後半や層限定のAFMに、全層・全区間のzeroを唯一の対照として割り当てない。T1／T4で固定する。
2. **区間の意味の変更:** 8ステップを維持し、index 0–3／4–7が重複も欠落もなく選ばれる。後半でcurveを再開始しない。T0／T1で固定する。
3. **「比較を可能にする」ための来歴チェック緩和:** 変更してよい因子だけを比較仕様で宣言し、モデル、prompt、seed、精度、backend等の不一致は拒否する。T2／T4で固定する。
4. **観測点と適用点の混同:** 診断scope外の未観測を編集なしと断定しない。scope外のstepでも、指定した観測点に後続影響が現れ得る。T0／T4で固定する。
5. **盲検・欠損・再試行による選別:** metadata漏洩、画像取り違え、同じrunの二重採用、採点欠損、失敗画像の除外を検出し、勝率を都合よく算出しない。T3／T4／T5で固定する。

---

## 1. 現状と、本計画で変える範囲

### 1.1 確認できた現状

基準commitのREADMEには、修正後CPU回帰470成功・GPU 1件除外、修正差分のGPU再検証は未実施と記載されている。本書作成時にそのテストを再実行したわけではない。[R1]

現行実機計画は、4 prompt × 4 seed × 5条件＝80画像のPilotと、別matrixの任意frequency-neutral対照16画像を定義する。前者はoriginal、preserve/manual zero、preserveの強度1.0／1.5／2.0。後者はcurve強度2.0と比較するneutral強度2.0である。[R2]

`block_scope`と編集区間の指定は既存機能である。runtimeの進行度は、渡されたsigma scheduleでのindexを`num_steps - 1`で割った値であり、sigma値そのものを0–1に線形変換した値ではない。[R1][R3]

現行解析はstrength、schedule、scope、編集区間等で比較グループを分け、局所zeroとの比較では共通設定・実経路を厳格に照合する。また、主なattention集約は終盤`progress >= 0.8`を対象とする。このため、新しい層×区間の比較を単に同一ラベルでまとめても、その比較を正しく実現したことにはならない。[R4]

### 1.2 差分一覧

| 領域 | 現行 | 本計画 |
|---|---|---|
| AFM数式・正規化 | 実装済み | 変更しない |
| 層・編集区間指定 | 実装済み | 既存入力をmatrixから使用 |
| 第一段階 | 80画像＋任意16画像の計画 | 同じ条件で評価を完遂できる補助を追加 |
| 第二段階 | 専用の層×区間matrixなし | 粗い層グループ×全体／前半／後半のmatrixを追加 |
| zero対照 | 主に全層・全区間 | 各scope／windowに同一のmanual zeroを用意 |
| 条件間解析 | 正規化方式中心の固定contrast | 実験仕様で宣言したscope／window／schedule比較を別CLIで追加 |
| attention集計 | 終盤指標中心 | step別・block別の記述と固定区間集約を追加 |
| 画像評価 | CSVテンプレートと手順 | 盲検画像・対比較CSV・集計を補助するツールを追加 |
| 変更量制限 | 今回の提案から除外 | 一切追加しない |
| 状態依存制御 | 未実装 | 今回は実装しない。別計画が必要 |

### 1.3 既存文書との優先関係

本書は新しい実験方針の正本とする。`docs/hardware-validation-plan-2026-10-15.md`の配備、来歴、CPU・接続gate、停止条件は継承する。旧文書の冒頭に「後続の層・区間比較は本書を参照」と追記するが、80画像／16画像の定義と過去の結果は残す。

旧計画の128画像／240画像等を、本書の実験数へ黙って合算しない。first-stage、ablation-screening、confirmation、benchmark、smoke、retryは別phaseとして記録する。

---

## 2. 実験で答える問い

### Q1：現行AFMに、目視で採用したい品質改善があるか

preserveの各strengthを、同一カーネルのzeroと比較する。originalとの比較も別に行う。画像品質、指示への忠実さ、破綻を分け、周波数指標の変化だけで改善と判定しない。

### Q2：curveとfrequency-neutralで違いがあるか

同じprompt／seed、strength=2.0、scope、編集区間で比較する。これは周波数・時間依存scheduleと一様な非DC gainの比較であり、編集量を等しくする実験ではない。norm合わせ、temperature合わせ、画像変化量合わせは追加しない。[R2][R5]

### Q3：同じstrengthで、適用する層・区間を選ぶ意義があるか

同じstrength・同じcurveのまま、対象層と編集区間を変える。「どの局所設定でも必ず全層適用を上回る」とは想定しない。画像品質が同程度で処理時間が減る可能性、品質は上がるが構図が変わる可能性も別々に報告する。

### Q4：探索で選んだ条件が、未使用prompt／seedでも有効か

第一・第二段階は探索であり、候補を選んだデータだけで優越を確定しない。候補を固定した後、独立した確認セットで全層・全区間との対比較を行う。

---

## 3. 段階別の実験仕様

### 3.1 共通条件

初回は512×512、batch=1、Euler／simple、8 steps、CFG=1、denoise=1。実機に存在するモデル・TE・VAEを明示的に選び、ファイルhashを保存する。過去の機器・ポート・モデルファイルの所在を推定して実行しない。

編集条件は`normalization_mode=preserve_text_mass`、`implementation=manual`、`branch_mode=positive_only`、`preserve_dc=true`、`cutoff=0.25`、`soft_mask=true`、`mask_width=0.05`、head chunk=2、query chunk=128、workspace=512 MiB、`fail_mode=raise`。実機のメモリ都合で変更が必要なら、同一比較の全条件へ適用した新revisionとして固定する。

LoRA、別attentionパッチ、cache系の追加は行わない。現行が拒否する機能を、検証を進めるために有効化しない。観測用head指定は「編集するheadの選択」ではない。

### 3.2 Stage 1：既存80画像Pilot

既存の`examples/validation_matrix_afm_pilot.json`と`examples/krea2_afm_experiment_api.json`を使用する。

| 条件 | 画像数 |
|---|---:|
| original/off | 16 |
| preserve/manual zero | 16 |
| preserve/manual strength=1.0 | 16 |
| preserve/manual strength=1.5 | 16 |
| preserve/manual strength=2.0 | 16 |
| **合計** | **80** |

既存prompt IDは`face_portrait`、`person_hands`、`detailed_still_life`、`spatial_objects`。seedは20261015、20261016、20261017、20261018。prompt全文はmatrixを正本とし、計画書から再入力しない。[R2]

第一段階の追加実装は、画像と来歴の対応付け、盲検対比較の作成・採点集計である。新しいAFM機能は必要ない。

### 3.3 Stage 1N：frequency-neutral対照

`examples/validation_matrix_afm_pilot_neutral.json`の16画像を使用する。既存80画像は変更しない。実行計画の`include_neutral`を明示的にtrue／falseで固定し、falseならQ2は未評価と報告する。本計画の目的上はtrueを推奨するが、runnerへ自動的に追加送信しない。

本版の対比較は`neutral strength=2.0`対`curve strength=2.0`に限定する。neutral対manual zeroの新規対比較は今回追加しない。両matrixはphaseが異なる場合があるため、共通の研究IDと明示contrastで結び付ける。phaseを偽装して同じ実行に見せない。

Stage 2の初回scheduleはcurveのまま固定する。neutralが良い場合、その結果は別候補として残すが、Stage 2でscheduleまで同時に変えない。

### 3.4 Stage 1からStage 2への受け渡し

`stage1_selection.json`を人が確認して保存する。最低限の項目は以下。

```json
{
  "schema_version": 1,
  "status": "selected_for_exploration",
  "selected_strength": 2.0,
  "schedule": "curve",
  "evidence_manifests": [],
  "ratings_sha256": null,
  "decision_reason": "実際の採点を確認して記入する",
  "decision_maker": "実際の確認者",
  "independent_confirmation_done": false
}
```

上記は形式説明であり、実際の選定結果ではない。実行用の選定記録では空の証拠一覧、nullの採点hash、説明用文字列を拒否する。製品コードや既定値を変更する承認にはしない。

選択strengthは探索で固定する値であり、最適値の宣言ではない。Stage 1で改善が乏しくても、技術gateを満たしていれば「全層では悪化するが局所適用では改善するか」を調べる目的でStage 2へ進める。その場合は理由を明記する。コードに画像品質ベースの自動停止や自動減衰を組み込まない。

### 3.5 Stage 2：層×編集区間の粗い比較

#### 対象層

初回protocolは本体28 blocksを前提にする。実機のdiscoverで28であることを確認し、不一致ならこのprotocolでは実行しない。層数を見ずに28と推定しない。

| scope ID | `block_scope` | block数 |
|---|---|---:|
| `all` | `all` | 28 |
| `g0` | `0-8` | 9 |
| `g1` | `9-17` | 9 |
| `g2` | `18-27` | 10 |

これはモデル内の前・中・後の位置による分割であり、構図層／顔層／細部層といった機能名は付けない。9／9／10の違いを明記し、10層の群だけstrengthを補正しない。

#### 編集区間

| window ID | start | end | 8-step時の編集index |
|---|---:|---:|---|
| `full` | 0.0 | 1.0 | 0–7 |
| `early` | 0.0 | 0.5 | 0–3 |
| `late` | 0.5 | 1.0 | 4–7 |

現行は両端を含む比較だが、8 stepsでは`u=i/7`に0.5が存在しないのでearly／lateは重複しない。9 steps等にはこのまま一般化せず、初回builderは8 steps以外を拒否する。

**windowを変えても、`u=i/7`、sigma列、LF/HF gainは元のまま。** 例えばstrength=2.0、step index=4では、lateでも`alpha_lf=1+2×3/7`、`alpha_hf=1+2×4/7`を使う。late開始時にcurveを最初から始めない。

#### 比較条件と画像数

4 scope × 3 window＝12セル。各セルに同じscope／windowの2条件を用意する。

- `edit`：Stage 1から固定した非zero strength。
- `zero`：strength=0、`mode=edit`、`zero_strength_mode=manual`。他の意味上の設定はeditと同一。

さらに各prompt／seedに1つのoriginal/offを追加する。

初回screeningは既存4 promptとseed 20261015／20261016の2つに限定する。これはStage 1と重なる探索セットであり、独立確認セットではない。

**4 prompt × 2 seed × (12セル×2条件＋original 1条件)＝200画像。**

| 群 | 画像数 |
|---|---:|
| 層・区間ごとのedit | 96 |
| 各editに対応するmanual zero | 96 |
| original/off | 8 |
| **合計** | **200** |

Stage 1画像の自動再利用は初版では行わない。Stage 2を単独で追跡可能な一式とし、全層・全区間とoriginalも取り直す。再利用機能は別の最適化課題とする。

各セルの期待カーネル適用call数は`block数 × 対象step数`であり、all/fullは224、g0/earlyは36、g2/lateは40。manual zeroもカーネルを通るため同数になる。`edited_calls>0`を非zeroの画像効果と混同しない。

#### 診断の範囲

screeningの全manual条件で、既存と同じ診断block `0,13,27`、head `0,16,32,47`、全8 steps、top-K=8、radial bins=32を固定する。実機のhead数を確認し、範囲外なら実行しない。

これにより比較する観測座標は全セル共通になる。ただし、全28 blocks・全headの保存性を保証したとは報告しない。

接続smokeでは別系列として、診断block `0-27`、head `0`で全blockへの予定編集／非編集を調べる。screeningの画像や主attention統計と混ぜない。

### 3.6 Stage 3：選択条件の独立確認

Stage 2完了後に候補セルを1つ固定し、新規8 prompt × 4 seed＝32ケースで確認する。最低限の条件は、選択edit、対応zero、all/full edit、all/full zero、originalの5つ、計160画像とする。

候補がall/fullそのものなら重複条件を生成せず3条件・96画像とし、縮約理由を記録する。「選択設定が全層を上回った」という比較は成立しないため、AFM対zero／originalの確認に目的を変更する。

独立セットのprompt案は§11に固定する。Stage 3用matrixは、固定後の候補セルとこのcase一覧から5条件を展開して別途準備する。今回の自動builderの必須対象はStage 1のspec作成とStage 2のmatrixであり、Stage 3の自動生成CLIは追加しない。使用前に全文hashでStage 1／2との重複を検査し、すでに探索に使った場合は独立確認とは呼ばない。

初回は512×512で比較し、768×768等の実用解像度は、その解像度の接続gateを通した別stageとする。512の結果をそのまま高解像度での改善としない。

### 3.7 実験数と実行範囲

| 段階 | 計画画像数 | 実行開始条件 |
|---|---:|---|
| Stage 1 | 80 | CPU・実機接続gateと実行許可 |
| Stage 1N | 16 | neutral実施を明示選択 |
| Stage 2 | 200 | strength選定記録、28-block対応確認、別途実行許可 |
| Stage 3 | 160、候補all/fullなら96 | 候補セル固定、独立セット固定、別途実行許可 |

通常構成で全段階を実施する場合は456画像、neutralを省けば440画像。smoke、baseline反復、性能warm-up／反復、再試行はこの数に含めない。一括送信を既定にせず、段階ごとに止める。

---

## 4. 比較・採点の仕様

### 4.1 比較は三種類に分ける

1. **局所のAFM効果:** 各セルのedit対同じセルのmanual zero。カーネル経路と適用範囲を揃えた主対照。
2. **実用上の設定比較:** 各セルのedit対all/full edit。同じstrengthで、どちらの完成画像を採用したいかを直接比較する。
3. **通常生成との比較:** edit対original、zero対original。実カーネルの違いを含む比較として別表示する。

「局所edit−局所zero」と「all/full edit−all/full zero」の得点差を比べる補助集計も可能だが、評点は順序尺度であるため、厳密な効果の分解や純粋な因果寄与とは呼ばない。完成画像の直接選好を優先する。

### 4.2 画像評価

主評価はpromptを提示した盲検A/B/tie。採点者は「品質とprompt忠実性を踏まえて、最終採用したい画像」を選ぶ。副評価として次を個別画像へ記録する。

| 指標 | 範囲 | 注意 |
|---|---|---|
| 全体品質 | 1–5 | 1=明確に悪い、3=許容、5=良い |
| prompt忠実性 | 1–5 | 指定個数、属性、位置関係等を含む |
| 細部品質 | 1–5 | シャープさだけでなく質感と不自然さを見る |
| 破綻の重症度 | 0–4 | 0=なし、4=重度 |
| 顔／手／文字／物体重複等 | 該当する指標のみ | 適用不能はNA。0点で埋めない |

構図が大きく変わっても、それ自体を失敗や除外理由にしない。prompt違反や品質の低下があるかを評価する。変更量の大きさによる足切りは行わない。

A/Bの左右を固定乱数seed=20261016でランダム化し、条件名、strength、scope、window、元run ID、PNG metadataを採点者へ出さない。prompt以外の条件識別情報は別の`key.private.json`に保存する。画像はリサイズせず、RGB画素が元と一致するmetadataなしPNGを作る。

主対比較数はStage 1で3強度×16=48、Stage 1Nのneutral対curveで16、Stage 2の局所edit対zeroで12×8=96、all/full以外対all/fullで11×8=88。Stage 2の主・実用比較は計184対。採点者ごとの件数であり、生成画像数ではない。その他の比較はsecondaryとして明示追加する。

### 4.3 統計と候補選定

対ごとの勝ち=1、tie=0.5、負け=0として、同じprompt内のseedを先に平均し、そのprompt平均を等重みで集約する。勝ち／負け／tieの生件数、未採点、NA、欠損runを併記する。

95%区間はprompt単位のcluster bootstrap、10000反復、seed=20261016。探索は4 promptしかないため、区間の精度は限定的である。12セルの探索結果を、多重比較調整済みの優越性や一般的な最適設定とはしない。

候補セルは画像選好、破綻、忠実性、別測定の性能を見て人が1つ選び、理由・元データhash・固定時点を`stage2_selection.json`へ保存する。自動ランキングから製品既定値を更新しない。採点者が複数なら採点者別結果を先に出し、同一画像への複数評価を独立promptとして数えない。

### 4.4 attention評価

画像評価とは別出力にする。既存の`rho_text_raw`、centered、radial-binned、text mass、image probabilityの保存性をその定義のまま使用する。

新CLIでは、各runについてstep別・block別の値と、固定観測区間の前半0–3／後半4–7／終盤6–7／全体0–7を出力する。集約順はhead→block→stepの等重み。比較ごとに同じ観測座標を用いる。

**編集区間と評価区間を分ける。** early編集の効果が後半の非編集callにも伝わることがあるため、early条件を前半だけ評価して終えない。同一callのbefore/after差と、別生成run間のtrajectory差も分ける。

観測点が欠けた場合は当該run・metric・区間を未検証にする。nullを0で埋めず、観測できたheadだけで分母を縮めない。rawが有効でcenteredが未定義なら、rawまで自動失格にしない。

originalにはAFMのspectralイベントがないため、画像対照として扱う。必要なattention baselineは別のobserve／zeroから取得する。originalへ架空のspectral PASSを付けない。

---
## 5. データと互換性の仕様

### 5.1 matrixとmanifest

既存matrix schema 1、生成manifest schema 2を維持する。Stage 2の各runへ任意の`ablation`メタデータを追加し、runnerはそれをdeep copyしてmanifestへ渡す。既存の`condition_role`、`pair_id`、`condition_id`、`case_identity`、`expected_coverage`を置き換えない。[R6]

```json
{
  "ablation": {
    "schema_version": 1,
    "study_id": "afm-scope-window-v1",
    "design_sha256": "実際のdesignのcanonical SHA-256",
    "prompt_id": "face_portrait",
    "cell_id": "g0.early",
    "scope_id": "g0",
    "window_id": "early",
    "arm": "edit",
    "condition_key": "g0.early.edit"
  }
}
```

editの既存roleは`preserve_text_mass`、zeroは`manual_zero`、originalは`baseline`とする。originalのcell／scope／windowはnull、`condition_key=native`とする。run IDは既存の安全文字制約・128文字以内に従う。

研究仕様は別の`study-spec.json`（schema 1）に保存する。必須項目はstudy ID、protocol、source Pilotのbytes hash、対象prompt ID／全文hash、seed、block/head数、固定設定、各cell、condition key、contrast一覧、乱数seed、予定画像数、実行段階。実験designのhashはこれらの意味上の値から計算し、hash自身、出力ディレクトリ、実行結果を含めない。生成後のmatrix／specファイルのbytes hashは別の`artifact-hashes.json`に保存して循環参照を避ける。

Stage 1／1Nは既存matrixを変更せず、sidecarのspecで既存controlに対応付ける。neutralの既存controlは`frequency_neutral_2`、比較相手は`preserve_2`である。[R7]

### 5.2 明示contrast

最初の実装で認めるcontrast種別は次の4つに限定する。利用者が任意の一致条件を除去できる汎用ignore-fields機能は作らない。

| 種別 | 比較 | 意味上、異なってよい項目 |
|---|---|---|
| `matched_zero` | セルedit対同セルzero | strengthのみ。双方manual、zero側は0 |
| `scope_window` | セルedit対all/full edit | block_scope、start_percent、end_percentのみ |
| `schedule` | neutral_2対preserve_2 | scheduleのみ |
| `native_image` | editまたはzero対original | AFM有効／無効による差。画像評価専用 |

`zero_strength_mode`は比較対象の双方で`manual`に揃える。scope／window比較ではstrengthを一致させる。ログファイル名、run ID、実行時刻等の識別情報は当然別だが、意味上の入力差から除外するフィールドを明示する。

scope／window比較では、一方がmanual・他方がoriginalとなるcallが予定どおり生じる。これを「runtime全体が同一でなければ比較不可」とはしない。各runをそれぞれの期待適用範囲に対して検証したうえで、共通のbackend／dtype／scale等を照合する。予定外の経路差は拒否する。

Stage 1Nはneutral対curveの比較だけを登録する。neutral対zeroを扱うために`matched_zero`のschedule一致を緩めない。これにより、初版では4種類のcontrast契約を例外なく維持する。

### 5.3 比較時に維持する同一性

比較pairの結合は、run名ではなく、実際のcase identityと証拠から行う。

- prompt全文（positive／negative）、seed、batch、resolution、sampler、scheduler、steps、CFG、denoise、入力画像の有無。
- diffusion／TE／VAEのファイルhash。ファイル名だけでは不足。LoRAが混入したらこのprotocolから除外して理由を報告する。
- AFM推論コードの来歴、ComfyUI、torch、精度、backend、attention scale、モデル構造、比較に影響する実行環境。
- 対象条件とspecに記載した差分以外のAFM設定。
- spectral比較ではmetric定義、cutoff、top-K、radial bins、観測step／block／headの一致。

既存manifestはモデルファイルhashやロード済みソースの証明をすべて自動収集しているわけではない。[R2][R6] 旧実機計画Gate Bのsource／runtime／model証拠を収集し、新解析へ`--evidence-index`で渡す。indexは各source manifestのhashと各runを、対応する証拠ファイルのpath／bytes hashへ結び付ける。

証拠ファイルのhash一致だけで「そのソースが実行中processにロードされている」とは判定しない。旧Gate Bの起動・実ロード確認を人がレビューした記録も必要。確認がない場合はruntime来歴をunverifiedとする。古いログや他runのruntime IDの付け替えは禁止。

### 5.4 複数manifestと再試行

ツールは複数のmanifestを読み取れるようにするが、元manifestの内容は書き換えない。`source_manifest_sha256 + original_run_id`を複合参照として保持し、元の`invocation_id`、prompt ID、runtime ID、history、PNG、JSONLへのリンクを残す。

同じcase／conditionに複数のcompleted runがある場合は「最後のもの」「良い画像」を自動採用しない。初版は曖昧として止め、採用するattemptを明記したreview済みのrun-resolution記録を要求する。全失敗と再試行は別欄に残す。

画像ゼロ件のcompleted、画像複数件で選択元が不明、PNG破損、history失敗、runtime report再利用、hash不一致は検出する。originalはJSONLがN/Aでも、historyと画像・実行来歴の確認は省略しない。

### 5.5 観測coverageと画像の有効性を分離

次を別のstatusとして保持する。

| status | 意味 |
|---|---|
| `execution_validity` | 実行成功・予定した編集・非有限値／fallback等の技術的検査 |
| `artifact_validity` | history／画像／JSONLと元runの対応、hashの検査 |
| `spectral_coverage` | 予定観測点と必要metricが揃ったか |
| `image_rating_status` | 盲検採点が揃ったか |
| `benchmark_validity` | 同期測定・診断off・実sampler実行が確認されたか |

spectral metricのundefinedだけで、技術的に有効な画像を品質評価から捨てない。逆に、画像が存在するだけでcoverageをPASSにしない。欠損理由を分けて出力する。

### 5.6 出力形式

新しい評価CSVのfield名は固定する。既存のratings CSVは上書きせず、移行する場合は明示的な変換として扱う。

- `paired-preference.csv`：`blind_pair_id,rater_id,image_a_id,image_b_id,choice,reason,rubric_version,rated_at`。choiceはA／B／tie、空欄は未採点。blind_pair_idは条件を含まないopaque ID。
- `image-ratings.csv`：`blind_image_id,rater_id,overall_quality_1_to_5,prompt_fidelity_1_to_5,detail_quality_1_to_5,artifact_severity_0_to_4,notes,rubric_version,rated_at`。顔・手等の所見は初版ではnotesへ記録する。
- `key.private.json`：schema_version、study／spec hash、元manifestとrun、blind image／pairの対応、A/B各側のcondition key、元画像と盲検画像のhash、乱数seed。採点者用HTMLにはこの内容を含めない。

- `study-spec.json`、生成matrix、`artifact-hashes.json`：予定条件とその固定。
- 各実行の既存manifest／history／PNG／JSONL：実測証拠。
- `review/images/`、`review.html`、`paired-preference.csv`、`image-ratings.csv`：採点者へ渡す一式。
- `review-private/key.private.json`：条件対応と元画像hash。採点者用フォルダーに置かない。
- `analysis/quality.json`／CSV：画像選好と個別得点。
- `analysis/attention.json`／CSV：coverage、step／block別、固定区間集計。
- `analysis/benchmark.json`／CSV：同期sampling時間とVRAM。全生成時間と誤称しない。
- `analysis/summary.md`：予定／完了／失敗／未採点数、主比較、破綻例、制限事項、候補選定理由。

---

## 6. ファイル構成

新しい抽象的な実験frameworkは作らず、今回のprotocolに必要な小さいモジュールに留める。

| ファイル | 操作 | 責務 |
|---|---|---|
| `scripts/afm_validation_design.py` | 新規 | scope／window定義、研究spec、contrast契約、canonical hash |
| `scripts/build_afm_ablation_matrix.py` | 新規 | 既存PilotからStage 2用matrix／specを送信なしで生成 |
| `scripts/prepare_afm_review.py` | 新規 | 証拠照合、metadataなし画像、盲検CSV／HTMLと非公開key |
| `scripts/analyze_afm_ablation.py` | 新規 | 明示contrast、画像評点／選好、時系列attention、性能の集計 |
| `scripts/run_afm_matrix.py` | 最小変更 | 任意`ablation`とsource matrix hashをmanifestへ引き継ぐ |
| `scripts/analyze_afm_experiment.py` | 最小変更 | 既存coverage validatorを公開wrapper経由で再利用可能にする |
| `tests/test_scope_window_contract.py` | 新規 | 既存推論のscope／区間／zero／curveを固定 |
| `tests/test_afm_validation_design.py` | 新規 | spec、28-block／8-step、contrastの厳密な契約 |
| `tests/test_afm_ablation_matrix.py` | 新規 | 200件、局所zero、既存matrix非改変、dry-run |
| `tests/test_afm_review.py` | 新規 | 盲検、画素不変、欠損／重複／取り違え |
| `tests/test_afm_ablation_analysis.py` | 新規 | 条件間比較、時系列集約、統計、性能の分離 |
| `examples/afm_scope_window_study.json` | 新規 | Stage 2 designの人が読む固定例。選定実績ではない |
| `examples/afm_stage1_study.json` | 新規 | Stage 1／1Nのcondition対応・contrastの固定例。実行証拠ではない |
| `examples/afm_confirmation_prompts.json` | 新規 | §11の独立確認用prompt案とseed |
| `README.md` | 追記 | 新しい検証方針と文書へのリンク、未検証範囲 |
| `docs/hardware-validation-plan-2026-10-15.md` | 冒頭追記 | 本書への参照。過去の定義・数値は保持 |
| `docs/superpowers/plans/2026-10-16-afm-scope-window-validation.md` | 新規 | 本書 |

`afm/spectral.py`、`joint_attention.py`、`backend.py`、`runtime.py`、`config.py`、`nodes.py`は本機能の通常の変更対象にしない。新テストで実際の不具合が判明した場合だけ別の修正タスクに分離し、原因・影響を示す。検証を通すために式や許容差を変更しない。

---

## 7. 実装タスク

### T0：基準の固定と、既存scope／window契約の回帰化

**Files:** Create `tests/test_scope_window_contract.py`。参照する既存実装は`afm/runtime.py::progress_from_sigmas`、`afm/backend.py::Krea2AFMAttentionOverride`、`afm/spectral.py::schedule_alphas`。

**Consumes:** 基準commit、既存テストfixtureと現在のComfyUIソース契約。  
**Produces:** 推論を変更せず、scopeとwindow、zero、global scheduleの動作を検査するCPUテスト。

- [ ] 作業開始時にHEAD、branch、dirty diff、使用Python／torchを記録する。基準から進んでいる場合は差分を読み、本書の仮定を再確認する。ユーザーの未commit変更を消さない。
- [ ] 分離した作業環境を用意し、既存`python -m pytest tests -q`とcompileを実行する。470という件数への一致ではなく、現在の実行結果を記録する。
- [ ] 次のテストを作成する。既存機能の回帰なので、最初からPASSすることは正常。新しい欠陥を発見した場合は修正前に失敗を保存する。

| テスト名 | 固定するassert |
|---|---|
| `test_eight_step_windows_are_disjoint_and_complete` | early={0,1,2,3}、late={4,5,6,7}、共通集合は空、和集合は0–7 |
| `test_late_window_keeps_global_curve_progress` | strength=2、index=4、u=4/7のgainは13/7と15/7。lateの開始をu=0にしない |
| `test_edit_scope_does_not_change_sampler_step_count` | AFM編集の有無にかかわらず8回のmodel evaluationを保持するmock run |
| `test_manual_zero_uses_the_same_scope_and_window` | 指定セルの対象callでmanual、範囲外はoriginal。zeroでも対象callは計上 |
| `test_diagnostics_do_not_expand_edit_scope` | 診断all blocksでも、編集は指定scope／windowだけ |
| `test_requested_strength_is_not_reduced_by_edit_size` | 有効な入力のstrength=3.0をscheduleまで保持。変更量に応じた追加clip／減衰なし |

- [ ] `python -m pytest tests/test_scope_window_contract.py -q`を実行し、その後に全suiteを確認する。
- [ ] 計画外の推論修正がなければテスト追加だけを独立レビューし、実装作業でcommitが許可されている場合のみ小さいcommitにする。

**完了条件:** 全8-step維持、区間の分割、同一scope zero、global curve、診断／編集scopeの独立性がCPUで確認できる。GPU実証とは呼ばない。

### T1：研究specと200画像matrix生成

**Files:** Create `scripts/afm_validation_design.py`、`scripts/build_afm_ablation_matrix.py`、`examples/afm_scope_window_study.json`、`examples/afm_stage1_study.json`、`tests/test_afm_validation_design.py`、`tests/test_afm_ablation_matrix.py`。

**Interfaces:**

```python
# scripts/afm_validation_design.py
resolve_scope(scope_id: str, *, block_count: int) -> tuple[str, tuple[int, ...]]
resolve_window(window_id: str, *, steps: int) -> tuple[float, float, tuple[int, ...]]
build_study_spec(pilot_matrix: dict, *, strength: float,
                 source_pilot_sha256: str) -> dict
build_pilot_spec(pilot_matrix: dict, neutral_matrix: dict | None, *,
                 source_pilot_sha256: str,
                 source_neutral_sha256: str | None) -> dict
validate_study_spec(spec: dict) -> None

# scripts/build_afm_ablation_matrix.py
build_matrix(pilot_matrix: dict, pilot_dir: Path, *, strength: float,
             source_pilot_sha256: str) -> tuple[dict, dict]
main(argv: list[str] | None = None) -> int
```

**Consumes:** 既存Pilot matrixの実内容、Stage 2の場合は非zero有限strength。  
**Produces:** Stage 2ではmatrix schema 1とstudy spec schema 1。Stage 1では既存matrixを変えずにstudy specのみ生成する。ネットワークアクセス・GPU処理なし。

- [ ] 未実装の関数に対する失敗テストを作る。以下のassertを固定する。

```python
matrix, spec = build_matrix(pilot, pilot_dir, strength=2.0,
                            source_pilot_sha256=source_hash)
assert matrix["schema_version"] == 1
assert len(matrix["runs"]) == 200
assert spec["planned_images"] == 200
assert {r["seed"] for r in matrix["runs"]} == {20261015, 20261016}
# 各case: 12 edit + 12 zero + native 1。元pilotのdeep copyは変更されない。
```

- [ ] `python -m pytest tests/test_afm_validation_design.py tests/test_afm_ablation_matrix.py -q`を実行し、新規部分が未実装として失敗することを確認する。
- [ ] scope／window定義とmatrix builderを実装する。28 blocks以外、8 steps以外、CFG≠1、sampler不一致、欠けたsource prompt、重複case、strength=0／NaN／Infは明確な例外にする。
- [ ] strength=3.0や負の有限非zero値も無言補正せず保持するテストを追加する。Stage 1の候補外を使う場合は研究specの別revisionとし、最適値と呼ばない。
- [ ] 各caseのsource workflowを型で確認し、単一のKSampler、AFM node、latent、positive textに対応することを検査する。異なるgraphを固定node番号で誤更新しない。参照workflowは出力先から実在するpathに解決する。
- [ ] originalはAFM off、manual zeroはedit／strength=0／zero_strength_mode=manualとして生成する。全セルで条件外の入力を一致させる。
- [ ] run順序はseed=20261016で再現可能にランダム化する。状態に依存する日時やUUIDをdesignそのものへ埋め込まず、実行識別は既存runnerに任せる。
- [ ] Stage 1用の`build_pilot_spec`を実装する。既存80件のcontrolを変えずに3強度対zeroの48対を登録し、neutralを明示採用した場合のみneutral対preserve_2の16対を追加する。元matrixを結合・改変しない。
- [ ] CLIに`--phase {pilot,screening}`（既定screening）、`--source-pilot`、`--out-dir`を用意する。screeningでは`--strength`必須。pilotでは`--neutral {include,omit}`を必須にし、includeの場合に限り`--source-neutral`を要求する。pilotにstrengthが与えられたら曖昧な指定として拒否する。既存出力を既定で上書きしない。`--execute`は作らない。
- [ ] 新規testsと全suiteを再実行し、レビューする。既存80／16画像matrixのbytes hashが変わっていないことを確認する。

**完了条件:** 指定strengthの12セルと局所zeroが正しく生成され、1ケース25件、8ケース200件、8ステップ維持、dry-run可能。

### T2：runnerのメタデータ引き継ぎとcoverage再利用口

**Files:** Modify `scripts/run_afm_matrix.py`、`scripts/analyze_afm_experiment.py`。Extend `tests/test_afm_ablation_matrix.py`、`tests/test_afm_ablation_analysis.py`。

**Interfaces:**

```python
# scripts/analyze_afm_experiment.py に薄いpublic wrapperを追加
validate_run_coverage(run: dict, report: dict | None) -> tuple[dict, dict]
```

wrapperは既存`_coverage`へ委譲し、判定の内容を変えない。必要に応じて既存内部呼び出しをwrapperへ統一するが、旧解析の結果を固定する回帰を先に置く。

**Consumes:** T1のmatrix、既存runnerとcoverage validator。  
**Produces:** optional `ablation`が保持されたmanifest schema 2、既存と同じcoverage判定。

- [ ] runnerに失敗するテストを追加する。`ablation`がmanifestへそのまま渡ること、`source_matrix_sha256`が実ファイルbytesから計算されることをassertする。
- [ ] dry-runでopenerが一度も呼ばれず、200件すべて`not_submitted`、runtime ID=null、server metrics=unverifiedであるテストを作る。
- [ ] 新しいmetaがない旧matrixでも旧動作が変わらないこと、optional metaからnode入力を上書きできないこと、入力dictを共有して隣のrunが変わらないことをテストする。
- [ ] runnerに必要な引き継ぎだけを実装する。run単位の`ablation`とmanifestのsource matrix hash以外に、送信／再試行／停止制御を変更しない。
- [ ] public wrapperを実装し、既存validatorと同じ結果を返すことを欠損・duplicate・非有限値のfixtureで比較する。
- [ ] `python -m pytest tests/test_afm_ablation_matrix.py tests/test_afm_ablation_analysis.py -q`と全suiteを確認する。

**完了条件:** 旧matrix／旧CLIとの互換性が保たれ、新しい実験因子が実測結果まで追跡できる。既存coverageを「比較できるように」弱めていない。

### T3：盲検画像評価パッケージ

**Files:** Create `scripts/prepare_afm_review.py`、`tests/test_afm_review.py`。

**Interfaces:**

```python
prepare_review(manifest_paths: list[Path], spec: dict, *,
               evidence_index: Path, out_dir: Path, blind_seed: int) -> dict
main(argv: list[str] | None = None) -> int
```

**Consumes:** 実行manifest、spec、旧Gate Bの証拠index、PNG。  
**Produces:** 採点者用画像／HTML／CSV、別ディレクトリの非公開key、検証結果JSON。

- [ ] 合成PNGに条件名とworkflow metadataを入れたfixtureを作り、コピー画像にmetadataが残らず、RGB bytesが一致する失敗テストを置く。
- [ ] 決定的なblind ID、左右順、同case内だけのpair、未採点欄が空であることをテストする。
- [ ] 異なるmanifestのrun ID衝突、runtime ID再利用、PNG hash不一致、missing image、複数画像、実行失敗、未提出を検出するテストを追加する。
- [ ] 最小実装を行う。既存Pillowがなければ不足を報告し、metadataを消せたと装わず終了する。自動pip installを行わない。
- [ ] `review.html`はローカル静的HTMLとし、promptはHTML escapeする。条件名や非公開keyを埋め込まない。Web serverや外部画像送信は不要。
- [ ] CSVはUTF-8、固定field名、空欄は未採点。spreadsheetで開く自由記載欄は式として評価される文字列に注意し、正本のraw JSONと表示用CSVを区別する。
- [ ] `python -m pytest tests/test_afm_review.py -q`と全suiteを確認する。作成できた画像数／予定画像数、未作成理由をsummaryへ残す。

**完了条件:** 画像を変えずに条件を隠せる。keyは採点者に渡す一式から分離され、未採点を架空の得点やtieに変換しない。

### T4：新しい条件間解析

**Files:** Create `scripts/analyze_afm_ablation.py`、`tests/test_afm_ablation_analysis.py`。必要なspec共用関数のみ`afm_validation_design.py`へ追加する。

**Interfaces:**

```python
analyze_study(manifest_paths: list[Path], spec: dict, *,
              evidence_index: Path,
              ratings_path: Path | None,
              preference_path: Path | None,
              key_path: Path | None,
              bootstrap_seed: int = 20261016,
              bootstrap_samples: int = 10000) -> dict
main(argv: list[str] | None = None) -> int
```

**Consumes:** T1の明示spec、T2のrun来歴とcoverage、T3の採点とkey。  
**Produces:** §5.6の品質／attention結果とsummary。採点がなければ品質は未評価。

- [ ] `matched_zero`でscopeやwindowの異なるzeroを拒否するテストを作る。
- [ ] `scope_window`で予定されたscope／window差は受け入れ、モデルhash、prompt、seed、CFG、dtype、cutoff等の不一致は拒否するテストを作る。
- [ ] `schedule`でfrequency_neutral_2とpreserve_2を結び付けるテストを作る。単なるcontrol名の一致では採用しない。
- [ ] originalにspectral reportを要求しない一方で、画像・history・来歴を必要とするテストを作る。
- [ ] 同じ観測点でのearly／late／late2／all集計、1観測点missing、rawのみ有効、診断scope違い、重複eventのテストを作る。
- [ ] 既存`validate_run_coverage`を使い、新しいcontrast契約に沿った照合を実装する。旧`CONTRASTS`や`SHARED_EDIT_FIELDS`を削って既存解析の意味を変えない。
- [ ] 品質の合成fixtureで、勝ち1・tie1・負け0ならraw選好平均0.75になることを確認する。異なるseed数のpromptを単純に全pair平均しないテストを追加する。
- [ ] 欠損採点、unknown blind ID、左右反転key、複数評価者、無効な1–5／0–4の範囲、NAと0の区別、bootstrapの再現性をテストする。
- [ ] `python -m pytest tests/test_afm_ablation_analysis.py -q`と全suiteを確認する。

**完了条件:** 「局所edit対局所zero」と「局所edit対全層edit」を別に報告でき、画像・attention・技術失敗が混同されない。比較不成立の理由と予定／実測の母数を示せる。

### T5：性能集計と段階別の実行手順

**Files:** Extend `scripts/analyze_afm_ablation.py`、`tests/test_afm_ablation_analysis.py`。Update 本書と`docs/hardware-validation-plan-2026-10-15.md`の参照関係。

**Consumes:** 既存benchmark=trueで得る同期server計測。  
**Produces:** all/full editと候補editの同期sampling時間／VRAM比較、cold／warmの分離。

- [ ] 合成reportで診断on、同期なし、cache hit／model evaluationなし、warm-up混入、VRAM null、host違いを検出するテストを作る。
- [ ] 診断off、各条件warm-up 1回以上＋本測定5回以上、固定prompt／seed、条件順交互という性能protocolを記述する。
- [ ] `timing_scope=outer_sample_cuda_synchronized`をsampling時間と表示し、VAE decodeやPNG保存込みの時間に読み替えない。
- [ ] original/offでは既存AFMの同期計測がないことを明示し、時間を0にしない。新しいnative測定wrapperは今回実装しない。
- [ ] `benchmark_validity`をspectral coverageと分けて実装する。診断offのspectral unverifiedを性能測定失敗と混同しない。
- [ ] 診断onのscreening画像から高速化率を算出しないこと、GPU未実行なら性能値はnullであることをテストする。
- [ ] 全suiteとCLIの送信なし統合テストを確認する。

**完了条件:** 性能の主比較は候補edit対all/full editとして成立し、値が未取得なら未測定と表示する。適用call数比を実測高速化率としない。

### T6：文書統合・独立レビュー・引き継ぎ

**Files:** README、本書、旧実機計画への参照追記、`examples/afm_confirmation_prompts.json`。

**Consumes:** T0–T5の実装、CPU結果、送信なし成果物。  
**Produces:** Codex実装完了報告と、まだ実行していないGPU段階の実行手順。

- [ ] READMEへ「演算変更なし、層／区間比較の基盤追加、変更量制限なし」を記載する。実行前の画質改善を記載しない。
- [ ] §11の確認promptをJSON化し、Stage 1／2との重複がないことをhash検査する。
- [ ] §9の検証コマンドを実行し、CPU／dry-runとGPU実績を別欄で報告する。
- [ ] 別のレビュー担当が、変更量制限の混入、推論コードの意図しない変更、来歴検証の緩和、盲検漏洩、データの事後選別を確認する。独立レビュー不能なら未実施と明記する。
- [ ] source diff、テスト出力、旧matrixのhash、新matrix件数、未実装／未実行項目を提出する。
- [ ] GPU実験、サーバー再起動、既定値変更、profile採用、pushは自動的に続けない。

**完了条件:** 検証基盤の実装がCPU／mockで確認され、次に行うGPU作業と、未検証の事項が明確である。

---
## 8. 実機検証gateと、今回追加しないもの

### 8.1 実装完了と実験完了を分ける

| Gate | 確認するもの | 合格しても主張できないこと |
|---|---|---|
| G0：CPU | 回帰、新しいmatrix／解析／盲検処理、無通信dry-run | GPU互換性、画質改善 |
| G1：実機接続 | 実ロード版、モデルhash、28 blocks、head、baseline／zero、予定編集／非編集 | 広いpromptでの品質改善 |
| G2：Stage 1 | 80画像と選択した16画像、盲検採点、attention | 層・区間の最適値 |
| G3：Stage 2 | 200画像、各セルzero、全層との直接比較 | 独立データでの優越 |
| G4：独立確認 | 固定候補、未使用prompt／seed、無選別の結果 | 別モデル・別解像度への一般化 |
| G5：性能 | 診断off、同期、warm-up後5回以上、実sampler実行 | native基準の速度（測定していない場合） |

技術gateの失敗は新規送信を止める。画質が悪い／変化が大きいという理由で画像を落としたり、そのrunだけstrengthを下げて置き換えたりしない。品質が悪い結果も、その設定の評価結果として残す。

実行permissionと技術検証は別である。ツール実装だけを依頼された段階ではG0まで。G1以降は実際の環境と実行許可を確認して行う。

### 8.2 維持する数値検証

旧計画の同一call内text mass／image probabilityのFP32保存誤差基準などを維持する。値を緩めて採用画像を増やさない。[R2]

`afm/metrics.py`の`effect_rel_l2`は、二つの実装が再現する「効果の差」の相対誤差であり、生成画像への変更量そのものの上限ではない。これを理由なく削除したり、新たなstrength制限へ転用したりしない。[R10]

### 8.3 今回の非目標

変更量制限、状態依存gain、entropy gate、層別strength、head別編集、SEGAの移植、CoG、Krea2T Enhancer統合、追加CFG分岐、画像→画像attentionのFFT、reference／video対応、preserveのaugmented化は実装しない。

**先頭3ステップで打ち切るpilot、seed選別、sigma列の短縮、再開生成も今回の別課題である。** 本書のpilotは画質比較を目的とする完成画像生成であり、全8ステップを使う。既存の別プロジェクトの短縮pilotと混同しない。

---

## 9. コマンド契約と検証手順

この節にある新しいCLIは、T1–T5で実装する仕様である。本書作成時点でリポジトリに存在すると扱わない。Pythonは対象ComfyUIまたは承認された評価環境の実行ファイルを使い、依存を勝手に置換しない。

### 9.1 新CLIの引数

| CLI | 必須引数 | 任意引数／既定 |
|---|---|---|
| `build_afm_ablation_matrix.py` | `--source-pilot`, `--out-dir`。screeningでは`--strength`、pilotでは`--neutral` | `--phase=screening`。pilotでneutral=includeなら`--source-neutral`必須。実行機能・既定の上書きなし |
| `prepare_afm_review.py` | `--manifest`（1つ以上）, `--spec`, `--evidence-index`, `--out-dir` | `--blind-seed=20261016` |
| `analyze_afm_ablation.py` | `--manifest`（1つ以上）, `--spec`, `--evidence-index`, `--out-dir` | `--ratings`, `--preferences`, `--key`, `--bootstrap-seed=20261016`, `--bootstrap-samples=10000` |

`--ratings`／`--preferences`を渡す場合は`--key`を必須にする。未採点のままanalysisを実行する場合は品質欄をnot_evaluatedとし、attention／技術的検査だけを出す。

既存runnerの`--matrix`、`--out-dir`、`--comfy-url`、`--execute`の意味は変えない。source matrix hashとoptional ablation metaを追加しても、旧matrixへ必須入力を増やさない。pilotモードは`study-spec.json`と`artifact-hashes.json`、screeningモードはそれらに加えて`matrix.json`を出力する。

### 9.2 CPU・静的検査

```console
python -m pytest tests -q
python -m compileall -q afm nodes.py __init__.py scripts
git diff --check
python scripts/validate_workflow.py examples/krea2_afm_experiment_api.json
python scripts/build_afm_ablation_matrix.py --help
python scripts/prepare_afm_review.py --help
python scripts/analyze_afm_ablation.py --help
```

全コマンドの終了コードを確認する。GPU testの除外や必要なsource-contract検査のskipは別記し、無視して合格にしない。

### 9.3 既存Pilotのspec作成とdry-run

まず既存matrixの条件に対応するspecを作る。次の例はneutralを含める計画だが、GPUへの送信はしない。

```console
python scripts/build_afm_ablation_matrix.py --phase pilot --source-pilot examples/validation_matrix_afm_pilot.json --neutral include --source-neutral examples/validation_matrix_afm_pilot_neutral.json --out-dir outputs/afm-scope-window/stage1-design
```

このpilotモードはstudy-specとartifact hashを出すだけで、80／16画像matrixを書き換えたり1つのmatrixに結合したりしない。

```console
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 --matrix examples/validation_matrix_afm_pilot.json --out-dir outputs/afm-scope-window/stage1-planned
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 --matrix examples/validation_matrix_afm_pilot_neutral.json --out-dir outputs/afm-scope-window/neutral-planned
```

これらは`--execute`がないため通信しない。URLはdry-run用の引数例であり、実サーバーの所在を確認したという意味ではない。80件／16件、全not_submitted、各runtime ID=nullを確認する。

### 9.4 Stage 2 matrix作成とdry-run

```console
python scripts/build_afm_ablation_matrix.py --source-pilot examples/validation_matrix_afm_pilot.json --strength 2.0 --out-dir outputs/afm-scope-window/stage2-design
python scripts/run_afm_matrix.py --comfy-url http://127.0.0.1:8188 --matrix outputs/afm-scope-window/stage2-design/matrix.json --out-dir outputs/afm-scope-window/stage2-planned
```

`2.0`はCPU／dry-run動作確認用の入力例であり、Stage 1未実施のまま最適strengthを確定した値ではない。GPU用matrixは選定記録の値で生成し直し、別のhashとして固定する。

確認項目は200件、8ケース、12セル×2＋native、同セルzero、全8 steps、CFG=1、範囲外の非編集予定、旧matrix不変。すべての新CLIは実行前後で通信回数0をテストする。

### 9.5 実行後のレビュー作成例

実際にGPU実行済みのmanifest、証拠index、specが存在する場合だけ使用する。

```console
python scripts/prepare_afm_review.py --manifest outputs/afm-scope-window/stage2-actual/manifest.json --spec outputs/afm-scope-window/stage2-design/study-spec.json --evidence-index outputs/afm-scope-window/evidence-index.json --out-dir outputs/afm-scope-window/stage2-review
```

`stage2-review/review/`を採点者へ渡し、`stage2-review/review-private/`は渡さない。元画像を上書きせず、対応表を開示する前に採点CSVを凍結してhashを保存する。

### 9.6 実行後の解析例

```console
python scripts/analyze_afm_ablation.py --manifest outputs/afm-scope-window/stage2-actual/manifest.json --spec outputs/afm-scope-window/stage2-design/study-spec.json --evidence-index outputs/afm-scope-window/evidence-index.json --ratings outputs/afm-scope-window/stage2-review/review/image-ratings.csv --preferences outputs/afm-scope-window/stage2-review/review/paired-preference.csv --key outputs/afm-scope-window/stage2-review/review-private/key.private.json --out-dir outputs/afm-scope-window/stage2-analysis
```

実行用manifestを複数に分けた場合は`--manifest`へ複数pathを渡す。元run IDやinvocationを単一実行に書き換える集約はしない。

---

## 10. 受入基準と最終提出物

### 10.1 実装の受入チェックリスト

- [ ] AFM演算、ノードwidget、CFG分岐、samplerのstep数を変更していない。
- [ ] 出力差分clip、変更量由来のstrength減衰、編集量上限が追加されていない。
- [ ] 8-stepのearly／lateが正確に0–3／4–7を選び、sigma列とglobal curveを維持する。
- [ ] Stage 2の全editに同じscope／windowのmanual zeroがある。
- [ ] 200件のmatrixをdry-runでき、送信・モデル評価は発生しない。
- [ ] 旧80／16画像matrix、旧workflow、既存CLIの互換性を維持する。
- [ ] 明示contrast以外の差分、モデルhash不一致、unknown runtime、duplicate runを検出する。
- [ ] 採点者用画像のRGBは元と同一で、条件metadataが除去され、keyが分離されている。
- [ ] 画像採点、spectral coverage、保存性、性能を別々のstatusで報告する。
- [ ] step／block別と固定評価区間のattention集計ができ、missingを0にしない。
- [ ] old analyzerの判定条件を緩めずに新しい比較を実現している。
- [ ] GPU未実行の場合、速度・VRAM・画質改善は未検証と記録する。

### 10.2 Codex実装完了時の報告書

提出内容は、対象HEAD／branch／diff、変更ファイル一覧、CPU実行コマンドと結果、source-contractの状態、dry-run件数、旧ファイルhash、新matrix／spec hash、レビュー結果、未実装事項、GPUで次に行うgate。

「200画像matrixを生成した」と「200画像を生成した」を区別する。予定200件・not_submittedであれば、画像生成数は0と報告する。

### 10.3 実験完了時の報告書

予定画像数、実際の画像数、完了／失敗／再試行、全元画像と盲検画像の対応、採点とkey、prompt別の結果、破綻例、技術gate、attention、性能、独立確認の有無を含める。

第一段階で強度が良く見えたこと、第二段階で特定scopeが良く見えたこと、独立確認で再現されたことをそれぞれ別に記述する。自動の既定値変更や公開は行わない。

---

## 11. 独立確認セット案

T6で以下を`examples/afm_confirmation_prompts.json`へ保存する。seedは20261021／20261022／20261023／20261024を使用する。seedは乱数IDであり、将来の実行日やスケジュールではない。

negative text、モデル、サンプラー等は探索と同じ条件で固定する。prompt本文は生成結果を見る前に凍結する。

| ID | 主な評価対象 | Prompt |
|---|---|---|
| `confirm_portrait_man` | 顔・髪・肌 | A realistic close-up portrait of a middle-aged man with short gray hair and a neatly trimmed beard, subtle skin pores, a relaxed expression, soft light from a window on his right, a plain dark background. |
| `confirm_watch_repair` | 指・小物との接触 | An adult woman repairing a wristwatch at a wooden workbench, holding a small screwdriver in one hand and steadying the watch with the other, both hands fully visible, realistic documentary photography. |
| `confirm_pottery` | 両手・形状の連続性 | An adult potter holding a finished ceramic cup with both hands at chest height, all fingers visible around the cup, natural hand posture, a softly lit pottery studio, realistic photography. |
| `confirm_still_life` | 細部・材質 | A carefully arranged still life with a peeled orange, an embroidered linen cloth, a brass pocket watch, and a clear glass of water on a dark wooden table, natural side lighting, realistic material textures. |
| `confirm_spatial_shapes` | 個数・位置・属性 | Three objects on a white tabletop: a red cube on the left, a green sphere on the right, and a blue cylinder behind the red cube. Exactly three objects, clear separation, soft shadows, a clean studio photograph. |
| `confirm_two_adults` | 複数人物・属性の混同 | Two adults standing side by side in a bright kitchen: a woman in a blue sweater on the left and a man in a gray shirt on the right. Both look toward the camera, their bodies clearly separated, realistic photography. |
| `confirm_shop_sign` | 文字・周辺構造 | The exterior of a small bakery with a single wooden sign reading MORNING BREAD in large clear capital letters, a glass door, two potted plants, warm morning light, realistic architectural photography. |
| `confirm_fabric_detail` | 布・輪郭・金属 | A close-up product photograph of a cream knitted wool jacket with a silver zipper resting on a wooden chair, clearly visible individual yarn loops, natural fabric folds, soft indirect light. |

この8 promptは本書で提案する独立確認用の固定案で、検証済みベンチマークではない。特定カテゴリで有利な画像だけを報告せず、全promptの結果を提示する。

---

## 12. 実装担当へ渡す開始指示

> この計画のT0–T6に従い、AFMの検証基盤を実装してください。変更量制限は不要であり、clip、変更量依存の自動減衰、編集量上限を追加しないでください。既存の推論式とノード入力を維持し、現行80画像Pilot＋明示選択のneutral対照、および層×編集区間の200画像探索を扱えるmatrix・来歴・盲検・比較集計を追加してください。各editには同じscope／windowのmanual zeroを対応させ、全8-stepと元のsigma／curveを維持してください。CPUテストと送信なしdry-runまでを実施し、GPU実験・モデル取得・サーバー再起動・既定値変更・profile更新・pushは自動的に行わないでください。実装前後の差分と検証結果、未検証事項を報告してください。

この開始指示は後で実装担当へ渡すための文章であり、本書を作成した時点で実装を実行したという意味ではない。

---

## 13. 参照したリポジトリ資料

すべて基準commit `3c1adb2994f957082afcad3d57377f0e007da8e1` に固定する。これはソース・既存文書の参照先であり、本書で提案した追加機能の実装証拠ではない。

- **[R1] README**：現在地、既存入力、正規化、zero、診断、対応範囲。  
  [README.md](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/README.md)
- **[R2] 修正後Codex実機検証計画**：80画像、neutral16画像、来歴、盲検、性能、停止条件。  
  [hardware-validation-plan-2026-10-15.md](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/docs/hardware-validation-plan-2026-10-15.md)
- **[R3] runtime**：`progress_from_sigmas`とrunの分離。  
  [afm/runtime.py](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/afm/runtime.py)
- **[R4] 既存解析**：比較group、同一性、coverage、終盤指標。  
  [scripts/analyze_afm_experiment.py](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/scripts/analyze_afm_experiment.py)
- **[R5] 周波数演算**：curveとfrequency-neutral、DC処理。  
  [afm/spectral.py](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/afm/spectral.py)
- **[R6] 既存runner**：matrix→prompt→manifest、条件identity、dry-run。  
  [scripts/run_afm_matrix.py](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/scripts/run_afm_matrix.py)
- **[R7] neutral matrix**：実際のcontrol名、強度、診断設定。  
  [examples/validation_matrix_afm_pilot_neutral.json](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/examples/validation_matrix_afm_pilot_neutral.json)
- **[R8] 設定契約**：strength、scope、start/end、normalization。  
  [afm/config.py](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/afm/config.py)
- **[R9] attention dispatch**：scope／window、zero経路、診断と編集の分離。  
  [afm/backend.py](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/afm/backend.py)
- **[R10] 数値比較指標**：effect_rel_l2は実装間の効果再現誤差。  
  [afm/metrics.py](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/afm/metrics.py)
- **[R11] 既存Pilot matrix**：4 prompt、4 seed、5条件の正本。  
  [examples/validation_matrix_afm_pilot.json](https://github.com/Shiba-2-shiba/AFM-for-Krea2-experimental/blob/3c1adb2994f957082afcad3d57377f0e007da8e1/examples/validation_matrix_afm_pilot.json)

---

**要点:** 今回の実装はAFMを弱める仕組みを追加するものではない。現在のAFM演算を保ち、どの層・生成段階への適用が有効かを、局所zero、全層適用、通常生成と区別して評価するための基盤を整える。
