# 02 — 校準：那個 0.94 準不準？怎麼量、怎麼修

> 01 篇講「confidence 是什麼」。這一篇講 **「它跟真實正確率對不對得起來」**。
> 這是「信心值可參考程度」這個 open question 的核心。

---

## 1. 校準的定義（一句話）

> **一個完美校準的模型，在所有它說「80% 把握」的句子裡，應該剛好答對 80%。**

形式化：對所有信心值 p，`P(預測正確 | confidence = p) = p`。

注意這是一個 **群體性質，不是單句性質**。
你永遠沒辦法驗證「這一句的 0.8 對不對」——0.8 對單一句子沒有可驗證的意義。
只有蒐集一批信心都是 0.8 的句子，看看是不是 80% 答對，才驗得起來。

> 這點很重要，因為下游（品靜那邊）如果想用「信心 < X 就忽略」，
> 她其實是在對 **群體行為** 下賭注，而不是在讀取某句話的可靠度。
> 這個賭注成不成立，完全取決於校準好不好。

---

## 2. 怎麼量：四個指標，看的東西不一樣

### 2.1 ECE（Expected Calibration Error）——最常見，但有坑

做法：把所有預測按信心分箱（例如 10 或 15 箱），每箱算「平均信心」與「實際準確率」，
取兩者差的絕對值，用樣本數加權平均。

```
ECE = Σ_b (n_b / N) · |acc(b) − conf(b)|
```

**ECE = 0.15 的意思**：平均而言，模型講的信心跟實際準確率差了 15 個百分點。

⚠️ **兩個實務上的坑**：

1. **等寬分箱在分數偏斜時會失真。** 我們的模型大部分句子信心都在 0.9 以上，
   於是 0.0–0.9 那九個箱幾乎是空的，ECE 幾乎完全由最後一箱決定，
   而且換個 bins 數字就會大幅變動。
   → 解法：同時報 **等質量分箱（adaptive ECE）**，每箱樣本數相同。
   我們的 `conflib.py` 兩種都算。

2. **ECE 只看最高分那一格。** 其他 7 類的機率再怎麼離譜，ECE 都看不到。

### 2.2 MCE（Maximum Calibration Error）

所有箱裡最差的那一箱的落差。ECE 是平均，MCE 是最壞情況。
要設安全門檻時看 MCE 比看 ECE 有意義。

### 2.3 Brier Score ——看整個機率向量

```
Brier = mean( Σ_k (p_k − onehot_k)² )
```

跟 ECE 互補：ECE 只看第一名，Brier 看整個分布。
Brier 可以分解成「校準項 + 銳利度項」，所以一個把所有句子都輸出
「每類都 1/8」的模型，ECE 可以是完美的 0（因為它 12.5% 的信心真的對應 12.5% 準確率），
但 Brier 會很難看。**只看 ECE 會被這種廢話模型騙過去。**

### 2.4 NLL（Negative Log-Likelihood）

`-log p_true` 的平均。溫度縮放的最佳化目標就是它。
對「答錯又很有信心」懲罰極重（log 在 0 附近趨近無限大），
所以 NLL 對過度自信特別敏感。

**建議一起看的組合**：`ECE(等質量) + MCE + Brier + NLL`。
四個都動才是真的變好，只有 ECE 變好可能是分箱假象。

---

## 3. 可靠度圖（Reliability Diagram）怎麼讀

X 軸信心、Y 軸實際準確率，畫對角線當參考：

```
準確率
1.0 |                                    ╱ ← 完美校準（對角線）
    |                                 ╱
0.8 |                          ╱   ●        ● 落在對角線下方
    |                       ╱    ●            = 過度自信（overconfident）
0.6 |                    ╱   ●                  講 0.9 卻只有 0.7 準確率
    |                 ╱  ●
0.4 |              ╱ ●
    |___________╱________________________
      0.4   0.6   0.8   1.0   信心
```

- **點在對角線下方** → 過度自信（絕大多數深度模型的預設狀態）
- **點在對角線上方** → 保守（label smoothing 或溫度 <1 過頭時會出現）
- **點分布很集中在右上角** → 模型「什麼都很有把握」，這時 ECE 數字漂亮但沒鑑別力

---

## 4. 預訓練 Transformer 的校準：好消息與壞消息

Desai & Durrett《Calibration of Pre-trained Transformers》(EMNLP 2020) 是這領域對
BERT/RoBERTa 最直接的研究，結論很具體：

| 情境 | 結論 |
|---|---|
| **同領域（in-domain）** | 預訓練 Transformer **開箱即用就相當校準**，比非預訓練的基線好很多 |
| **跨領域（out-of-domain）** | 校準誤差變大，但仍比基線 **低到 3.5 倍** |
| **同領域修正** | **溫度縮放很有效**，能再顯著降低 ECE |
| **跨領域修正** | 溫度縮放幫助有限；**label smoothing**（訓練時刻意增加不確定性）在 OOD 上比較有用 |

> 這對我們的直接含意：
> - 我們的模型跑 **同作者的對話資料集** ≈ 同領域 → 校準可能不差，溫度縮放有機會修好。
> - 跑 **SMP2020 微博** ≈ 跨領域 → 校準會退化，而且溫度縮放救不太動。
> - 真實使用場景（Moshi 轉錄的口語）是 **第三個領域**，兩份測試集都沒涵蓋。
>   這是評估上的一個真實缺口，不是可以靠算式補的。

而 Ovadia et al. 2019《Can You Trust Your Model's Uncertainty?》大規模驗證了更嚴厲的一點：

> **「傳統的事後校準（post-hoc calibration）在分布偏移下會失效。」**
> 在校準集上調好的溫度，換到偏移後的資料上就不準了；
> 而且 **偏移越嚴重，準確率掉得越多，校準誤差也同步惡化** ——
> 正好是最需要可靠不確定性的時候，它最不可靠。

---

## 5. 溫度縮放（Temperature Scaling）——CP 值最高的修法

### 5.1 做法

```
p_i = softmax(z_i / T)
```

一個純量 T，在 **驗證集上** 用最小化 NLL 求出來。就這樣。

- `T > 1` → 把分數壓平（原本過度自信）
- `T < 1` → 把分數拉尖（原本不夠自信）
- `T = 1` → 不動

### 5.2 為什麼它幾乎是免費的

| 性質 | 說明 |
|---|---|
| **不改變 argmax** | 除以正數是單調變換，排序不變 → **準確率一定不變**，零風險 |
| **不用重訓** | 只在推論後多做一次 softmax |
| **一個參數** | 一條驗證集就夠，不會過擬合 |
| **推論成本** | 一次除法。對延遲的影響可以視為 0 |

Guo et al. 的結論就是：這麼簡單的方法，效果打平或勝過一堆複雜方法。

### 5.3 它救不了什麼（重要）

1. **不會讓模型變準。** 準確率一模一樣，只是把分數講得誠實一點。
2. **不改變排序。** 所以 **選擇性預測的 AUROC / AURC 完全不變**——
   如果你只是想「用信心排序、砍掉最爛的那批」，溫度縮放對你毫無幫助。
   它只在你需要「0.8 這個數字本身有意義」時才有用。
3. **分布偏移下會失效**（Ovadia 2019）。校準集必須跟實際流量同分布。

> **對我們最關鍵的一句**：如果下游只是要「信心低於 X 就忽略」，
> 那要看的是 **AUROC / 風險-覆蓋率曲線**（03 篇），
> 而溫度縮放 **改變不了那件事**。溫度縮放只解決「數字要不要能當機率讀」。

### 5.4 一個技術細節（我們的實作）

我們只存了機率沒存 logits。但 softmax 對加常數不變，
所以 `log(p)` 就是一組合法的等價 logits，做溫度縮放完全正確
（見 `conflib.py: probs_to_logits`）。
唯一算不出來的是需要 logits 絕對尺度的方法（energy score）。

---

## 6. 其他校準方法（知道有就好）

| 方法 | 要動什麼 | 適用 | 對我們 |
|---|---|---|---|
| **Temperature Scaling** | 事後，1 參數 | 同領域 | ✅ 先試這個 |
| **Vector / Matrix Scaling** | 事後，K 或 K² 參數 | 類別多時容易過擬合 | 8 類還行，但收益通常不如 TS |
| **Label Smoothing** | 要重訓 | OOD 校準較佳（Desai & Durrett） | 只有走微調路線才考慮 |
| **Focal Loss** | 要重訓 | 天然比 CE 校準好 | 同上 |
| **Histogram / Isotonic Regression** | 事後，非參數 | 資料多時 | 需要較大校準集，8 類 ×1200 句偏少 |
| **Deep Ensembles** | 訓多個模型 | 校準最強，但成本 ×N | ❌ 即時性硬需求下不可能 |
| **MC Dropout** | 推論多次 | 中等 | ❌ 推論成本 ×N，跟 17ms 目標衝突 |

（後兩個在 04 篇細講為什麼對我們不可行。）

---

## 7. 我們自己的數字

實驗腳本：`research/confidence/experiments/exp_calibration.py`
結果：`research/confidence/results/calibration.json` 與 `RESULTS.md`

三份資料：
- **A 同領域**：`_local/out/stream_cpu.jsonl`，200 句，原生 8 類
- **B 跨領域**：`eval/results/johnson-small/cpu/{zh_cn,zh_tw}/predictions.jsonl`，1200 句 SMP2020
- **C 截斷壓力**：`_local/stress/fragment_detail.jsonl`，200 對

→ 數字與解讀見 `research/confidence/RESULTS.md`。

---

## 8. 這一篇的結論

1. 校準 = 「說 80% 就真的對 80%」，是 **群體性質**，單句無法驗證。
2. 量測要 **ECE（等質量）+ MCE + Brier + NLL 一起看**，只看 ECE 會被騙。
3. 預訓練 Transformer **同領域校準不錯、跨領域退化**（Desai & Durrett 2020）。
4. **溫度縮放是零風險、零成本的修法**，但它 **只修數字、不改排序**——
   如果目的是設門檻過濾，它幫不上忙。
5. **分布偏移下事後校準會失效**（Ovadia 2019），而我們的真實場景（口語 STT 輸出）
   跟兩份測試集都不同分布。

---

**下一篇**：`03_信心值的用途.md` —— 排序、設門檻、偵測異常，哪些做得到哪些做不到。

## 參考

- [Calibration of Pre-trained Transformers (Desai & Durrett, EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.21/) · [PDF](https://arxiv.org/pdf/2003.07892) · [程式碼](https://github.com/shreydesai/calibration)
- [On Calibration of Modern Neural Networks (Guo et al., ICML 2017)](https://arxiv.org/pdf/1706.04599)
- [Can You Trust Your Model's Uncertainty? (Ovadia et al., NeurIPS 2019)](https://arxiv.org/abs/1906.02530)
- [When Does Label Smoothing Help? (Müller et al., NeurIPS 2019)](https://arxiv.org/pdf/1906.02629)
- [Bag of Tricks for In-Distribution Calibration of Pretrained Transformers](https://arxiv.org/pdf/2302.06690)
