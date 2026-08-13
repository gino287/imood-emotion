# 01 — softmax 信心值到底是什麼

> 這一篇只講「那個 0.94 是怎麼來的、它在數學上代表什麼」。
> 能不能信、怎麼用，在 02、03 篇。

---

## 1. 它的定義：一個被歸一化的相對分數

分類模型最後一層吐出的是 **logits**（未歸一化分數），八個類別就是八個實數，
可以是負的、可以大於 1、彼此沒有機率意義。softmax 把它變成加總為 1 的向量：

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)        T = 1（預設）
```

我們程式裡取的 `confidence` 就是 `max(p)`，文獻上通稱
**MSP（Maximum Softmax Probability）**。

`imood_emotion/classifier.py` 的這段：

```python
probs = torch.softmax(logits, dim=-1)[0].tolist()
best = max(raw, key=raw.get)
return Prediction(label=best, confidence=raw[best], probs=raw, ...)
```

所以 `confidence=0.944` 的字面意思只是：**「在這八個候選裡，厭惡語調拿到了 94.4% 的相對票數」**。

### 這裡有三個容易誤會的點

**(a) 它是相對的，不是絕對的。**
softmax 的分母只包含這 8 個類別。一句完全不屬於這 8 類的話（例如「今天禮拜三」
的中性陳述，或一段亂碼），模型不會輸出「都不是」——它只能在 8 個裡挑一個，
而且分數照樣可以很高。分母裡沒有「以上皆非」這個選項。

**(b) softmax 對加常數不變。**
`softmax(z) == softmax(z + c)`。這代表 logits 的絕對大小（可以理解成「模型看到多少證據」）
在 softmax 之後就被丟掉了，只剩下差值。這是後面「energy score 為什麼比 MSP 好」的根源
（見 03 篇）：energy 用的是 logsumexp(z)，那個被 softmax 丟掉的量。

**(c) 指數放大差距。**
logits 差 3（例如 5.0 vs 2.0）→ softmax 出來約 0.95 vs 0.05。
logits 只要差一點點，機率就會非常極端。所以 **高信心不代表模型「很確定」，
只代表 logits 之間差得夠開**，而 logits 的尺度是訓練過程的產物，沒有校準過。

---

## 2. 為什麼訓練出來的模型幾乎必然過度自信

訓練用的是 cross-entropy loss：`-log p_true`。

- 這個 loss 只有在 `p_true → 1` 時才會趨近 0。
- 也就是說，**只要模型還能把正確類的機率再推高一點，loss 就還能再降一點**，
  即使它早就已經答對了。
- 於是最佳化過程會持續把 logits 往外拉開，直到被正則化或早停擋住。

Guo et al. 2017《On Calibration of Modern Neural Networks》就是在講這件事：
現代神經網路（相對於 90 年代的淺層網路）**分類準確率變好了，但校準變差了**。
論文歸因於容量變大、batch norm、以及 weight decay 變弱——模型有足夠容量去
記住訓練集並把每一筆都推到接近 1。

> 對我們的意義：模型輸出 0.99 不是因為它「看得很清楚」，
> 有很大一部分是因為 **訓練目標在獎勵它輸出 0.99**。

---

## 3. 「信心」其實混了兩種不同的不確定性

文獻上把預測不確定性拆成兩塊，這個區分對情緒分類特別重要：

| 類型 | 中文 | 來源 | 舉例 | 能不能靠更多資料消除 |
|---|---|---|---|---|
| **Aleatoric** | 資料本身的不確定性 | 標註本身就有歧義 | 「哦，是喔」可以是平淡、可以是厭惡 | ❌ 不能 |
| **Epistemic** | 模型的無知 | 沒看過這種輸入 | 微博用語、被切斷的半句話 | ✅ 可以（更多資料/更大模型） |

**softmax 的 MSP 把兩者混在一起，而且對 epistemic 特別不敏感。**
這是它最大的弱點：模型面對完全沒看過的輸入時，並不會輸出「我不知道」，
它會很有自信地亂猜。

這件事跟你已經量到的數字完全對得起來：

> 截斷壓力測試：信心只降 7.7%，但預測類別改變達 29.5%。

「句子被切斷」是典型的 epistemic 不確定性（分布外輸入），
而 MSP 對這種情況幾乎沒有反應——**這不是模型壞掉，這是 MSP 的已知性質**。
（詳細數字重算與更多角度見 `results/` 與 05 篇。）

---

## 4. 情緒分類的特殊處境：標籤本身就不唯一

一般分類任務（貓/狗）有客觀答案，情緒沒有。搜到的研究裡有一組很直白的數字：

> 在 EmotionLines 資料集裡，**72.6% 的語句在 5 位標註者之間有分歧，
> 14.3% 甚至沒有多數決結果。**

這代表：

1. 對很多句子而言，「正確答案」根本是一個分布而不是一個類別。
   硬壓成單一標籤，是把不確定性的資訊直接扔掉。
2. 用單一標籤訓練出來的模型，被迫學會「對一個本質模糊的東西給出確定答案」——
   它學到的信心分布必然跟人類的實際共識程度脫節。
3. 所以在情緒任務上，**即使模型校準得很好，高信心也不等於「這句話真的就是這個情緒」**，
   只等於「多數標註者會這樣標」。

> 這對本專案的直接含意：`厭惡→fear` 那個映射爭議、
> 以及「憤怒被讀成悲傷」的 60% 準確率問題，有一部分不是模型的錯，
> 是任務本身的 aleatoric 上限。要區分「模型爛」和「題目本身沒有唯一答案」，
> 需要的是標註者分歧資料（我們現在的資料集沒有），不是更高的信心值。

---

## 5. 怎麼查看：實務操作

### 5.1 我們的程式已經存了什麼

`Prediction` 這個 dataclass 已經是最完整的形式了：

```python
Prediction(
    label="厭惡語調",
    confidence=0.944036,        # = max(probs)
    probs={...8 類完整分布...},   # ← 關鍵，很多專案只存 argmax 就沒了
    latency_ms={...},
)
```

**存完整機率分布這個決策現在開始回本**：底下所有分析（校準、溫度縮放、
margin、entropy、映射後合併機率）全部都能用既有的 jsonl 重算，一次模型都不用重跑。

### 5.2 從機率向量能再算出來的東西

同一個機率向量可以換算成好幾個不同的「不確定度」指標，各有各的敏感點：

| 指標 | 公式 | 敏感於什麼 | 什麼時候比 MSP 好 |
|---|---|---|---|
| **MSP** | `max(p)` | 第一名有多高 | 基準線 |
| **Margin** | `p_(1) - p_(2)` | 前兩名有多接近 | 「在兩類之間猶豫」時 MSP 可能還有 0.5，margin 會趨近 0 |
| **Entropy** | `-Σ p log p` | 整個分布有多平 | 「八類都有一點點」時 |
| **Energy** | `-logsumexp(z)` | logits 的絕對大小 | 偵測分布外輸入（見 03 篇）|

⚠️ **注意**：我們只存了機率沒存 logits，所以 energy 算不出來。
`log(p)` 只能還原到「差一個常數」的 logits，而 energy 要的正好是那個常數。
若之後要試 energy score，`classifier.py` 要多存一欄 raw logits。
（這是目前資料的一個實際限制，值得記一筆。）

### 5.3 兩行就能看到分布長怎樣

```python
# 看某句的完整分布，按機率排序
p = clf.predict("你怎麼這麼討厭啊")
for lab, v in sorted(p.probs.items(), key=lambda kv: -kv[1]):
    print(f"{lab:6} {v:.4f} {'█' * int(v * 50)}")
```

---

## 6. 這一篇的結論

1. `confidence` 是 **8 個候選之間的相對票數**，不是「這句話有 94% 機率是厭惡」。
2. cross-entropy 訓練 **在結構上就在獎勵過度自信**，所以高分是預設狀態，不是特例。
3. MSP 混了兩種不確定性，而且 **對「沒看過的輸入」幾乎不反應**——
   這正好解釋了截斷測試那個 7.7% vs 29.5% 的落差。
4. 情緒任務的標籤本身有大量分歧，信心值天生有個天花板。
5. 我們存了完整機率分布，所以所有進階分析都不用重跑模型；
   唯一缺的是 raw logits（energy score 用得到）。

---

**下一篇**：`02_校準.md` —— 那個 0.94 到底準不準？怎麼量？怎麼修？

## 參考

- [On Calibration of Modern Neural Networks (Guo et al., ICML 2017)](https://arxiv.org/pdf/1706.04599)
- [Calibration in Machine Learning: Confidence, Accuracy & ECE](https://mbrenndoerfer.com/writing/calibration-machine-learning-confidence-accuracy-ece)
- [Soft-Label Training Preserves Epistemic Uncertainty](https://arxiv.org/pdf/2511.14117)
- [Don't waste a single annotation: improving single-label classifiers through soft labels (EMNLP Findings 2023)](https://aclanthology.org/2023.findings-emnlp.355/)
