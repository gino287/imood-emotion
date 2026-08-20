# 待GINO改寫
"""L5：產出人工看表情用的檢視頁（單一 HTML，封面圖內嵌）。

    python sourcing/deliver/make_review_page.py
    python sourcing/deliver/make_review_page.py --min-labels 4 --out review.html

這一頁的用途只有一個：**讓人用最少的點擊確認候選人的表情能不能用。**

每個切點都做成 `bilibili.com/video/BVxxx?t=秒數` 的連結，點下去直接跳到
那個情緒發生的那一秒，不必自己拉進度條。四種情緒各列幾個時間戳，
掃過去就知道這個人的表情夠不夠用。

封面圖用 data URI 內嵌，所以整頁是一個檔案，丟到哪裡都能開。
"""
import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402

# 只列這四類。中性不列 —— 要挑的是「有表情」的片段，中性臉對挑選沒有幫助。
TARGET = ["哀", "怒", "樂", "驚"]
CLIPS_PER_LABEL = 4
THUMB_W = 320


def thumb_data_uri(uid: str) -> str:
    """封面縮到 320px 寬再轉 base64。原圖動輒幾百 KB，二十幾張就把頁面撐爆。"""
    src = paths.COVER_DIR / f"{uid.replace(':', '_')}.jpg"
    if not src.exists():
        return ""
    try:
        from PIL import Image
    except ImportError:
        return ""
    try:
        img = Image.open(src).convert("RGB")
    except Exception:
        return ""
    w, h = img.size
    img = img.resize((THUMB_W, max(1, round(h * THUMB_W / w))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def load() -> list:
    if not paths.TIMELINE_DIR.exists():
        raise SystemExit(f"還沒有任何時間軸（{paths.TIMELINE_DIR}）")
    out = []
    for f in sorted(paths.TIMELINE_DIR.glob("*.json")):
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def card_html(r: dict) -> str:
    got = [c for c in TARGET if c in (r.get("labels_present") or [])]
    thumb = thumb_data_uri(r["uid"])
    ratio = (r.get("non_neutral_ratio") or 0) * 100

    # 四段式覆蓋條：哪幾種情緒達標。這是真實狀態，不是裝飾。
    meter = "".join(
        f'<span class="seg {"on" if c in got else "off"}" data-emo="{c}" '
        f'title="{c}{"（達標）" if c in got else "（不足）"}"></span>'
        for c in TARGET
    )

    cols = []
    for c in TARGET:
        clips = (r.get("clips") or {}).get(c, [])[:CLIPS_PER_LABEL]
        n_all = len((r.get("clips") or {}).get(c, []))
        if not clips:
            cols.append(
                f'<div class="col" data-emo="{c}">'
                f'<div class="col-head"><b>{c}</b><span class="n">0</span></div>'
                f'<p class="none">沒有片段</p></div>'
            )
            continue
        chips = "".join(
            f'<a class="chip" target="_blank" rel="noopener" '
            f'href="{html.escape(r["url"])}?t={int(cl["start"])}">'
            f'<span class="t">{mmss(cl["start"])}</span>'
            f'<span class="q">{html.escape((cl["text"] or "")[:26])}</span></a>'
            for cl in clips
        )
        cols.append(
            f'<div class="col" data-emo="{c}">'
            f'<div class="col-head"><b>{c}</b><span class="n">{n_all}</span></div>'
            f'{chips}</div>'
        )

    img = (f'<img src="{thumb}" alt="" loading="lazy">' if thumb
           else '<div class="noimg">無封面</div>')

    return f"""<article class="card">
  <div class="cover">{img}</div>
  <div class="body">
    <h3><a href="{html.escape(r['url'])}" target="_blank" rel="noopener">{html.escape(r['title'])}</a></h3>
    <p class="meta">
      <span class="who">{html.escape(r.get('author_name') or '')}</span>
      <span class="dot">·</span><span>{mmss(r.get('duration_s') or 0)}</span>
      <span class="dot">·</span><span>起伏 {ratio:.0f}%</span>
      <span class="meter">{meter}</span>
      <span class="count">{len(got)}/4</span>
    </p>
    <div class="cols">{''.join(cols)}</div>
  </div>
</article>"""


PAGE = """<title>情緒素材候選台</title>
<style>
:root {{
  --ground:#f4f6f9; --surface:#ffffff; --line:#dfe4ec;
  --ink:#1b2027; --muted:#69727f; --faint:#8d96a3;
  --accent:#2f6fdb; --accent-soft:#e8effb;
  --e-ai:#5a6fb5; --e-nu:#c8442e; --e-le:#c07d00; --e-jing:#0f8a86;
  --off:#cbd2dd;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#11141a; --surface:#191e26; --line:#2b323d;
    --ink:#e4e8ee; --muted:#98a2b0; --faint:#7d8794;
    --accent:#6ea1f5; --accent-soft:#1d2836;
    --e-ai:#8496d8; --e-nu:#e2705c; --e-le:#e0a53a; --e-jing:#3cb5b0;
    --off:#39414d;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#11141a; --surface:#191e26; --line:#2b323d;
  --ink:#e4e8ee; --muted:#98a2b0; --faint:#7d8794;
  --accent:#6ea1f5; --accent-soft:#1d2836;
  --e-ai:#8496d8; --e-nu:#e2705c; --e-le:#e0a53a; --e-jing:#3cb5b0;
  --off:#39414d;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI","PingFang TC","Microsoft JhengHei",sans-serif;
  line-height:1.55; font-size:15px;
}}
.wrap {{ max-width:1120px; margin:0 auto; padding:40px 22px 72px; }}
header {{ border-bottom:2px solid var(--ink); padding-bottom:18px; margin-bottom:8px; }}
h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.01em; text-wrap:balance; }}
.sub {{ margin:0; color:var(--muted); font-size:14px; max-width:64ch; }}
.stats {{ display:flex; gap:26px; flex-wrap:wrap; padding:16px 0 26px; }}
.stat b {{ display:block; font-size:24px; font-variant-numeric:tabular-nums; }}
.stat span {{ font-size:12px; color:var(--faint); letter-spacing:.06em; text-transform:uppercase; }}
.card {{
  display:grid; grid-template-columns:200px 1fr; gap:18px;
  background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:16px; margin-bottom:14px;
}}
.cover img {{ width:100%; border-radius:6px; display:block; }}
.noimg {{
  aspect-ratio:16/9; display:grid; place-items:center;
  background:var(--ground); border-radius:6px; color:var(--faint); font-size:12px;
}}
h3 {{ font-size:16px; margin:0 0 4px; line-height:1.35; text-wrap:balance; }}
h3 a {{ color:var(--ink); text-decoration:none; }}
h3 a:hover, h3 a:focus-visible {{ color:var(--accent); text-decoration:underline; }}
.meta {{
  margin:0 0 12px; font-size:13px; color:var(--muted);
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
}}
.who {{ color:var(--ink); font-weight:600; }}
.dot {{ color:var(--off); }}
.meter {{ display:inline-flex; gap:3px; margin-left:4px; }}
.seg {{ width:16px; height:7px; border-radius:2px; background:var(--off); }}
.seg.on[data-emo="哀"] {{ background:var(--e-ai); }}
.seg.on[data-emo="怒"] {{ background:var(--e-nu); }}
.seg.on[data-emo="樂"] {{ background:var(--e-le); }}
.seg.on[data-emo="驚"] {{ background:var(--e-jing); }}
.count {{ font-variant-numeric:tabular-nums; font-weight:600; color:var(--ink); }}
.cols {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }}
.col-head {{
  display:flex; justify-content:space-between; align-items:baseline;
  padding-bottom:5px; margin-bottom:6px; border-bottom:2px solid var(--off);
}}
.col[data-emo="哀"] .col-head {{ border-color:var(--e-ai); }}
.col[data-emo="怒"] .col-head {{ border-color:var(--e-nu); }}
.col[data-emo="樂"] .col-head {{ border-color:var(--e-le); }}
.col[data-emo="驚"] .col-head {{ border-color:var(--e-jing); }}
.col-head b {{ font-size:14px; }}
.col-head .n {{ font-size:11px; color:var(--faint); font-variant-numeric:tabular-nums; }}
.chip {{
  display:block; text-decoration:none; color:inherit;
  border:1px solid var(--line); border-radius:6px;
  padding:5px 7px; margin-bottom:4px; font-size:12px;
}}
.chip:hover, .chip:focus-visible {{ border-color:var(--accent); background:var(--accent-soft); }}
.chip .t {{
  display:block; font-family:ui-monospace,Consolas,monospace;
  font-variant-numeric:tabular-nums; font-weight:600; font-size:12px;
}}
.col[data-emo="哀"] .chip .t {{ color:var(--e-ai); }}
.col[data-emo="怒"] .chip .t {{ color:var(--e-nu); }}
.col[data-emo="樂"] .chip .t {{ color:var(--e-le); }}
.col[data-emo="驚"] .chip .t {{ color:var(--e-jing); }}
.chip .q {{ display:block; color:var(--muted); line-height:1.35; margin-top:1px; }}
.none {{ font-size:12px; color:var(--faint); margin:2px 0 0; }}
.note {{
  border-left:3px solid var(--accent); background:var(--accent-soft);
  padding:12px 14px; border-radius:0 8px 8px 0; font-size:13.5px;
  margin:0 0 26px; color:var(--ink);
}}
.note p {{ margin:0 0 6px; }} .note p:last-child {{ margin:0; }}
@media (max-width:860px) {{
  .card {{ grid-template-columns:1fr; }}
  .cols {{ grid-template-columns:repeat(2,1fr); }}
}}
</style>

<div class="wrap">
<header>
  <h1>情緒素材候選台</h1>
  <p class="sub">點任何一個時間戳會直接跳到 B 站的那一秒，用來確認該片段的臉部表情能不能用。</p>
</header>

<div class="stats">
  <div class="stat"><b>{n_total}</b><span>已分析</span></div>
  <div class="stat"><b>{n_full}</b><span>四種齊全</span></div>
  <div class="stat"><b>{n_three}</b><span>三種</span></div>
  <div class="stat"><b>{best}%</b><span>最高起伏</span></div>
</div>

<div class="note">
  <p><b>怎麼看：</b>「起伏」是非中性段落的佔比，數字越高代表這個人講話時情緒變化越多。
  四格條顯示哀／怒／樂／驚哪幾種達標（同一類至少 3 段才算，只有一兩段的多半是誤判）。</p>
  <p><b>要注意：</b>這些判定來自<b>逐字稿的語意</b>，不是臉部表情。有人會面無表情地說「我好生氣」。
  所以還是得點進去看臉——這一頁的作用是把要看的地方縮到幾十個時間點，不是替你決定。</p>
</div>

{cards}
</div>
"""


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L5 產生人工檢視頁")
    p.add_argument("--out", type=Path, default=paths.REVIEW_PAGE)
    p.add_argument("--min-labels", type=int, default=3,
                   help="至少幾種目標情緒達標才列出（預設 3）")
    args = p.parse_args()

    rows = load()
    for r in rows:
        r["_got"] = [c for c in TARGET if c in (r.get("labels_present") or [])]
    rows = [r for r in rows if len(r["_got"]) >= args.min_labels]
    # 先看情緒齊不齊，再看起伏大小
    rows.sort(key=lambda r: (-len(r["_got"]), -(r.get("non_neutral_ratio") or 0)))

    if not rows:
        raise SystemExit("沒有符合條件的候選，把 --min-labels 調低試試")

    ratios = [(r.get("non_neutral_ratio") or 0) * 100 for r in rows]
    page = PAGE.format(
        n_total=len(rows),
        n_full=sum(1 for r in rows if len(r["_got"]) == 4),
        n_three=sum(1 for r in rows if len(r["_got"]) == 3),
        best=f"{max(ratios):.0f}",
        cards="\n".join(card_html(r) for r in rows),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"{len(rows)} 個候選 → {args.out}（{size_kb:.0f} KB）")


if __name__ == "__main__":
    main()
