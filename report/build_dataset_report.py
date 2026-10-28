"""Builds report/Dataset_Report.docx with statistics computed from the real files."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

RAW, P, OUT = "data/raw/", "data/processed/", "results/"
BLUE, ORANGE = "#4c78a8", "#e8743b"
raw = {s: pd.read_csv(f"{RAW}Video_Games.{s}.csv") for s in ("train", "valid", "test")}
tr, va, te = (pd.read_csv(f"{P}{s}.csv") for s in ("train", "val", "test"))
items = pd.read_csv(P + "items.csv")
nu, ni = tr.user.nunique(), len(items)
ym = lambda ms: pd.to_datetime(ms, unit="ms").strftime("%b %Y")

# ---- figures
def style(ax): ax.spines[["top", "right"]].set_visible(False)

r = raw["train"].rating.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 3.2)); ax.bar(r.index.astype(int), r.values / 1e3, color=BLUE)
ax.axvline(3.5, ls="--", color=ORANGE); ax.text(3.6, r.max() / 1e3 * .9, "positive\n(rating >= 4)", color=ORANGE)
ax.set(xlabel="rating", ylabel="reviews (thousands)", title="Rating distribution (raw 5-core train)"); style(ax)
fig.tight_layout(); fig.savefig(OUT + "ds_ratings.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, s, t in ((ax[0], tr.groupby("user").size(), "Interactions per customer"), (ax[1], tr.groupby("item").size(), "Interactions per product")):
    a.hist(s, bins=np.logspace(0, np.log10(s.max() + 1), 40), color=BLUE); a.set(xscale="log", yscale="log", xlabel="count (log)", ylabel="nodes (log)", title=t); style(a)
fig.tight_layout(); fig.savefig(OUT + "ds_degrees.png", dpi=150); plt.close()

allr = pd.concat(raw.values())
yr = pd.to_datetime(allr.timestamp, unit="ms").dt.year.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 3.2)); ax.bar(yr.index, yr.values / 1e3, color=BLUE)
ax.set(xlabel="year", ylabel="reviews (thousands)", title="Reviews per year (raw 5-core, all splits)"); style(ax)
fig.tight_layout(); fig.savefig(OUT + "ds_years.png", dpi=150); plt.close()

plat = items.category.fillna("").str.split(" > ").str[0].replace("", "Unknown")
pc = plat.value_counts().head(10)[::-1]
fig, ax = plt.subplots(figsize=(6, 3.6)); ax.barh(pc.index, pc.values, color=BLUE)
ax.set(xlabel="products", title="Top platforms / categories among kept products"); style(ax)
fig.tight_layout(); fig.savefig(OUT + "ds_categories.png", dpi=150); plt.close()

# ---- document
d = Document(); d.styles["Normal"].font.name = "Calibri"; d.styles["Normal"].font.size = Pt(11)
H = lambda t, l=1: d.add_heading(t, l)
para = lambda t: d.add_paragraph(t)
def table(head, rows):
    t = d.add_table(rows=1, cols=len(head)); t.style = "Light Grid Accent 1"
    for c, h in zip(t.rows[0].cells, head): c.text = h
    for r_ in rows:
        for c, v in zip(t.add_row().cells, r_): c.text = str(v)
    d.add_paragraph()
def fig_(path, cap, w=5.6):
    d.add_picture(path, width=Inches(w)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = d.add_paragraph(cap); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.runs[0].italic = True

d.add_heading("Dataset Report", 0)
para("Retail Product Recommendation Using GNN - Amazon Reviews 2023, Video Games")
para("Team: [Team member names]   Enrollment: [Enrollment numbers]   Guide: [Guide name]")

H("1. Source")
para("Amazon Reviews 2023, collected by the McAuley Lab (UC San Diego) and hosted on Hugging Face "
     "(McAuley-Lab/Amazon-Reviews-2023). We use one category, Video_Games, and two kinds of files: the benchmark "
     "5-core interaction files split by timestamp (train / valid / test CSV) and the raw item metadata file "
     "(meta_Video_Games.jsonl). 'Core' means every user and every item in the full data has at least five interactions.")

H("2. Schema")
table(["File", "Field", "Meaning"], [
    ("interactions CSV", "user_id", "anonymised reviewer ID (string)"),
    ("", "parent_asin", "product ID; variants of one product share it"),
    ("", "rating", "1.0 - 5.0 stars"),
    ("", "timestamp", "review time, Unix milliseconds"),
    ("metadata JSONL", "parent_asin, title", "product ID and name"),
    ("", "categories", "list: Video Games > platform > type > ..."),
    ("", "price", "USD, missing for many items")])
para("Other metadata fields (images, description, store, details) exist but are not used by the model.")

H("3. Raw data size")
table(["Split", "Rows", "Users", "Products", "Mean rating", "Period"],
      [(s.capitalize(), f"{len(x):,}", f"{x.user_id.nunique():,}", f"{x.parent_asin.nunique():,}", f"{x.rating.mean():.2f}",
        f"{ym(x.timestamp.min())} - {ym(x.timestamp.max())}") for s, x in raw.items()])
para("The authors split by global time, so every test review is later than every training review. The earliest "
     "training reviews are from around 2000. Review volume is concentrated in recent years.")
fig_(OUT + "ds_years.png", "Figure 1. Reviews per year.", 5.0)
fig_(OUT + "ds_ratings.png", "Figure 2. Ratings are strongly skewed positive.", 4.4)
para(f"{(raw['train'].rating>=4).mean()*100:.0f}% of training reviews are rated 4 or 5. We therefore treat rating >= 4 "
     "as a positive implicit interaction and discard lower ratings rather than turning them into negatives.")

H("4. Preprocessing and the kept subset")
steps = [("Raw 5-core train reviews", len(raw["train"]), raw["train"].user_id.nunique(), raw["train"].parent_asin.nunique())]
pt = raw["train"][raw["train"].rating >= 4]
steps.append(("Keep positives (rating >= 4)", len(pt), pt.user_id.nunique(), pt.parent_asin.nunique()))
steps.append(("Subsample users + min-degree filter (final train)", len(tr), nu, ni))
table(["Step", "Interactions", "Users", "Products"], [(a, f"{b:,}", f"{c:,}", f"{e:,}") for a, b, c, e in steps])
para("Subsampling keeps training on a laptop CPU to about two minutes. Users with activity in the validation or test "
     "periods are kept first (otherwise almost no sampled user would have future data to evaluate on), then random "
     "users fill up to 30,000; afterwards users with fewer than 3 and products with fewer than 5 positive training "
     "interactions are removed repeatedly until stable. Validation and test rows are then restricted to the kept users "
     "and products. Selection depends on which users appear later, but only for choosing who is modelled; no later "
     "interaction enters the graph or training. The kept users are somewhat more active than the full population.")
table(["Split", "Positive interactions", "Users"], [(n, f"{len(x):,}", f"{x.user.nunique():,}") for n, x in (("Train", tr), ("Validation", va), ("Test", te))])
dens = len(tr) / (nu * ni)
para(f"Final graph (training edges only): {nu:,} customers, {ni:,} products, {len(tr):,} edges, density {dens:.5f} "
     f"({dens*100:.3f}% of possible pairs).")

H("5. Distributions")
ud, idg = tr.groupby("user").size(), tr.groupby("item").size()
table(["", "Mean", "Median", "Max", "Min"], [
    ("Interactions per customer", f"{ud.mean():.1f}", int(ud.median()), int(ud.max()), int(ud.min())),
    ("Interactions per product", f"{idg.mean():.1f}", int(idg.median()), int(idg.max()), int(idg.min()))])
top10 = idg.sort_values(ascending=False).head(max(1, ni // 100)).sum() / len(tr)
fig_(OUT + "ds_degrees.png", "Figure 3. Both degree distributions are long-tailed.", 6.2)
para(f"The most popular 1% of products ({max(1, ni//100)} items) account for {top10*100:.1f}% of training edges, a moderate "
     "concentration; most interactions are spread over the long tail. A popularity ranking is still a reasonable "
     "baseline, while the many low-degree customers make per-user embeddings hard to learn without graph smoothing.")

H("6. Item metadata quality")
has_title = (items.title != items.asin).mean() * 100
has_price = items.price.notna().mean() * 100
has_cat = (items.category.fillna("") != "").mean() * 100
table(["Field", "Coverage among kept products"], [("title", f"{has_title:.1f}%"), ("category path", f"{has_cat:.1f}%"), ("price", f"{has_price:.1f}%")])
para("Items without a metadata record keep their ID as the title. Price is often missing, so it is shown in the demo "
     "but not used as a model feature.")
fig_(OUT + "ds_categories.png", "Figure 4. Leading top-level categories (platforms) among kept products.", 5.2)

H("7. Data quality, bias and ethics")
for t in ["Reviews are not purchases: a review is a proxy for a purchase, and only people who chose to review appear.",
          "Positive skew: most ratings are 4-5, so ratings carry little negative signal; we use only positives.",
          "Popularity bias: a few products dominate interactions, which recommenders tend to amplify.",
          "Subsample bias: our users are more active than average; results may not carry to the whole population.",
          "Privacy: user IDs are anonymised hashes; no personal text is used by the model. The dataset is public for research use; check its terms before any commercial use."]:
    d.add_paragraph(t, style="List Bullet")

H("8. Reproduction")
para("Run `python -m src.data` to download the files from Hugging Face and regenerate data/processed/ "
     "(train.csv, val.csv, test.csv with columns user, item, timestamp; users.csv and items.csv are ID maps, the latter with metadata). "
     "Random seed 42.")
H("9. Citation")
para("Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. Bridging Language and Items for Retrieval and Recommendation. arXiv 2024 "
     "(Amazon Reviews 2023).")
d.save("report/Dataset_Report.docx"); print("saved")
