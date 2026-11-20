"""
Build every chart and GIF used in the README.

All numbers come from the recorded notebook outputs
(dialogsum_lora_finetuning.ipynb) and from fine_tuned_error_analysis.csv.

Run from the repository root:

    python scripts/make_figures.py

Output goes to ./assets/
"""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Recorded results (copied from notebook outputs)
# ---------------------------------------------------------------------------
METRICS = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
SCORES = {
    "Base model": [0.0784, 0.0146, 0.0689],
    "LoRA r=4": [0.2625, 0.0788, 0.2238],
    "LoRA r=16": [0.3131, 0.1027, 0.2599],
}

# Training loss logged every 50 optimizer steps
LOSS_R4 = [20.601030, 18.010392, 15.096062, 14.285001, 13.917115,
           13.657555, 13.763347, 13.529440, 13.594231, 13.075264]
LOSS_R16 = [10.272559, 7.595305, 7.174207, 6.901099, 6.621743, 6.421418,
            6.483375, 6.403085, 6.306946, 6.543156, 6.427615, 6.175931,
            6.441552, 6.459424, 6.300762, 6.252971, 6.434138, 6.400040,
            6.167140, 5.944777]
LOSS_FP16 = [1422179.2, 14374993.92, 92290385.92, 34889743.36, 60792345.6,
             101929359.36, 16532766.72, 45328486.4, 54370892.8, 88542771.2]

TOTAL_PARAMS = 76_961_152          # FLAN-T5-small
TRAINABLE_R4 = 172_032             # printed by print_trainable_parameters()
TRAINABLE_R16 = TRAINABLE_R4 * 4   # LoRA params scale linearly with rank

# ---------------------------------------------------------------------------
# Theme (validated palette: first three categorical slots)
# ---------------------------------------------------------------------------
THEMES = {
    "light": dict(
        surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781",
        grid="#e1e0d9", frozen="#d9d7cf",
        series=["#2a78d6", "#eb6834", "#1baf7a"],
    ),
    "dark": dict(
        surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", frozen="#3a3a37",
        series=["#3987e5", "#d95926", "#199e70"],
    ),
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "savefig.dpi": 160,
})


def style_axes(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    ax.tick_params(colors=t["text2"], length=0)
    ax.spines["bottom"].set_color(t["grid"])
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(t["text2"])


def new_fig(t, w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(t["surface"])
    style_axes(ax, t)
    return fig, ax


def title(fig, t, main, sub=None):
    fig.text(0.02, 0.96, main, fontsize=15, fontweight="bold",
             color=t["text"], va="top")
    if sub:
        fig.text(0.02, 0.885, sub, fontsize=10.5, color=t["text2"], va="top")


def save(fig, name, mode):
    path = ASSETS / f"{name}_{mode}.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", path.relative_to(ROOT))


def rounded_bar(ax, x, y0, width, height, color, horizontal=False):
    """Bar with a small rounded data-end (square-ish at the baseline)."""
    ax.bar(x, height, width, bottom=y0, color=color) if not horizontal else \
        ax.barh(x, height, width, left=y0, color=color)


# ---------------------------------------------------------------------------
# 1. ROUGE comparison (grouped bars)
# ---------------------------------------------------------------------------
def fig_rouge(mode):
    t = THEMES[mode]
    fig, ax = new_fig(t, 9, 5)
    fig.subplots_adjust(top=0.72, bottom=0.1, left=0.07, right=0.98)
    title(fig, t, "Fine-tuning made the scores 3-7x higher",
          "ROUGE F1 on 499 test dialogues (higher = closer to the human summary)")
    x = np.arange(len(METRICS))
    w = 0.24
    for i, (name, vals) in enumerate(SCORES.items()):
        xs = x + (i - 1) * (w + 0.02)
        ax.bar(xs, vals, w, color=t["series"][i], label=name)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.006, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9.5, color=t["text2"])
    ax.set_xticks(x, METRICS)
    ax.set_ylim(0, 0.36)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=3,
                    frameon=False, fontsize=10.5, handlelength=1, handleheight=1)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    save(fig, "rouge_comparison", mode)


# ---------------------------------------------------------------------------
# 2. Training loss curves (stable r=4 vs r=16)
# ---------------------------------------------------------------------------
def fig_loss(mode):
    t = THEMES[mode]
    fig, ax = new_fig(t, 9, 4.8)
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.07, right=0.88)
    title(fig, t, "Training loss went down for both runs",
          "Loss logged every 50 steps (lower = the model is fitting the data better)")
    s4 = np.arange(1, len(LOSS_R4) + 1) * 50
    s16 = np.arange(1, len(LOSS_R16) + 1) * 50
    for steps, loss, c, name in [(s4, LOSS_R4, t["series"][1], "LoRA r=4"),
                                 (s16, LOSS_R16, t["series"][2], "LoRA r=16")]:
        ax.plot(steps, loss, color=c, lw=2, solid_capstyle="round", label=name)
        ax.scatter(steps[-1], loss[-1], s=60, color=c, zorder=3,
                   edgecolor=t["surface"], linewidth=2)
        ax.text(steps[-1] + 18, loss[-1], f"{name}\n{loss[-1]:.2f}",
                va="center", fontsize=10, color=t["text"])
    ax.set_xlabel("Training step", color=t["text2"])
    ax.set_ylim(0, 22)
    ax.set_xlim(0, 1050)
    leg = ax.legend(loc="upper right", frameon=False, ncol=2,
                    bbox_to_anchor=(1.0, 1.12))
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    save(fig, "training_loss", mode)


# ---------------------------------------------------------------------------
# 3. The failed FP16 run
# ---------------------------------------------------------------------------
def fig_fp16(mode):
    t = THEMES[mode]
    fig, ax = new_fig(t, 9, 4.2)
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.1, right=0.97)
    title(fig, t, "The first try (half precision) blew up",
          "FP16 + learning rate 2e-4: loss jumped into the millions. Switching to FP32 + 1e-4 fixed it.")
    steps = np.arange(1, len(LOSS_FP16) + 1) * 50
    ax.plot(steps, LOSS_FP16, color=t["series"][1], lw=2, marker="o", ms=5,
            label="FP16 run (broken)")
    ax.plot(steps, LOSS_R4, color=t["series"][2], lw=2, marker="o", ms=5,
            label="FP32 run (stable)")
    ax.set_yscale("log")
    ax.set_ylabel("Loss (log scale)", color=t["text2"])
    ax.set_xlabel("Training step", color=t["text2"])
    ax.text(steps[-1], LOSS_FP16[-1] * 2.2, "~88,000,000", ha="right",
            fontsize=10, color=t["text"])
    ax.text(steps[-1], LOSS_R4[-1] * 3, "13.1", ha="right", fontsize=10,
            color=t["text"])
    leg = ax.legend(loc="center right", frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    save(fig, "fp16_vs_fp32", mode)


# ---------------------------------------------------------------------------
# 4. Trainable parameters
# ---------------------------------------------------------------------------
def fig_params(mode):
    t = THEMES[mode]
    fig, ax = new_fig(t, 9, 3.6)
    style_axes(ax, t, grid_axis="x")
    fig.subplots_adjust(top=0.72, bottom=0.12, left=0.2, right=0.9)
    title(fig, t, "LoRA trains less than 1% of the model",
          "Number of weights that are updated during training")
    rows = [("Full fine-tuning", TOTAL_PARAMS, t["frozen"]),
            ("LoRA r=16", TRAINABLE_R16, t["series"][2]),
            ("LoRA r=4", TRAINABLE_R4, t["series"][1])]
    for i, (name, n, c) in enumerate(rows):
        ax.barh(i, n, 0.55, color=c)
        pct = n / TOTAL_PARAMS * 100
        ax.text(n + TOTAL_PARAMS * 0.012, i,
                f"{n/1e6:.2f}M  ({pct:.2f}%)" if n < TOTAL_PARAMS else "76.96M  (100%)",
                va="center", fontsize=10.5, color=t["text"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, TOTAL_PARAMS * 1.18)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    ax.spines["bottom"].set_visible(False)
    save(fig, "trainable_params", mode)


# ---------------------------------------------------------------------------
# 5. How LoRA works (diagram)
# ---------------------------------------------------------------------------
def fig_lora_diagram(mode):
    t = THEMES[mode]
    fig = plt.figure(figsize=(9, 4.4))
    fig.patch.set_facecolor(t["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, color, label, sub=None, txt=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=color, ec="none"))
        ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), label, ha="center",
                va="center", fontsize=13, fontweight="bold", color=txt or t["text"])
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
                    fontsize=9.5, color=txt or t["text2"])

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=14, color=t["muted"], lw=1.5))

    ax.text(2, 47, "How LoRA works", fontsize=15, fontweight="bold",
            color=t["text"], va="top")
    ax.text(2, 42.5, "The big original weights stay frozen. Two tiny matrices learn the new skill.",
            fontsize=10.5, color=t["text2"], va="top")

    ax.text(9, 22, "Input\nx", ha="center", va="center", fontsize=12, color=t["text"])
    arrow(13, 22, 21, 29)
    arrow(13, 22, 21, 13)

    box(21, 22, 30, 14, t["frozen"], "W  (frozen)", "original FLAN-T5 weight\n512 x 384 numbers")
    box(21, 5, 13, 12, t["series"][1], "A", "512 x r", txt="#ffffff")
    box(38, 5, 13, 12, t["series"][1], "B", "r x 384", txt="#ffffff")
    arrow(34, 11, 38, 11)

    ax.text(67, 22, "+", ha="center", va="center", fontsize=26, color=t["text"])
    arrow(51, 29, 63.5, 23.5)
    arrow(51, 11, 63.5, 20.5)
    arrow(70.5, 22, 79, 22)
    ax.text(88, 22, "Output\nWx + BAx", ha="center", va="center", fontsize=12,
            color=t["text"])

    ax.text(36, 1.2, "only these are trained  (r = 4 or 16)", ha="center",
            fontsize=9.5, color=t["text2"], style="italic")
    save(fig, "lora_diagram", mode)


# ---------------------------------------------------------------------------
# 6. Error analysis (from the CSV)
# ---------------------------------------------------------------------------
def load_csv():
    return pd.read_csv(ROOT / "fine_tuned_error_analysis.csv")


def fig_errors(df, mode):
    t = THEMES[mode]
    fig, ax = new_fig(t, 9, 3.8)
    style_axes(ax, t, grid_axis="x")
    fig.subplots_adjust(top=0.7, bottom=0.12, left=0.27, right=0.9)
    n = len(df)
    title(fig, t, "About 1 in 5 summaries has an obvious problem",
          f"Automatic checks on the r=4 model's {n} test summaries (one summary can have several problems)")
    good = int((~(df.too_short | df.too_long | df.repeated_sentence)).sum())
    rows = [("No flag raised", good),
            ("Repeats a sentence", int(df.repeated_sentence.sum())),
            ("Too long", int(df.too_long.sum())),
            ("Too short", int(df.too_short.sum()))]
    for i, (name, c) in enumerate(rows):
        color = t["series"][2] if i == 0 else t["series"][1]
        ax.barh(i, c, 0.55, color=color)
        ax.text(c + 6, i, f"{c}  ({c/n:.0%})", va="center", fontsize=10.5,
                color=t["text"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, n * 0.95)
    ax.spines["bottom"].set_visible(False)
    save(fig, "error_breakdown", mode)


def fig_rouge_hist(df, mode):
    t = THEMES[mode]
    fig, ax = new_fig(t, 9, 4.2)
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.08, right=0.97)
    title(fig, t, "How good is each single summary?",
          "ROUGE-L score of every test summary from the r=4 model (0 = no overlap, 1 = perfect)")
    bins = np.arange(0, 0.8, 0.05)
    ax.hist(df.rougeL, bins=bins, color=t["series"][0], rwidth=0.9)
    med = df.rougeL.median()
    ax.axvline(med, color=t["text2"], lw=1.2)
    ax.text(med + 0.008, ax.get_ylim()[1] * 0.92, f"median {med:.2f}",
            color=t["text"], fontsize=10)
    ax.set_xlabel("ROUGE-L", color=t["text2"])
    ax.set_ylabel("Number of summaries", color=t["text2"])
    save(fig, "rougeL_distribution", mode)


def fig_lengths(df, mode):
    t = THEMES[mode]
    fig, ax = new_fig(t, 9, 4.8)
    style_axes(ax, t, grid_axis="both")
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.08, right=0.97)
    title(fig, t, "Summary length: model vs human",
          "Each dot is one test example. Dots above the line = the model wrote more words than the human.")
    ax.scatter(df.reference_words, df.prediction_words, s=22,
               color=t["series"][0], alpha=0.55, edgecolor="none")
    lim = 120
    ax.plot([0, lim], [0, lim], color=t["muted"], lw=1)
    ax.text(60, 69, "same length", rotation=0, color=t["text2"], fontsize=9.5)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Human summary (words)", color=t["text2"])
    ax.set_ylabel("Model summary (words)", color=t["text2"])
    save(fig, "summary_lengths", mode)


# ---------------------------------------------------------------------------
# GIFs (light theme only)
# ---------------------------------------------------------------------------
FONT_DIR = Path("C:/Windows/Fonts")


def font(size, bold=False):
    names = ["segoeuib.ttf" if bold else "segoeui.ttf",
             "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for n in names:
        for base in [FONT_DIR, Path(matplotlib.get_data_path()) / "fonts/ttf"]:
            p = base / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def save_gif(frames, durations, name):
    path = ASSETS / name
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    print("saved", path.relative_to(ROOT), f"({path.stat().st_size/1024:.0f} KB)")


DEMO_EXAMPLES = [
    dict(
        turns=[("Person1", "Can I help you?"),
               ("Person2", "I'd like to buy a new mobile phone please."),
               ("Person1", "Ok, would you like a phone with camera and MP3 player?"),
               ("Person2", "Yes please. And I'd like to be able to make video calls too.")],
        pred="#Person2# wants to buy a new mobile phone with camera and MP3 player.",
        ref="#Person2# wants to buy a new mobile phone from #Person1#.",
    ),
    dict(
        note="shortened to fit",
        turns=[("Person1", "I'm going to New York for the first time, but I don't have a tour guide. Can you give me any suggestions?"),
               ("Person2", "There's a service called 'A friend in New York'. It's a personal tour guide service."),
               ("Person1", "Good. Where can I get the questionnaire?"),
               ("Person2", "You can easily download it from their website.")],
        pred="#Person1# is going to New York for the first time. #Person2# gives #Person1# advice on a personal tour guide service. #Person1# will download the questionnaire from their website.",
        ref="#Person1# is going to New York for the first time. #Person2# suggests #Person1# use a personal tour guide service to make #Person1#'s trip plan.",
    ),
]


def make_demo_gif():
    t = THEMES["light"]
    W, H = 960, 540
    f_title = font(22, True)
    f_body = font(16)
    f_small = font(14)
    f_bold = font(16, True)
    blue, orange, aqua = t["series"]

    def wrap(txt, width):
        return textwrap.wrap(txt, width)

    def base_frame(step_label):
        im = Image.new("RGB", (W, H), t["surface"])
        d = ImageDraw.Draw(im)
        d.text((28, 20), "DialogSum LoRA  -  live demo", font=f_title, fill=t["text"])
        d.text((28, 52), step_label, font=f_small, fill=t["text2"])
        d.line([(28, 80), (W - 28, 80)], fill=t["grid"], width=1)
        # model chip in the middle
        return im, d

    def draw_chat(d, turns, n_shown):
        y = 98
        for who, txt in turns[:n_shown]:
            lines = wrap(txt, 34)
            h = 22 * len(lines) + 16
            left = who == "Person1"
            x0 = 28 if left else 150
            text_w = max(d.textlength(ln, font=f_body) for ln in lines)
            x1 = x0 + text_w + 24
            color = "#e6effa" if left else "#fde8df"
            d.rounded_rectangle([x0, y, x1, y + h], radius=12, fill=color)
            for i, ln in enumerate(lines):
                d.text((x0 + 12, y + 8 + 22 * i), ln, font=f_body, fill=t["text"])
            d.text((x0 + 4, y + h + 1), f"#{who}#", font=font(11), fill=t["muted"])
            y += h + 20

    def draw_model(d, active):
        x0, y0, x1, y1 = 488, 220, 612, 300
        d.rounded_rectangle([x0, y0, x1, y1], radius=14,
                            fill=blue if active else t["frozen"])
        col = "#ffffff" if active else t["text2"]
        d.text((x0 + 62, y0 + 26), "FLAN-T5", font=f_bold, fill=col, anchor="mm")
        d.text((x0 + 62, y0 + 52), "LoRA r=4", font=f_bold, fill=col, anchor="mm")
        # arrows
        d.line([(462, 260), (486, 260)], fill=t["muted"], width=2)
        d.polygon([(486, 254), (486, 266), (494, 260)], fill=t["muted"])
        d.line([(614, 260), (636, 260)], fill=t["muted"], width=2)
        d.polygon([(636, 254), (636, 266), (644, 260)], fill=t["muted"])

    def draw_summary(d, text, chars, ref=None):
        x0, y0, x1 = 650, 98, W - 28
        d.text((x0, y0), "Model summary", font=f_bold, fill=t["text"])
        shown = text[:chars]
        lines = wrap(shown, 30)
        box_h = max(60, 24 * len(wrap(text, 30)) + 20)
        d.rounded_rectangle([x0, y0 + 28, x1, y0 + 28 + box_h], radius=12,
                            outline=aqua, width=2, fill="#ffffff")
        for i, ln in enumerate(lines):
            d.text((x0 + 12, y0 + 38 + 24 * i), ln, font=f_body, fill=t["text"])
        if ref:
            ry = y0 + 28 + box_h + 24
            d.text((x0, ry), "Human summary (for comparison)", font=f_small, fill=t["text2"])
            for i, ln in enumerate(wrap(ref, 34)):
                d.text((x0, ry + 24 + 21 * i), ln, font=f_small, fill=t["text2"])

    frames, durs = [], []
    for ex_i, ex in enumerate(DEMO_EXAMPLES, start=1):
        label = f"Example {ex_i} of 2: real test-set chat" + \
            (f" ({ex['note']})" if ex.get("note") else "")
        # 1. chat appears turn by turn
        for n in range(1, len(ex["turns"]) + 1):
            im, d = base_frame(label + "   |   step 1: read the dialogue")
            draw_chat(d, ex["turns"], n)
            draw_model(d, False)
            frames.append(im); durs.append(700)
        # 2. model "thinks"
        for k in range(4):
            im, d = base_frame(label + "   |   step 2: the fine-tuned model reads it")
            draw_chat(d, ex["turns"], len(ex["turns"]))
            draw_model(d, k % 2 == 0)
            frames.append(im); durs.append(300)
        # 3. summary types out
        txt = ex["pred"]
        for c in list(range(3, len(txt), 4)) + [len(txt)]:
            im, d = base_frame(label + "   |   step 3: it writes a short summary")
            draw_chat(d, ex["turns"], len(ex["turns"]))
            draw_model(d, True)
            draw_summary(d, txt, c)
            frames.append(im); durs.append(45)
        # 4. hold with reference
        im, d = base_frame(label + "   |   done: compare with the human summary")
        draw_chat(d, ex["turns"], len(ex["turns"]))
        draw_model(d, True)
        draw_summary(d, txt, len(txt), ref=ex["ref"])
        frames.append(im); durs.append(3200)
    save_gif(frames, durs, "demo.gif")


def make_rouge_gif():
    """Bars grow from base-model height to fine-tuned height."""
    t = THEMES["light"]
    frames, durs = [], []
    names = list(SCORES)
    n_steps = 18
    for phase, (start, end, label) in enumerate([
        (names[0], names[0], "Before fine-tuning (base FLAN-T5-small)"),
        (names[0], names[1], "After LoRA fine-tuning, rank r=4"),
        (names[1], names[2], "After LoRA fine-tuning, rank r=16"),
    ]):
        steps = [1.0] if phase == 0 else np.linspace(0, 1, n_steps)
        for k, a in enumerate(steps):
            a = 1 - (1 - a) ** 3  # ease-out
            vals = [s + (e - s) * a for s, e in zip(SCORES[start], SCORES[end])]
            fig, ax = new_fig(t, 8, 4.5)
            fig.subplots_adjust(top=0.78, bottom=0.1, left=0.08, right=0.97)
            title(fig, t, "ROUGE score on 499 test dialogues", label)
            color = t["series"][[0, 1, 2][phase]]
            ax.bar(range(3), vals, 0.45, color=color)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=12,
                        color=t["text"])
            ax.set_xticks(range(3), METRICS)
            ax.set_ylim(0, 0.36)
            fig.canvas.draw()
            im = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                  fig.canvas.buffer_rgba()).convert("RGB")
            plt.close(fig)
            frames.append(im)
            durs.append(1800 if k == len(steps) - 1 else 40)
    save_gif(frames, durs, "rouge_growth.gif")


def make_loss_gif():
    """Loss curves drawing themselves left to right."""
    t = THEMES["light"]
    frames, durs = [], []
    s4 = np.arange(1, len(LOSS_R4) + 1) * 50
    s16 = np.arange(1, len(LOSS_R16) + 1) * 50
    for upto in list(range(50, 1001, 50)):
        fig, ax = new_fig(t, 8, 4.3)
        fig.subplots_adjust(top=0.8, bottom=0.14, left=0.08, right=0.86)
        title(fig, t, "Watching the model learn",
              f"Training loss, step {upto} of 1000 (lower is better)")
        for steps, loss, c, name in [(s4, LOSS_R4, t["series"][1], "r=4"),
                                     (s16, LOSS_R16, t["series"][2], "r=16")]:
            m = steps <= upto
            if m.sum() == 0:
                continue
            ax.plot(steps[m], np.array(loss)[m], color=c, lw=2.2)
            ax.scatter(steps[m][-1], np.array(loss)[m][-1], s=50, color=c,
                       zorder=3, edgecolor=t["surface"], linewidth=2)
            ax.text(steps[m][-1] + 20, np.array(loss)[m][-1],
                    f"{name}: {np.array(loss)[m][-1]:.1f}", va="center",
                    fontsize=10.5, color=t["text"])
        ax.set_xlim(0, 1100)
        ax.set_ylim(0, 22)
        ax.set_xlabel("Training step", color=t["text2"])
        fig.canvas.draw()
        im = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                              fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)
        frames.append(im)
        durs.append(2500 if upto == 1000 else 160)
    save_gif(frames, durs, "training_loss.gif")


if __name__ == "__main__":
    df = load_csv()
    for mode in THEMES:
        fig_rouge(mode)
        fig_loss(mode)
        fig_fp16(mode)
        fig_params(mode)
        fig_lora_diagram(mode)
        fig_errors(df, mode)
        fig_rouge_hist(df, mode)
        fig_lengths(df, mode)
    make_demo_gif()
    make_rouge_gif()
    make_loss_gif()
