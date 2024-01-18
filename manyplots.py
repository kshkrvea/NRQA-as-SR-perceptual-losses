import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle


fontsize = 24
figsize = (16, 9)

sns.set_style("darkgrid")
plt.rcParams.update({
    "font.size": fontsize,
    "grid.linewidth": 3,
    "figure.constrained_layout.use": True,
})

palette = sns.color_palette("muted")

less_is_better = {
    'charbonnier': True,
    'lpips_vgg': True,
    'lpips_alex': True,
    'psnr': False,
    'ssim': False,
    'maniqa': False,
    'dists': True,
    'mdtvsfa': False,
    'hyperiqa': False,
    'nima': False,
    'clipiqa': False,
    'pieapp': True,
    'dbcnn': False,
    'paq2piq': False,
}

metric_fullnames = {
    'charbonnier': "Charbonnier Loss",
    'lpips_vgg': "LPIPS (VGG)",
    'lpips_alex': "LPIPS (AlexNet)",
    'psnr': "PSNR",
    'ssim': "SSIM",
    'maniqa': "MANIQA",
    'dists': "DISTS",
    'mdtvsfa': "MDTVSFA",
    'hyperiqa': "HyperIQA",
    'nima': "NIMA",
    'clipiqa': "CLIP-IQA",
    'pieapp': "PieAPP",
    'dbcnn': "DBCNN",
    'paq2piq': "PaQ-2-PiQ",
}

run_fullnames = {
    'baseline': "baseline",
    'charbonnier': 'Only Charbonnier Loss',
    'mdtvsfa_001': "0.05 MDTVSFA",
    'mdtvsfa_002': "0.005 MDTVSFA",
    'hyperiqa': "0.005 HyperIQA",
    'lpips_001': '0.05 LPIPS (VGG)',
    'lpips_000': '0.05 LPIPS (VGG) (From Start)',
    'maniqa_000': '0.005 MANIQA (Pseudo FR)',
    'maniqa_001': '0.005 MANIQA',
    'nima_001': "0.005 NIMA",
    'clipiqa_001': "0.005 CLIP-IQA",
    'pieapp_001': "0.005 PieAPP",
    'dbcnn_001': "0.005 DBCNN",
    'paq2piq_001': "0.00005 PaQ-2-PiQ",
    'lpips_nima_001': "0.05 LPIPS (VGG) + 0.005 NIMA",
}

tuned_metrics = {
    'baseline': [],
    'charbonnier': [],
    'mdtvsfa_001': ['mdtvsfa'],
    'mdtvsfa_002': ['mdtvsfa'],
    'hyperiqa': ['hyperiqa'],
    'lpips_001': ['lpips_vgg'],
    'lpips_000': ['lpips_vgg', ],
    'maniqa_000': ['maniqa'],
    'maniqa_001': ['maniqa'],
    'nima_001': ['nima'],
    'clipiqa_001': ['clipiqa'],
    'pieapp_001': ['pieapp'],
    'dbcnn_001': ['dbcnn'],
    'paq2piq_001': ['paq2piq'],
    'lpips_nima_001': ['lpips_vgg', 'nima'],
}

testset_fullnames = {
    "vimeo": "Vimeo-90K Subset (101 videos)",
    "reds": "REDS (30 videos)",
    "realhero": "RealHero (35 videos)",
}


def drop_runs(df, runs):
    return df[~df.index.isin(runs)]


def drop_metrics(df, metrics):
    return df.drop(columns=metrics)


def rename_runs(df):
    return df.rename(index=run_fullnames)


def rename_metrics(df):
    return df.rename(columns=metric_fullnames)


def read_dataframe(io, testset):
    df = pd.read_excel(io, index_col=0, sheet_name=testset)

    assert set(testset_fullnames.keys()) == set(pd.ExcelFile(io).sheet_names), set(pd.ExcelFile(io).sheet_names) - set(
        testset_fullnames.keys())
    assert set(less_is_better.keys()) == set(df.columns), set(df.columns) - set(less_is_better.keys())
    assert set(metric_fullnames.keys()) == set(df.columns), set(df.columns) - set(metric_fullnames.keys())
    assert set(run_fullnames.keys()) == set(df.index), set(df.index) - set(run_fullnames.keys())
    assert set(tuned_metrics.keys()) == set(df.index), set(df.index) - set(tuned_metrics.keys())

    return df


def compute_relative_gain(df):
    def func(col):
        baseline = col[col.index == "charbonnier"].item()

        return (-1 if less_is_better[col.name] else 1) * (col / baseline - 1) * 100

    return df.apply(func, axis=0).drop("charbonnier")


def average_relative_gain(relative_gain, with_tuned=False):
    if with_tuned:
        mean_relative_gain = pd.Series(name="mean_relative_gain", data=relative_gain.mean(axis=1))
    else:
        mean_relative_gain = pd.Series(name="mean_relative_gain", data={
            run: row[~row.index.isin(tuned_metrics[run])].mean()
            for run, row in relative_gain.iterrows()
        })

    return mean_relative_gain.sort_values()


def add_study_details(ax, testset, with_tuned=None, additional_loss=None):
    invisible_unit = Rectangle(xy=(0, 0), width=1, height=1, fc="w", fill=False, edgecolor="none", linewidth=0)

    text = (
        "$\mathbf{Study\ Details:}$\n" +
        "• BasicVSR++ is trained on Vimeo-90K\n"
        "• Trained 5K iterations with Charbonnier Loss\n"
        f"• Finetuned 15K iterations with {additional_loss if additional_loss else 'Additional Loss'}\n"
        f"• Tested on {testset_fullnames[testset]}\n"
        "• Gain is relative to 20K with Charbonnier Loss"
    )

    if with_tuned:
        text += "\n• Tuned metrics are included in computations"
    elif with_tuned is not None:
        text += "\n• Tuned metrics are excluded from computations"

    ax.legend([invisible_unit], (text,), loc="best", handlelength=0, handletextpad=0, fancybox=True, facecolor="white",
              edgecolor="black", framealpha=1)

    return ax


def plot_relative_gain(relative_gain, testset):
    for run, row in relative_gain.iterrows():
        plt.figure()
        colors = row.apply(lambda value: palette[0] if value > 0 else palette[1])
        title = f"Relative Gain on {testset_fullnames[testset]} for {run}"

        ax = row.plot(
            kind="bar",
            xlabel="Metric",
            ylabel="Relative Gain, %",
            title=title + "\n",
            width=0.9,
            figsize=figsize,
            fontsize=fontsize,
            color=colors,
        )

        ax.bar_label(ax.containers[-1], fmt="{:.1f}%", padding=2)
        ax.margins(y=0.2)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        add_study_details(ax, testset, additional_loss=run)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        os.makedirs("manyplots", exist_ok=True)
        plt.savefig(os.path.join("manyplots", title + ".png"))
        plt.close()


def plot_mean_relative_gain(mean_relative_gain, testset, with_tuned=False):
    plt.figure()
    colors = mean_relative_gain.apply(lambda value: palette[0] if value > 0 else palette[1])
    title = f"Mean Relative Gain on {testset_fullnames[testset]} over {len(stats.columns) - int(not with_tuned)} Metrics"

    ax = mean_relative_gain.plot(
        kind="barh",
        xlabel="Mean Relative Gain, %",
        ylabel="Additional Loss",
        title=title + "\n",
        width=0.9,
        figsize=figsize,
        fontsize=fontsize,
        color=colors,
    )

    ax.bar_label(ax.containers[-1], fmt="{:.1f}%", padding=2)
    ax.margins(x=0.2)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    add_study_details(ax, testset, with_tuned=with_tuned)

    os.makedirs("manyplots", exist_ok=True)
    plt.savefig(os.path.join("manyplots", title + ".png"))
    plt.close()


if __name__ == "__main__":
    for testset in testset_fullnames.keys():
        stats = (
            read_dataframe("stats.xlsx", testset)
            .pipe(drop_runs, runs=["baseline", "maniqa_000", "mdtvsfa_001", "lpips_000", "pieapp_001"])
            .pipe(drop_metrics, metrics=["dists", "charbonnier", "lpips_alex"])
        )
        (
            stats
            .pipe(compute_relative_gain)
            .pipe(average_relative_gain, with_tuned=True)
            .pipe(rename_runs)
            .pipe(plot_mean_relative_gain, testset, with_tuned=True)
        )
        (
            stats
            .pipe(compute_relative_gain)
            .pipe(average_relative_gain, with_tuned=False)
            .pipe(rename_runs)
            .pipe(plot_mean_relative_gain, testset, with_tuned=False)
        )
        (
            stats
            .pipe(compute_relative_gain)
            .pipe(rename_runs)
            .pipe(rename_metrics)
            .pipe(plot_relative_gain, testset)
        )
