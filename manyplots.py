import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


less_is_better = {
    "charbonnier": True,
    "psnr": False,
    "ssim": False,
    "lpips_vgg": True,
    "pieapp": True,
    "dbcnn": False,
    "nima": False,
    "paq2piq": False,
    "hyperiqa": False,
    "mdtvsfa": False,
    "maniqa": False,
    "clipiqa": False,
    "qalign": False,
}

metric_fullnames = {
    "charbonnier": "Charbonnier Loss",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips_vgg": "LPIPS",
    "pieapp": "PieAPP",
    "dbcnn": "DBCNN",
    "nima": "NIMA",
    "paq2piq": "PaQ-2-PiQ",
    "hyperiqa": "HyperIQA",
    "mdtvsfa": "MDTVSFA",
    "maniqa": "MANIQA",
    "clipiqa": "CLIP-IQA",
    "qalign": "Q-Align",
}

run_fullnames = {
    "baseline": "baseline",
    "charbonnier": "Only Charbonnier Loss",
    "mdtvsfa_002": "MDTVSFA",
    "hyperiqa_001": "HyperIQA",
    "lpips_001": "LPIPS",
    "maniqa_001": "MANIQA",
    "nima_001": "NIMA",
    "clipiqa_001": "CLIP-IQA",
    "pieapp_001": "PieAPP",
    "dbcnn_001": "DBCNN",
    "paq2piq_001": "PaQ-2-PiQ",
    "lpips_nima_001": "LPIPS & NIMA",
    "lpips_nima_clipiqa_001": "LPIPS & NIMA & CLIP-IQA",
    "lpips_hyperiqa_001": "LPIPS & HyperIQA",
    "lpips_maniqa_001": "LPIPS & MANIQA",
    "lpips_hyperiqa_pieapp_001": "LPIPS & HyperIQA & PieAPP",
    "qalign_001": "Q-Align",
}


def drop_runs(df, runs):
    return df[~df.index.isin(runs)]


def drop_metrics(df, metrics):
    return df.drop(columns=metrics)


def compute_relative_gain(df):
    def func(col):
        baseline = col[col.index == "charbonnier"].item()

        return (-1 if less_is_better[col.name] else 1) * (col / baseline - 1) * 100

    return df.apply(func, axis=0).drop("charbonnier")


def save_heatmap(df):
    plt.rc("font", size=24)
    plt.figure(figsize=(16, 9), layout="constrained")
    ax = sns.heatmap(df, cmap="vlag", center=0, robust=True, annot=True, fmt="+.0f", cbar=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.xlabel("Relative Gain of IQA Methods, %")
    plt.ylabel("Additional Loss")
    plt.savefig("heatmap.pdf")


if __name__ == "__main__":
    # Read all sheets, drop unnecessary runs and metrics, and compute relative gain
    dfs = []
    for paths in (
        # Each line contains tuple of .xlsx files for *one* VSR method
        ("../mnt/calypso/attacks/basicvsrpp/stats.xlsx",),
        ("../mnt/calypso/attacks/vrt/stats.xlsx",),
        ("../mnt/calypso/attacks/iseebetter/stats.xlsx",),
    ):
        for test_set in ("vimeo", "reds", "realhero"):
            dfs.append(
                pd.concat((pd.read_excel(path, index_col=0, sheet_name=test_set) for path in paths))
                .pipe(drop_runs, runs=["baseline", "maniqa_000", "mdtvsfa_001", "hyperiqa", "lpips_000", "pieapp_002", "pieapp_003"])
                .pipe(drop_metrics, metrics=["charbonnier", "lpips_alex", "dists"])
                .pipe(compute_relative_gain)
            )

    # Average results over methods and test sets
    df = pd.concat(dfs).groupby(pd.concat(dfs).index).mean()

    # Rename known runs and metrics
    df = df.rename(index=run_fullnames, columns=metric_fullnames)

    # Add mean column and sort values by it
    df["Mean"] = df.mean(axis=1)
    df = df.sort_values(by="Mean", ascending=False)

    # Rearrange metrics in publication order
    df = df[["PSNR", "SSIM", "LPIPS", "PieAPP", "DBCNN", "NIMA", "PaQ-2-PiQ", "HyperIQA", "MDTVSFA", "MANIQA", "CLIP-IQA", "Mean"]]

    df.pipe(save_heatmap)
