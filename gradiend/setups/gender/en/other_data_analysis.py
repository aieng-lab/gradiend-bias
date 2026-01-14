import os.path

from gradiend.model import ModelWithGradiend
from gradiend.setups.gender.en import GenderEnSetup
from gradiend.setups.gender.en.other_data import TOKEN_GROUPS
from gradiend.export import models as pretty_models
from gradiend.util import init_matplotlib

models = [
    'bert-base-cased',
    'bert-large-cased',
    'distilbert-base-cased',
    'roberta-large',
    'gpt2',
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Llama-3.2-3B-Instruct',
]
setup = GenderEnSetup()


def create_data(models):
    all_metrics = []
    output = f'results/cache/other_data_analysis/{"_".join(sorted(models))}.csv'
    if os.path.isfile(output):
        return pd.read_csv(output)

    data = []
    for model_name in models:
        print(f'Processing {model_name}')
        model_with_gradiend = ModelWithGradiend.from_pretrained(f'results/models/{model_name}')
        analysis = setup.analyze_models_with_other_data(model_with_gradiend, tokens=list(TOKEN_GROUPS.keys()))
        #metrics = setup.get_other_model_metrics(analysis)

        analysis['model'] = model_name
        data.append(analysis)

        # Flatten metrics into a dict for table
        #row = {"model": model_name} #, "overall_pearson": metrics["overall_pearson"]}

        # Per token correlations
        #for token, sub_fg in analysis.groupby('token'):
        #    row['token'] = token
        #    row['encoded'] = sub_fg['encoded'].mean()
        #    all_metrics.append(row.copy())

        # Per group correlations
        #for group, val in metrics["per_group"].items():
        #    row[f"group:{group}"] = val


    # Create DataFrame
    #metrics_df = pd.DataFrame(all_metrics)
    metrics_df = pd.concat(data)

    # Pretty print
    #print(tabulate(metrics_df, headers='keys', tablefmt='grid', showindex=False))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    metrics_df.to_csv(output)
    return  metrics_df

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_single_token_pair_violins(df_all, token_groups=TOKEN_GROUPS):
    init_matplotlib(use_tex=True)
    for model, df in df_all.groupby("model"):
        # --- Build pair groups ---
        pairs = {}
        for tok, grp in token_groups.items():
            pairs.setdefault(grp, []).append(tok)

        # filter df to contain only tokens in TOKEN_GROUPS
        df = df[df["token"].isin(token_groups.keys())]

        # Only keep groups with exactly 2 tokens (male/female style)
        valid_pairs = {g: tks for g, tks in pairs.items() if len(tks) == 2}

        # --- Construct plotting dataframe ---
        rows = []
        for group, (tok_a, tok_b) in valid_pairs.items():
            df_a = df[df["token"] == tok_a]
            df_b = df[df["token"] == tok_b]

            group_name = f"{tok_b}/{tok_a}"

            rows.append(pd.DataFrame({
                "group": group_name,
                "token": tok_b,
                "encoded": df_b["encoded"],
                "Gender": "Female"
            }))

            rows.append(pd.DataFrame({
                "group": group_name,
                "token": tok_a,
                "encoded": df_a["encoded"],
                "Gender": "Male"
            }))

        plot_df = pd.concat(rows, ignore_index=True)

        # --- Plot ---
        plt.figure(figsize=(8, 2))

        sns.violinplot(
            data=plot_df,
            x="group",
            y="encoded",
            hue="Gender",
            split=True,
            inner="quartile",
            linewidth=0.7,
            density_norm="width",
            zorder=5,
            cut=True,
            palette="Paired"
        )

        plt.grid(zorder=0)

        y_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        plt.yticks(y_ticks)
        plt.xlabel("Token Pair")
        plt.ylabel("$h$")
        pretty_model = pretty_models[model.split('/')[-1]]
        plt.title(f"{pretty_model}")
        # make

        #plt.xticks(rotation=45)
        plt.tight_layout()
        output = f'img/other_data_analysis/violin_{model.replace("/", "_")}.pdf'
        os.makedirs(os.path.dirname(output), exist_ok=True)
        plt.savefig(output)
        plt.show()

def plot_token_pair_violins(df_all, token_groups=TOKEN_GROUPS):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    init_matplotlib(use_tex=True)

    models = df_all["model"].unique()
    n_models = len(models)

    # --- Prepare plotting data for all models ---
    plot_data = {}
    for model, df in df_all.groupby("model"):
        # --- Build pair groups ---
        pairs = {}
        for tok, grp in token_groups.items():
            pairs.setdefault(grp, []).append(tok)

        # filter df to contain only tokens in TOKEN_GROUPS
        df = df[df["token"].isin(token_groups.keys())]

        # Only keep groups with exactly 2 tokens (male/female style)
        valid_pairs = {g: tks for g, tks in pairs.items() if len(tks) == 2}

        # --- Construct plotting dataframe ---
        rows = []
        for group, (tok_a, tok_b) in valid_pairs.items():
            df_a = df[df["token"] == tok_a]
            df_b = df[df["token"] == tok_b]

            group_name = f"{tok_b}/{tok_a}"

            rows.append(pd.DataFrame({
                "group": group_name,
                "token": tok_b,
                "encoded": df_b["encoded"],
                "Gender": "Female"
            }))

            rows.append(pd.DataFrame({
                "group": group_name,
                "token": tok_a,
                "encoded": df_a["encoded"],
                "Gender": "Male"
            }))

        plot_data[model] = pd.concat(rows, ignore_index=True)

    # --- Create subplots ---
    fig, axes = plt.subplots(n_models, 1, figsize=(8, 2 * n_models), sharex=True)

    # If only one model, axes is not a list
    if n_models == 1:
        axes = [axes]

    # Plot each model
    for ax, model in zip(axes, models):
        sns.violinplot(
            data=plot_data[model],
            x="group",
            y="encoded",
            hue="Gender",
            split=True,
            inner="quartile",
            linewidth=0.7,
            density_norm="width",
            zorder=5,
            cut=True,
            palette="Paired",
            ax=ax
        )
        ax.grid(zorder=0)
        ax.set_ylabel("$h$")
        pretty_model = pretty_models[model.split('/')[-1]]
        ax.set_title(pretty_model)

    axes[-1].set_xlabel("Token Pair")

    # --- Single legend on top ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Gender",
        loc="upper center",
        ncol=3,  # flat layout: title + two entries in one row
        frameon=False
    )
    # Remove individual legends
    for ax in axes:
        ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space on top for legend
    output = 'img/other_data_analysis/violin_all_models.pdf'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output)
    plt.show()

def plot_token_pair_violins(df_all, token_groups=TOKEN_GROUPS, add_combined=False, version='', y_legend_offset=0, font_size=12):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    import math

    init_matplotlib(use_tex=True)

    # -------------------------
    # Collect per-model data
    # -------------------------
    models = list(df_all["model"].unique())

    # Add combined plot?
    if add_combined:
        models = models + ["ALL"]

    plot_data = {}
    for model in models:
        if model == "ALL":
            df = df_all.copy()
        else:
            df = df_all[df_all["model"] == model].copy()

        # Build pair groups
        pairs = {}
        for tok, grp in token_groups.items():
            pairs.setdefault(grp, []).append(tok)

        df = df[df["token"].isin(token_groups.keys())]

        valid_pairs = {g: tks for g, tks in pairs.items() if len(tks) == 2}

        rows = []
        for group, (tok_a, tok_b) in valid_pairs.items():
            df_a = df[df["token"] == tok_a]
            df_b = df[df["token"] == tok_b]
            group_name = f"{tok_b}/{tok_a}"

            rows.append(pd.DataFrame({
                "group": group_name,
                "token": tok_b,
                "encoded": df_b["encoded"],
                "Gender": "Female"
            }))
            rows.append(pd.DataFrame({
                "group": group_name,
                "token": tok_a,
                "encoded": df_a["encoded"],
                "Gender": "Male"
            }))

        plot_data[model] = pd.concat(rows, ignore_index=True)

    # -------------------------
    # Create 2×4 grid
    # -------------------------
    n_models = len(models)
    n_cols = 1
    n_rows = len(models)  # fixed

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 2 * n_rows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()


    # -------------------------
    # Plot
    # -------------------------
    for ax, model in zip(axes[:n_models], models):
        sns.violinplot(
            data=plot_data[model],
            x="group",
            y="encoded",
            hue="Gender",
            split=True,
            inner="quartile",
            linewidth=0.7,
            density_norm="width",
            zorder=5,
            #cut=True,
            palette="Paired",
            ax=ax
        )
        ax.grid(zorder=0)
        ax.set_ylabel("$h$")
        ax.set_xlabel("Token")

        if model != "ALL":
            pretty_model = pretty_models[model.split('/')[-1]]
        else:
            pretty_model = "Combined"

        ax.set_title(pretty_model)
        ax.get_legend().remove()
        # rotate x labels
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

        x_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.set_yticks(x_ticks)

        # set font size
        ax.title.set_fontsize(font_size+2)
        ax.xaxis.label.set_fontsize(font_size + 2)
        for label in ax.get_xticklabels():
            label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size + 2)
        for label in ax.get_yticklabels():
            label.set_fontsize(font_size)

    # Hide unused axes
    for ax in axes[n_models:]:
        ax.set_visible(False)

    # Hide x label and x tick labels for all but bottom plots
    for i, ax in enumerate(axes):
        row = i // n_cols  # compute row index
        col = i % n_cols   # compute column index
        if (row < n_rows - 1 and col == 0) or (col == n_cols -1 and row < n_rows - 2):  # not in bottom row
            ax.set_xlabel("")
            ax.set_xticklabels([])

    # -------------------------
    # Single legend on top
    # -------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    import matplotlib.patches as mpatches
    title_patch = mpatches.Patch(color='none', label="Gender")

    fig.legend(
        [title_patch] + handles,
        ["Gender"] + labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.96 + y_legend_offset),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output = f'img/other_data_analysis/violin_all_models_grid{version}.pdf'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output)
    plt.show()



def plot_token_pair_violins_single(df_all, token_groups=TOKEN_GROUPS, version='', y_legend_offset=0):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    import matplotlib.patches as mpatches

    init_matplotlib(use_tex=True)

    models = list(df_all["model"].unique())

    # Build pair groups once
    pairs = {}
    for tok, grp in token_groups.items():
        pairs.setdefault(grp, []).append(tok)
    valid_pairs = {g: tks for g, tks in pairs.items() if len(tks) == 2}

    # --- helper for one model ---
    def build_plot_data(model):
        if model == "ALL":
            df = df_all.copy()
        else:
            df = df_all[df_all["model"] == model].copy()

        df = df[df["token"].isin(token_groups.keys())]
        rows = []

        for group, (tok_a, tok_b) in valid_pairs.items():
            df_a = df[df["token"] == tok_a]
            df_b = df[df["token"] == tok_b]
            group_name = f"{tok_b}/{tok_a}"

            rows.append(pd.DataFrame({
                "group": group_name,
                "token": tok_b,
                "encoded": df_b["encoded"],
                "Gender": "Female"
            }))
            rows.append(pd.DataFrame({
                "group": group_name,
                "token": tok_a,
                "encoded": df_a["encoded"],
                "Gender": "Male"
            }))

        return pd.concat(rows, ignore_index=True)

    # create output directory
    out_dir = f'img/other_data_analysis/single_plots{version}'
    os.makedirs(out_dir, exist_ok=True)

    font_size = 20

    for model in models:
        plot_data = build_plot_data(model)

        pretty = pretty_models.get(model.split("/")[-1], model)

        fig, ax = plt.subplots(figsize=(5.5, 3.8))

        sns.violinplot(
            data=plot_data,
            x="group",
            y="encoded",
            hue="Gender",
            split=True,
            inner="quartile",
            linewidth=0.7,
            density_norm="width",
            palette="Paired",
            ax=ax
        )

        # axis formatting
        ax.grid(zorder=0)
        ax.set_ylabel("$h$", fontsize=font_size + 2)
        ax.set_xlabel("Token", fontsize=font_size + 2)
        ax.set_title(pretty, fontsize=font_size + 2)

        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_fontsize(font_size)

        # build the **single-line legend** above plot
        handles, labels = ax.get_legend_handles_labels()
        title_patch = mpatches.Patch(color='none', label="Gender")

        fig.legend(
            [title_patch] + handles,
            ["Gender"] + labels,
            loc="upper center",
            ncol=3,
            frameon=True,
            bbox_to_anchor=(0.5, 1.07 + y_legend_offset),
        )

        # remove local legend
        ax.get_legend().remove()

        # save
        out = os.path.join(out_dir, f"violin_{model.replace('/', '_')}.pdf")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out)
        plt.show()
        plt.close(fig)

        print(f"Saved: {out}")





if __name__ == '__main__':
    metrics_df = create_data(models=['bert-base-cased', 'distilbert-base-cased'])
    plot_token_pair_violins(metrics_df, add_combined=False, version='bert-base-distilbert', y_legend_offset=0.02, font_size=13)

    metrics_df = create_data(models=['roberta-large', 'gpt2'])
    plot_token_pair_violins(metrics_df, add_combined=False, version='roberta-gpt2', y_legend_offset=0.02, font_size=13)

    exit(1)

    metrics_df = create_data(models=models)
    plot_token_pair_violins_single(metrics_df)

    exit(1)
    metrics_df = create_data(models=['bert-base-cased', 'bert-large-cased', 'distilbert-base-cased', 'roberta-large'])
    plot_token_pair_violins(metrics_df, add_combined=False, version='encoder')

    metrics_df = create_data(models=['gpt2', 'meta-llama/Llama-3.2-3B', 'meta-llama/Llama-3.2-3B-Instruct'])
    plot_token_pair_violins(metrics_df, add_combined=False, version='decoder')
