import hashlib
import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

from gradiend.model import GradiendModel
from gradiend.setups.gender.en import GenderEnSetup
from gradiend.util import init_matplotlib
from gradiend.export import models as pretty_models

def analyze_stability_single_model(setup, model_ids):
    encodeds = {}
    correlations = {}
    correlations_total = {}
    labels = {}
    for model_id in model_ids:
        analysis = setup.analyze_models(model_id)

        # check if encodings need to be inverted
        means = analysis.groupby('state')['encoded'].mean()
        if means['M'] > means['F']:
            print("Inverting encodings for model:", model_id)
            analysis['encoded'] = -analysis['encoded']

        encodeds[model_id] = analysis['encoded'].tolist()
        labels[model_id] = analysis['state'].map(lambda x: x if isinstance(x, str) else 'N').tolist()

        metrics = setup.get_model_metrics(model_id)
        correlations[model_id] = metrics['pearson']
        correlations_total[model_id] = metrics['pearson_total']

    # --- 1. Pairwise correlations ---
    corr_results = {}
    for (m1, f1), (m2, f2) in combinations(encodeds.items(), 2):
        corr, _ = pearsonr(f1, f2)
        corr_results[(m1, m2)] = corr

    corr_df = pd.DataFrame(
        [(*k, v) for k, v in corr_results.items()],
        columns=["model_a", "model_b", "pearson_corr"]
    )
    sns.set(style="whitegrid", context="talk")

    # Raw encodings dataframe
    df_raw_list = []
    for model_id, enc in encodeds.items():
        short = model_id.split('/')[-1]
        states = labels[model_id]
        df_raw_list.append(pd.DataFrame({
            "encoded": enc,
            "state": states,
            "model": short
        }))
    df_raw = pd.concat(df_raw_list, ignore_index=True)

    # Pairwise differences dataframe
    df_pairs_list = []
    for (m1, f1), (m2, f2) in combinations(encodeds.items(), 2):
        short1 = m1.split('/')[-1]
        short2 = m2.split('/')[-1]
        pair_name = f"{short1} - {short2}"
        diffs = np.array(f1) - np.array(f2)
        pair_states = labels[m1]  # assumes same ordering
        df_pairs_list.append(pd.DataFrame({
            "diff": diffs,
            "state": pair_states,
            "pair": pair_name
        }))
    df_pairs = pd.concat(df_pairs_list, ignore_index=True) if df_pairs_list else pd.DataFrame(
        columns=["diff", "state", "pair"])

    # x-axis state order
    preferred_order = ["M", "F", "N"]
    present = list(dict.fromkeys(list(df_raw["state"].unique()) + list(df_pairs["state"].unique())))
    state_order = [s for s in preferred_order if s in present] or present

    # Create side-by-side plots
    fig, (ax_raw, ax_diff) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # Raw encodings
    sns.violinplot(
        data=df_raw,
        x="state", y="encoded",
        hue="model",
        order=state_order,
        dodge=True, inner="quartile", scale="width",
        palette="Set2", ax=ax_raw
    )
    ax_raw.set_title("Raw encoded values by state")
    ax_raw.set_xlabel("State")
    ax_raw.set_ylabel("Encoded value")
    ax_raw.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Pairwise differences
    if not df_pairs.empty:
        sns.violinplot(
            data=df_pairs,
            x="state", y="diff",
            hue="pair",
            order=state_order,
            dodge=True, inner="quartile", scale="width",
            palette="Set2", ax=ax_diff
        )
        ax_diff.set_title("Pairwise differences by state")
        ax_diff.set_xlabel("State")
        ax_diff.set_ylabel("Δ encoding")
        ax_diff.legend(title="Model pair", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax_diff.text(0.5, 0.5, "No model pairs to show", ha="center", va="center")
        ax_diff.set_axis_off()

    plt.tight_layout()
    plt.show()

    # --- 3. Aggregate stability metric ---
    stability = np.mean(list(corr_results.values()))
    print(f"Aggregate stability across models: {stability:.3f}")

    # mean absolute difference per class
    for (m1, f1), (m2, f2) in combinations(encodeds.items(), 2):
        diffs = np.abs(np.array(f1) - np.array(f2))
        df_diff = pd.DataFrame({"diff": diffs, "label": labels[m1]})
        mean_diffs = df_diff.groupby("label")["diff"].mean()
        print(f"Mean absolute differences between {m1} and {m2}:")
        print(mean_diffs)



def prepare_stability_data(setup, architecture_map, cache_dir="cache", force_reload=False):
    """
    architecture_map: dict { 'Display Name': [path_seed_0, path_seed_1, ...] }

    Caches are stored individually per (model_name + paths).
    """
    os.makedirs(cache_dir, exist_ok=True)

    all_raw_dfs = []
    all_diff_dfs = []

    print(f"Checking status for {len(architecture_map)} models...")

    for model_name, run_paths in architecture_map.items():
        # --- 1. Generate Unique Hash for THIS Model ---
        # We combine name + paths to ensure uniqueness.
        # Sorting paths ensures [a, b] == [b, a] in case list order varies
        config_str = f"{model_name}_{str(sorted(run_paths))}"
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()

        # Make filename readable: "stability_BERT_a1b2c3.pkl"
        safe_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_')).strip()
        cache_file = os.path.join(cache_dir, f"stability_{safe_name}_{config_hash}.pkl")

        # --- 2. Try Loading Cache ---
        loaded = False
        if not force_reload and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    df_raw_sub, df_diff_sub = pickle.load(f)
                    # Append to main lists
                    all_raw_dfs.append(df_raw_sub)
                    if not df_diff_sub.empty:
                        all_diff_dfs.append(df_diff_sub)
                    print(f"✅ [{model_name}] Loaded from cache.")
                    loaded = True
            except Exception as e:
                print(f"⚠️ [{model_name}] Cache corrupted ({e}). Recomputing...")

        # --- 3. Compute if not cached ---
        if not loaded:
            print(f"🔄 [{model_name}] Computing fresh data...")

            run_data = {}
            raw_rows_sub = []
            diff_rows_sub = []

            # A. Process individual seeds
            for run_path in run_paths:
                # Analyze
                analysis = setup.analyze_models(run_path)

                # Map Neutrals
                analysis['state'] = analysis['state'].map(
                    lambda s: s if isinstance(s, str) and s in ['M', 'F'] else 'N'
                )

                # Directional Alignment (F > M)
                means = analysis.groupby('state')['encoded'].mean()
                if 'M' in means and 'F' in means and means['M'] > means['F']:
                    analysis['encoded'] = -analysis['encoded']

                seed_id = int(run_path.strip('/').split('/')[-1]) + 1
                run_data[seed_id] = analysis

                # Add Noise for Visualization
                temp_df = analysis.copy()
                temp_df['model'] = model_name
                temp_df['seed'] = seed_id

                raw_rows_sub.append(temp_df[['model', 'state', 'encoded', 'seed']])

            # B. Process Pairwise Differences
            seeds = list(run_data.keys())
            for s1, s2 in combinations(seeds, 2):
                # Note: Assuming dataframes are aligned.
                # If row counts differ, consider merging on index.
                vals1 = run_data[s1]['encoded'].values
                vals2 = run_data[s2]['encoded'].values
                states = run_data[s1]['state'].values

                diffs = vals1 - vals2

                diff_rows_sub.append(pd.DataFrame({
                    'model': model_name,
                    'state': states,
                    'diff': diffs,
                    'pair': f"{s1}-{s2}"
                }))

            # Create Sub-DataFrames
            df_raw_sub = pd.concat(raw_rows_sub, ignore_index=True)
            df_diff_sub = pd.concat(diff_rows_sub, ignore_index=True) if diff_rows_sub else pd.DataFrame()

            # C. Save THIS model to its own cache file
            with open(cache_file, 'wb') as f:
                pickle.dump((df_raw_sub, df_diff_sub), f)

            # Add to main lists
            all_raw_dfs.append(df_raw_sub)
            if not df_diff_sub.empty:
                all_diff_dfs.append(df_diff_sub)

    # --- 4. Combine All Models ---
    final_raw = pd.concat(all_raw_dfs, ignore_index=True) if all_raw_dfs else pd.DataFrame()
    final_diff = pd.concat(all_diff_dfs, ignore_index=True) if all_diff_dfs else pd.DataFrame()

    return final_raw, final_diff


# ---------------------------------------------------------
# 2. Plotting Logic
# ---------------------------------------------------------
def plot_six_stability_views(df_raw, df_diff):
    init_matplotlib(use_tex=True)

    font_size = 20
    # Setup Palette
    paired = get_cmap("Paired")
    colors = paired.colors
    # Custom colors indices based on your request (0,1,6,7)
    # We need enough colors for seeds.
    # If you have 3 seeds, we need 3 colors. If 5 seeds, 5 colors.
    # Let's generate a dynamic palette based on the "Paired" map or use your custom list
    custom_palette = [colors[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]

    # Define Rows (States)
    states = ['F', 'M', 'N']
    state_titles = {
        'F': r'Female',
        'M': r'Male',
        'N': r'Neutral'
    }

    # Ground Truth Labels for Yellow Dots
    # Aligned with the logic F > M
    label_map = {'F': 1.0, 'M': -1.0, 'N': 0.0}

    # Layout: 3 rows (F, M, N), 2 columns (Raw, Diff)
    fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharex=True)

    # Get unique models for X-axis ordering
    # map models to pretty_models for better display
    df_raw['model'] = df_raw['model'].map(lambda x: pretty_models.get(x.split('/')[-1], x))
    df_diff['model'] = df_diff['model'].map(lambda x: pretty_models.get(x.split('/')[-1], x))

    model_order = df_raw['model'].unique()
    custom_order = ['bert-base-cased', 'bert-large-cased', 'distilbert-base-cased', 'roberta-large', 'gpt2', 'Llama-3.2-3B', 'Llama-3.2-3B-Instruct']
    model_order = [pretty_models.get(m, m) for m in custom_order if pretty_models.get(m, m) in model_order]

    # --- Helper for Inline Legend ---
    def create_inline_legend(ax, title_text, bbox):
        # 1. Get handles/labels generated by Seaborn
        handles, labels = ax.get_legend_handles_labels()
        if not handles: return

        # 2. Create an invisible handle for the title
        # We use Latex bold for the title part
        title_handle = Line2D([], [], color='none', marker='', linewidth=0)

        # 3. Combine: [Title_Handle, Handle1, Handle2...]
        # We append a colon to the title text
        all_handles = [title_handle] + handles
        all_labels = [f"{title_text}:"] + labels

        # 4. Plot Legend
        # ncol=len(all_labels) forces everything into 1 row
        ax.legend(
            all_handles, all_labels,
            loc='upper center',
            bbox_to_anchor=bbox,
            ncol=len(all_labels),
            #frameon=False,  # cleaner look
            fontsize=font_size - 4,
            handletextpad=0.2,  # reduce gap between color and text
            columnspacing=1.5  # space between groups
        )

    for row_idx, state in enumerate(states):
        # Filter Data
        subset_raw = df_raw[df_raw['state'] == state]
        subset_diff = df_diff[df_diff['state'] == state] if not df_diff.empty else pd.DataFrame()

        # --- Left Column: Raw Distributions (Hue = Seed) ---
        ax_raw = axes[row_idx, 0]

        if not subset_raw.empty:
            sns.violinplot(
                data=subset_raw,
                x='model', y='encoded', hue='seed',
                order=model_order,
                palette=custom_palette,
                inner='quartile', density_norm='width', linewidth=0.7,
                #split=True,
                ax=ax_raw, dodge=True  # Use dodge to separate seeds side-by-side
            )

            # Add Yellow Dots (Ground Truth)
            # We calculate x-offsets because dodge shifts the violins
            # This is an approximation. With dodge=True, seaborn shifts bars.
            # Simplest way: Plot a line or dot at the mathematical center
            # or just 1 dot per model group indicating the ideal mean.
            #for i, mod in enumerate(model_order):
            #    ground_truth = label_map[state]
            #    ax_raw.scatter(i, ground_truth, color='yellow', edgecolor='black', s=40, zorder=10)

        ax_raw.set_ylabel(f"{state_titles[state]}\n$h$", fontsize=font_size)
        ax_raw.set_xlabel("")
        ax_raw.set_ylim(-1.25, 1.25)
        y_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax_raw.set_yticks(y_ticks)
        ax_raw.grid(True, axis='y', zorder=0, alpha=0.5)

        # Only show legend in the top-left plot to save space
        if row_idx == 0:
            create_inline_legend(ax_raw, "Run", bbox=(0.5, 1.30))
        else:
            if ax_raw.get_legend(): ax_raw.legend_.remove()

        # --- Right Column: Pairwise Diffs (Hue = Pair) ---
        ax_diff = axes[row_idx, 1]

        if not subset_diff.empty:
            sns.violinplot(
                data=subset_diff,
                x='model', y='diff', hue='pair',
                order=model_order,
                palette="Reds",  # Use a different palette for heat/diffs
                inner='quartile', density_norm='width', linewidth=0.7,
                ax=ax_diff, dodge=True
            )
            # Add Yellow Line at 0 (Ideal Diff)
            #ax_diff.axhline(0, color='yellow', linestyle='--', linewidth=1.5, zorder=10)

        ax_diff.set_ylabel(r"$\Delta h$", fontsize=font_size)
        ax_diff.set_xlabel("")
        ax_diff.set_ylim(-1.25, 1.25)
        ax_diff.set_yticks(y_ticks)
        ax_diff.grid(True, axis='y', zorder=0, alpha=0.5)

        if row_idx == 0:
            create_inline_legend(ax_diff, "Runs", bbox=(0.5, 1.30))
        else:
            if ax_diff.get_legend(): ax_diff.get_legend().remove()

    # Formatting Axes
    for ax in axes.flatten():
        # Font sizes
        ax.tick_params(axis='both', which='major', labelsize=font_size - 4)

    # Rotate X-labels on the bottom row
    for ax in axes[-1, :]:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right', fontsize=font_size - 2)
        plt.xlabel("Model", fontsize=font_size)

    plt.tight_layout()

    # Save
    output = 'img/stability_6_plots.pdf'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.show()

def print_stability_table(setup, architecture_map):
    """
    architecture_map = {
        "bert-base-cased": [
            ".../0",
            ".../1",
            ".../2",
        ],
        ...
    }
    """
    rows = []

    for model_name, paths in architecture_map.items():
        seeds = []
        encs = []

        # --- Load encodings for each seed ---
        for p in paths:
            analysis = setup.analyze_models(p)

            # normalize states
            analysis['state'] = analysis['state'].map(
                lambda s: s if isinstance(s, str) and s in ['M','F'] else 'N'
            )

            # align direction
            means = analysis.groupby('state')['encoded'].mean()
            if 'M' in means and 'F' in means and means['M'] > means['F']:
                analysis['encoded'] = -analysis['encoded']

            seeds.append(os.path.basename(p.rstrip('/')))
            encs.append(analysis['encoded'].values)

        # --- Pearson correlations per seed ---
        pearsons = []
        for p in paths:
            metrics = setup.get_model_metrics(p)
            pearsons.append(abs(metrics['pearson_total']))

        pearson_mean = np.mean(pearsons)

        # --- L2 differences ---
        diffs = []
        for i, j in combinations(range(len(encs)), 2):
            #d = np.linalg.norm(encs[i] - encs[j])
            d = np.abs(encs[i] - encs[j]).mean()
            diffs.append(d)

        # Pad to always print three pairs (0–1, 0–2, 1–2)
        diff_01 = diffs[0] if len(diffs) > 0 else np.nan
        diff_02 = diffs[1] if len(diffs) > 1 else np.nan
        diff_12 = diffs[2] if len(diffs) > 2 else np.nan
        diff_mean = np.nanmean([diff_01, diff_02, diff_12])

        rows.append((
            model_name,
            pearsons[0], pearsons[1], pearsons[2], pearson_mean,
            diff_01, diff_02, diff_12, diff_mean
        ))

    # --- Print LaTeX table body ---
    print()
    print("% --- Stability Table ---")
    for r in rows:
        (name,
         p0, p1, p2, pm,
         d01, d02, d12, dm) = r

        print(
            f"{name} & "
            f"{p0:.3f} & {p1:.3f} & {p2:.3f} & {pm:.3f} & "
            f"{d01:.3f} & {d02:.3f} & {d12:.3f} & {dm:.3f} \\\\"
        )
    print("% -----------------------")


if __name__ == '__main__':
    setup = GenderEnSetup()

    base_models = [
        'bert-base-cased',
        'bert-large-cased',
        'roberta-large',
        'distilbert-base-cased',
        'gpt2',
        'meta-llama/Llama-3.2-3B',
        'meta-llama/Llama-3.2-3B-Instruct',
    ]
    consistency_data = []
    data = {}

    for base_model in base_models:
        model_ids = [
            f'results/experiments/gradiend/gender-en/{base_model}/vFinal/0',
            f'results/experiments/gradiend/gender-en/{base_model}/vFinal/1',
            f'results/experiments/gradiend/gender-en/{base_model}/vFinal/2',
        ]
        print(f"Processing {base_model}...")
        data[base_model] = model_ids
#        encs, labels = extract_aligned_features(setup, model_ids)

        # 3. Package for Plotting
        # The plotter expects: (Name, EncodingsDict, LabelsDict)
#        consistency_data.append((base_model, encs, labels))


    df_raw, df_diff = prepare_stability_data(setup, data, force_reload=False)
    plot_six_stability_views(df_raw, df_diff)
    #print_stability_table(setup, data)

