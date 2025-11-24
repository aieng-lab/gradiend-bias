import argparse
import json
from collections import defaultdict


def print_results(
    base,
    finetuned,
    base_gradiend,
    finetuned_gradiend,
    gradiend_finetuned,
    gradiend_finetuned_gradiend,
    model_type,
    ss_model_type_suffix='',
):

    results = defaultdict(dict)

    # superglue wsc
    for model in [finetuned, finetuned_gradiend, gradiend_finetuned, gradiend_finetuned_gradiend]:
        model_id = model.split('/')[-1]
        key_model_id = model_id
        if model in [finetuned_gradiend, gradiend_finetuned_gradiend]:
            key_model_id += '-ft-wsc.fixed'
        file = f'results/super_glue-finetuning/wsc.fixed/{key_model_id}/eval_results.json'
        try:
            data = json.load(open(file, 'r'))
            results[model_id]['wsc'] = data['eval_accuracy']
        except FileNotFoundError:
            print(f'File not found: {file}')
            continue

    # seat
    all_models = [base, base_gradiend, finetuned, finetuned_gradiend, gradiend_finetuned, gradiend_finetuned_gradiend]
    for model in all_models:
        model_id = model.split('/')[-1]
        key_model_id = model_id
        if model in [finetuned, gradiend_finetuned]:
            key_model_id = f'super_glue-finetuning-wsc.fixed-{model_id}'

        file = f'../bias-bench/results/seat/gender/seat_m-{model_type}Model_c-{key_model_id}.json'
        try:
            data = json.load(open(file, 'r'))
            results[model_id]['seat'] = data['effect_size']
        except FileNotFoundError:
            print(f'File not found: {file}')
            continue

    # stereoset
    stereoset_file = '../bias-bench/results/stereoset.json'
    stereoset_data = json.load(open(stereoset_file, 'r'))
    for model in all_models:
        model_id = model.split('/')[-1]
        key_model_id = model_id
        if model in [finetuned, gradiend_finetuned]:
            key_model_id = f'super_glue-finetuning-wsc.fixed-{model_id}'
        stereoset_key = f'stereoset_m-{model_type}{ss_model_type_suffix}_c-{key_model_id}'
        try:
            results[model_id]['stereoset'] = stereoset_data[stereoset_key]
        except KeyError:
            print(f'Key not found: {stereoset_key}')
            continue

    base_id = base.split('/')[-1]
    finetuned_id = finetuned.split('/')[-1]
    base_gradiend_id = base_gradiend.split('/')[-1]
    gradiend_finetuned_id = gradiend_finetuned.split('/')[-1]
    finetuned_gradiend_id = finetuned_gradiend.split('/')[-1]
    gradiend_finetuned_gradiend_id = gradiend_finetuned_gradiend.split('/')[-1]


    base_latex = {
        'distilbert-base-cased': r'\distilbert',
        'bert-base-cased': r'\bertbase',
        'bert-large-cased': r'\bertlarge',
        'roberta-large': r'\roberta',
        'gpt2': r'\gpttwo',
        'Llama-3.2-3B': r'\llama',
        'Llama-3.2-3B-Instruct': r'\llamai',
    }[base_id]
    pretty_models = {
        base_id: base_latex,
        finetuned_id: rf'{base_latex} $\to$ WSC',
        base_gradiend_id: rf'{base_latex} $\to$ \gradiendbpi',
        finetuned_gradiend_id: rf'{base_latex} $\to$ WSC $\to$ \gradiendbpi',
        gradiend_finetuned_id: rf'{base_latex} $\to$ \gradiendbpi $\to$ WSC',
        gradiend_finetuned_gradiend_id: rf'{base_latex} $\to$ \gradiendbpi $\to$ WSC $\to$ \gradiendbpi',
    }

    base_score_map = {}
    fmt_string = '{:.2f}'

    # First pass: collect rows and diffs
    rows = []
    diff_maps = {"seat": [], "stereoset": [], "wsc": []}

    for model in all_models:
        model_id = model.split('/')[-1]
        line = {"model": pretty_models[model_id], "seat": None, "stereoset": None, "wsc": None}
        diffs = {}

        if model_id in results:
            model_data = results[model_id]

            # SEAT
            if "seat" in model_data:
                score = model_data["seat"] * 100
                base = base_score_map.setdefault("seat", score)
                diff = score - base
                diffs["seat"] = diff
            # SS
            if "stereoset" in model_data:
                score = model_data["stereoset"]["intrasentence"]["gender"]["SS Score"]
                base = base_score_map.setdefault("stereoset", score)
                diff = abs(50 - score) - abs(50 - base)
                diffs["stereoset"] = diff
            # WSC
            if "wsc" in model_data:
                score = model_data["wsc"] * 100
                base = base_score_map.setdefault("wsc", score)
                diff = score - base
                diffs["wsc"] = diff

        # store raw score + diff
        line.update({k: (results[model_id], diffs.get(k),) if model_id in results else None for k in ["seat","stereoset","wsc"]})
        rows.append((model_id, line, diffs))

        for k, d in diffs.items():
            diff_maps[k].append(d)

    # Determine best per column
    best = {
        "seat": min(diff_maps["seat"]) if diff_maps["seat"] else None,
        "stereoset": min(diff_maps["stereoset"]) if diff_maps["stereoset"] else None,
        "wsc": max(diff_maps["wsc"]) if diff_maps["wsc"] else None,
    }

    # Second pass: print with bold
    for model_id, line, diffs in rows:
        model_data = results.get(model_id, {})
        line_data = [pretty_models[model_id]]

        # SS
        if "stereoset" in model_data:
            score = model_data["stereoset"]["intrasentence"]["gender"]["SS Score"]
            diff = diffs["stereoset"]
            score_fmt = fmt_string.format(score)
            if diff != 0:
                arrow = "\\uan" if diff > 0 else "\\dan"
                score_fmt = rf'{arrow}{{{fmt_string.format(diff)}}} {score_fmt}'
            if diff == best["stereoset"]:
                score_fmt = rf'\textbf{{{score_fmt}}}'
            line_data.append(score_fmt)
        else:
            line_data.append("--")

        # SEAT
        if "seat" in model_data:
            score = model_data["seat"] * 100
            diff = diffs["seat"]
            score_fmt = fmt_string.format(score)
            if diff != 0:
                arrow = "\\uan" if diff > 0 else "\\dan"
                score_fmt = rf'{arrow}{{{fmt_string.format(diff)}}} {score_fmt}'
            if diff == best["seat"]:
                score_fmt = rf'\textbf{{{score_fmt}}}'
            line_data.append(score_fmt)
        else:
            line_data.append("--")

        # WSC
        if "wsc" in model_data:
            score = model_data["wsc"] * 100
            diff = diffs["wsc"]
            score_fmt = fmt_string.format(score)
            if diff != 0:
                arrow = "\\uagn" if diff > 0 else "\\dabn"
                score_fmt = rf'{arrow}{{{fmt_string.format(diff)}}} {score_fmt}'
            if diff == best["wsc"]:
                score_fmt = rf'\textbf{{{score_fmt}}}'
            line_data.append(score_fmt)
        else:
            line_data.append("--")

        print(" & ".join(map(str, line_data)) + r" \\")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export finetuning analysis results")
    parser.add_argument('--base', type=str, required=True, help='Base model name or path')
    parser.add_argument('--finetuned', type=str, required=True, help='Finetuned model name or path')
    parser.add_argument('--base_gradiend', type=str, required=True, help='Base Gradiend model name or path')
    parser.add_argument('--finetuned_gradiend', type=str, required=True, help='Finetuned Gradiend model name or path')
    parser.add_argument('--gradiend_finetuned', type=str, required=True, help='Gradiend Finetuned model name or path')
    parser.add_argument('--gradiend_finetuned_gradiend', type=str, required=True, help='Gradiend Finetuned Gradiend model name or path')
    parser.add_argument('--model_type', type=str, required=True, help='Model type key for bias-bench results')
    parser.add_argument('--ss_model_type_suffix', type=str, default='', help='Suffix for stereoset model type key')
    args = parser.parse_args()

    print_results(
        base=args.base,
        finetuned=args.finetuned,
        base_gradiend=args.base_gradiend,
        finetuned_gradiend=args.finetuned_gradiend,
        gradiend_finetuned=args.gradiend_finetuned,
        gradiend_finetuned_gradiend=args.gradiend_finetuned_gradiend,
        model_type=args.model_type,
        ss_model_type_suffix=args.ss_model_type_suffix,
    )