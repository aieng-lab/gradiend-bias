#!/bin/bash

base_model="distilbert-base-cased"
base_model="bert-large-cased"
base_model="gpt2"
base_model="bert-base-cased"
base_model="roberta-large"
base_model_id=$(basename "$base_model")
model_type="Distilbert"
model_type="GPT2"
model_type="Bert"
model_type="Roberta"
gradiend_model="results/models/${base_model_id}"
gradiend_modified_model="results/changed_models/${base_model_id}-N"
task="wsc.fixed"
benchmark_name="super_glue"
output_dir="results/${benchmark_name}-finetuning/${task}"
seed=0
per_device_train_batch_size=16
bias_bench_dir="../bias-bench"
persistent_dir="${bias_bench_dir}"
seat_tests="sent-weat6 "\
"sent-weat6b "\
"sent-weat7 "\
"sent-weat7b "\
"sent-weat8 "\
"sent-weat8b"
bias_type="gender"

declare -A model_type_suffixes=(
    ["Bert"]="ForMaskedLM"
    ["BertLarge"]="ForMaskedLM"
    ["Roberta"]="ForMaskedLM"
    ["Distilbert"]="ForMaskedLM"
    ["GPT2"]="LMHeadModel"
    ["Llama"]="ForCausalLM"
    ["LlamaInstruct"]="ForCausalLM"
)
model_type_ss_suffix="${model_type_suffixes[${model_type}]}"

gradiend_modified_model_id=$(basename "$gradiend_modified_model")

eval "$(conda shell.bash hook)"
conda activate bias-bench
export PYTHONPATH="${bias_bench_dir}:${PYTHONPATH}"

# Finetune on SuperGLUE.WSC on base_model_id and gradiend_modified_model
if [[ ! -f "${output_dir}/${base_model_id}-${task}/README.md" ]]; then
    python "${bias_bench_dir}/experiments/run_glue.py" \
            --model "${model_type}ForSequenceClassification" \
            --model_name_or_path ${base_model} \
            --task_name ${task} \
            --do_train \
            --do_eval \
            --do_predict \
            --keep_best_checkpoint \
            --max_seq_length 128 \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --seed ${seed} \
            --output_dir "${output_dir}/${base_model_id}-${task}" \
            --persistent_dir ${persistent_dir} \
            --benchmark_name ${benchmark_name}
else
    echo "${output_dir}/${base_model_id}-${task} already exists"
fi

if [[ ! -f "${output_dir}/${gradiend_modified_model_id}-${task}/README.md" ]]; then
    python "${bias_bench_dir}/experiments/run_glue.py" \
            --model "${model_type}ForSequenceClassification" \
            --model_name_or_path ${gradiend_modified_model} \
            --task_name ${task} \
            --do_train \
            --do_eval \
            --do_predict \
            --keep_best_checkpoint \
            --max_seq_length 128 \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --seed ${seed} \
            --output_dir "${output_dir}/${gradiend_modified_model_id}-${task}" \
            --persistent_dir ${persistent_dir} \
            --benchmark_name ${benchmark_name}
else
    echo "${output_dir}/${gradiend_modified_model_id}-${task} already exists"
fi

# train a GRADIEND model on fine-tuned WSC models and create GRADIEND modified models based on  the finetuned models
finetuned_base_model="${output_dir}/${base_model_id}-${task}"
finetuned_pre_gradiend_modified_model="${output_dir}/${gradiend_modified_model_id}-${task}"
conda deactivate
conda activate gradient-alex
finetuned_post_gradiend_modified_base_model="results/changed_models/${base_model_id}-${task}-finetuning-analysis-N"
finetuned_pre_post_gradiend_modified_base_model="results/changed_models/${gradiend_modified_model_id}-${task}-finetuning-analysis-N"
finetuned_post_gradiend_modified_base_model_id=$(basename "$finetuned_post_gradiend_modified_base_model")
finetuned_pre_post_gradiend_modified_base_model_id=$(basename "$finetuned_pre_post_gradiend_modified_base_model")

if [[ ! -d "${finetuned_post_gradiend_modified_base_model}" ]]; then
    python "gradiend/evaluation/finetuning_analysis.py" \
      --model_name ${finetuned_base_model} \
      --mlm_model ${base_model}
else
    echo "${finetuned_post_gradiend_modified_base_model_id} already exists"
fi

echo ${finetuned_base_model}

#exit
if [[ ! -d "${finetuned_pre_post_gradiend_modified_base_model}" ]]; then
    python "gradiend/evaluation/finetuning_analysis.py" \
        --model_name ${finetuned_pre_gradiend_modified_model} \
        --mlm_model ${base_model}
else
    echo "${finetuned_pre_post_gradiend_modified_base_model_id} already exists"
fi

conda deactivate


#exit
# evaluate the GRADIEND modified finetuned models on WSC without further finetuning
conda activate bias-bench
if [[ ! -f "${output_dir}/${finetuned_post_gradiend_modified_base_model_id}-ft-${task}/README.md" ]]; then
    python "${bias_bench_dir}/experiments/run_glue.py" \
            --model "${model_type}ForSequenceClassification" \
            --model_name_or_path ${finetuned_post_gradiend_modified_base_model} \
            --task_name ${task} \
            --do_eval \
            --do_predict \
            --keep_best_checkpoint \
            --max_seq_length 128 \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --seed ${seed} \
            --output_dir "${output_dir}/${finetuned_post_gradiend_modified_base_model_id}-ft-${task}" \
            --persistent_dir ${persistent_dir} \
            --benchmark_name ${benchmark_name}
else
    echo "${output_dir}/${finetuned_post_gradiend_modified_base_model_id}-ft-${task} already exists"
fi


if [[ ! -f "${output_dir}/${finetuned_pre_post_gradiend_modified_base_model_id}-ft-${task}/README.md" ]]; then
    python "${bias_bench_dir}/experiments/run_glue.py" \
            --model "${model_type}ForSequenceClassification" \
            --model_name_or_path ${finetuned_pre_post_gradiend_modified_base_model} \
            --task_name ${task} \
            --do_eval \
            --do_predict \
            --keep_best_checkpoint \
            --max_seq_length 128 \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --seed ${seed} \
            --output_dir "${output_dir}/${finetuned_pre_post_gradiend_modified_base_model_id}-ft-${task}" \
            --persistent_dir ${persistent_dir} \
            --benchmark_name ${benchmark_name}
else
    echo "${output_dir}/${finetuned_pre_post_gradiend_modified_base_model_id}-ft-${task} already exists"
fi

# evaluate


# evaluate base model, finetuned models, and finetuned models after GRADIEND on SS and SEAT
models="${base_model} ${finetuned_base_model} ${gradiend_modified_model} ${finetuned_pre_gradiend_modified_model} ${finetuned_pre_post_gradiend_modified_base_model} ${finetuned_post_gradiend_modified_base_model}"
for model in ${models[@]}; do
    base_model_id=$(basename "$model")
    experiment_id="seat_m-${model_type}Model_c-${base_model_id}"
    if [ ! -f "${persistent_dir}/results/seat/${bias_type}/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python "${bias_bench_dir}/experiments/seat.py" \
            --tests ${seat_tests} \
            --model "${model_type}Model" \
            --model_name_or_path ${model} \
            --persistent_dir "${persistent_dir}" \
            --bias_type ${bias_type}
    else
        echo "${experiment_id} already computed"
    fi

    # ss_model_id is model with removed prefix results/ and results/changed_models and / are replaced by -
    ss_model_id=$(echo "$model" | sed 's|results/changed_models/||' | sed 's|results/||' | sed 's|/|-|g')
    experiment_id="stereoset_m-${model_type}${model_type_ss_suffix}_c-${ss_model_id}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
        python "${bias_bench_dir}/experiments/stereoset.py" \
            --model "${model_type}${model_type_ss_suffix}" \
            --model_name_or_path ${model} \
            --persistent_dir "${persistent_dir}"
    else
        echo "${experiment_id} already computed"
    fi
done

python "${bias_bench_dir}/experiments/stereoset_evaluation.py"

conda deactivate

conda activate gradient-alex
python gradiend/evaluation/finetuning_analysis_export.py \
    --base ${base_model} \
    --finetuned ${finetuned_base_model} \
    --base_gradiend ${gradiend_modified_model} \
    --finetuned_gradiend ${finetuned_post_gradiend_modified_base_model} \
    --gradiend_finetuned ${finetuned_pre_gradiend_modified_model} \
    --gradiend_finetuned_gradiend ${finetuned_pre_post_gradiend_modified_base_model} \
    --model_type ${model_type} \
    --ss_model_type_suffix ${model_type_ss_suffix}

  echo "python gradiend/evaluation/finetuning_analysis_export.py \
  --base ${base_model} \
  --finetuned ${finetuned_base_model} \
  --base_gradiend ${gradiend_modified_model} \
  --finetuned_gradiend ${finetuned_post_gradiend_modified_base_model} \
  --gradiend_finetuned ${finetuned_pre_gradiend_modified_model} \
  --gradiend_finetuned_gradiend ${finetuned_pre_post_gradiend_modified_base_model} \
  --model_type ${model_type} \
  --ss_model_type_suffix ${model_type_ss_suffix}"

