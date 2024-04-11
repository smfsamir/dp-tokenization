# python compare_tokenizers.py \
#     --model_name google/mt5-small \
#     --filename data/english.tsv
# exit 1
for model in "google/mt5-small" "bert-base-multilingual-cased" "bigscience/bloom-3b" "facebook/xlm-v-base"; do
    for dataset in data/english.tsv data/german.tsv data/LADECv1-2019.csv; do
        python compare_tokenizers.py \
            --model_name $model \
            --filename $dataset
    done
done