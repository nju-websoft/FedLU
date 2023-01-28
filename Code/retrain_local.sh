gpu=0
log_folder="../Log/""$(date +%Y%m%d)"

while getopts "g:l:" opt;
do
    case ${opt} in
        g) gpu=$OPTARG ;;
        l) log_folder=$OPTARG ;;
        *) echo "Invalid option: $OPTARG" ;;
    esac
done

echo "log folder: " "${log_folder}"
if [[ ! -d ${log_folder} ]];then
    mkdir -p "${log_folder}"
    echo "create log folder: " "${log_folder}"
fi

kge_method=(TransE ComplEx RotatE)
dist_para=(2.0)
gamma=(10.0 20.0 50.0)
learningrate=(1e-4)
confusionmu=(0.05)
for km in "${kge_method[@]}"
do
for dm in "${dist_para[@]}"
do
for g in "${gamma[@]}"
do
for lr in "${learningrate[@]}"
do
for cm in "${confusionmu[@]}"
do
cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python -u retrain_local.py --cuda \
--local_file_dir ../Data/FB15k-237/C3FL \
--save_dir ../Output/FB15k-237/C3LocalRetrain"${km}" \
--fed_mode FedDist \
--dist_mu "${dm}" \
--agg weighted \
--model "${km}" \
--client_num 3 \
--max_iter 100 \
--learning_rate "${lr}" \
--hidden_dim 256 \
--batch_size 512 \
--test_batch_size 16 \
--gamma "${g}" \
--co_dist \
--max_unlearn_epoch 10 \
--max_retrain_epoch 200 \
--confusion_mu "${cm}" \
--cpu_num 0 \
| tee -a "${log_folder}"/RetrainLocalC3_"${km}"_confusion"${cm}"_dist"${dm}"_gamma"${g}"_lr"${lr}"_"${cur_time}".txt
sleep 8
done
done
done
done
done