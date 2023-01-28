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
for km in "${kge_method[@]}"
do
cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python -u retrain_global.py --cuda \
--local_file_dir ../Data/FB15k-237/C3GlobalUnlearn \
--save_dir ../Output/FB15k-237/C3GlobalRetrain"${km}" \
--model "${km}" \
--client_num 3 \
--max_epoch 200 \
--learning_rate 1e-4 \
--log_epoch 50 \
--valid_epoch 10 \
--early_stop_epoch 3 \
--batch_size 128 \
--test_batch_size 8 \
--hidden_dim 256 \
--gamma 10.0 \
--cpu_num 16 \
| tee -a "${log_folder}"/RetrainGlobalC3_"${km}"_"${cur_time}".txt
sleep 8
done