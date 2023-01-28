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
CUDA_VISIBLE_DEVICES=${gpu} python -u local.py --cuda \
--local_file_dir ../Data/FB15k-237/C3FL \
--save_dir ../Output/FB15k-237/C3Local \
--model "${km}" \
--agg weighted \
--client_num 3 \
--aggregate_iteration 1 \
--max_epoch 200 \
--valid_epoch 10 \
--hidden_dim 256 \
--early_stop_epoch 3 \
--learning_rate 1e-4 \
--batch_size 512 \
--test_batch_size 16 \
| tee -a "${log_folder}"/C3Local_"${km}"_"${cur_time}".txt
sleep 8
done