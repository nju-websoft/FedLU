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
CUDAA_VISIBLE_DEVICES=${gpu} python -u global.py --cuda \
--local_file_dir ../Data/FB15k-237/C3Global \
--save_dir ../Output/FB15k-237/C3Global \
--model "${km}" \
--client_num 3 \
--max_epoch 200 \
--learning_rate 1e-4 \
--log_epoch 10 \
--valid_epoch 10 \
--early_stop_epoch 3 \
| tee -a "${log_folder}"/C3Global_"${km}"_"${cur_time}".txt
sleep 8
done