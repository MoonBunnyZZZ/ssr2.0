SAVE_ROOT=/media/moon/783e2432-353f-4285-8222-0b39e149e3e4/ckpt/face/attribute/age/regression/ssr2.0/save/
TIME=$(date "+%m%d%H%M")

SAVE_DIR=$SAVE_ROOT$TIME
LOGFILE="$SAVE_ROOT$TIME""/log"

mkdir -p "$SAVE_DIR"

python -u train.py \
 --batch-size 128 \
 --threads-num 2\
 --gpu-id 0\
 --gpus-num 1\
 --db-dir "/home/moon/code/puppyProject/face/attribute/age/regression/ssrMX/rec/imdb/"\
 --print-freq 200 \
 --lr 0.0002 \
 --warmup-num 10 \
 --lr-decay-milestone "40,60,80" \
 --max-epoch 90 \
 --save-dir "$SAVE_DIR" >"$LOGFILE" 2>&1 &