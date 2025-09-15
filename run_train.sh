TRAIN_PATH="data/tinystory/train_tokens"
VAL_PATH="data/tinystory/valid_tokens"
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=16
D_FF=1344 #D_FF = 8/3 D_MODEL
DEVICE="cuda"
LR=0.001
BETAS="0.9,0.999"
EPS=1e-8
WEIGHT_DECAY=0.01
BATCH_SIZE=64
EPOCHS=20 #20
STEPS_PER_EPOCH=1000 #000
CLIP_GRAD=1.0
USE_SCHEDULER=true
WARMUP_T=2000
COS_CYCLE_T=10000
LR_MIN=1e-5
RESUME=false
CHECKPOINT_PATH="model/checkpoint"
CKPT_INTERVAL=1000 #000
LOG_INTERVAL=100 #00
VAL_TIMES=5

# 执行 Python 脚本
uv run python cs336_basics/train.py \
    --train_path $TRAIN_PATH \
    --validation_path $VAL_PATH \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --d_ff $D_FF \
    --device $DEVICE \
    --lr $LR \
    --betas $BETAS \
    --eps $EPS \
    --weight_decay $WEIGHT_DECAY \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --steps_per_epoch $STEPS_PER_EPOCH\
    --clip_grad $CLIP_GRAD \
    --use_scheduler \
    --warmup_t $WARMUP_T \
    --cos_cycle_t $COS_CYCLE_T \
    --lr_min $LR_MIN \
    --checkpoint_path $CHECKPOINT_PATH \
    --ckpt_interval $CKPT_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --val_times $VAL_TIMES \