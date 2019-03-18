export SQUAD_DIR=/home/zhangyue
# /data/home/jiawei/project/pytorch-pretrained-BERT/examples/squad

python train.py \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/data/train-v2.0.json \
  --predict_file $SQUAD_DIR/data/dev-v2.0.json \
  --test_file $SQUAD_DIR/data/test-v2.0.json \
  --dev_eval_file $SQUAD_DIR/data/dev_eval.json \
  --test_eval_file $SQUAD_DIR/data/test_eval.json \
  --eval_steps 5000 \
  --train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --name bert \
  --output_dir $SQUAD_DIR/results/ensemble \
  --train_ling_features_file /home/zhangyue/ling_features/train_ling_features.json \
  --eval_ling_features_file /home/zhangyue/ling_features/eval_ling_features.json \
  --test_ling_features_file /home/zhangyue/ling_features/test_ling_features.json 
