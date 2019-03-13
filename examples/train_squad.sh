export SQUAD_DIR=/data/home/jiawei/project/pytorch-pretrained-BERT/examples/squad

python train.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/data/train-v2.0.json \
  --predict_file $SQUAD_DIR/data/dev-v2.0.json \
  --dev_eval_file $SQUAD_DIR/data/dev_eval.json \
  --eval_steps 5000 \
  --train_batch_size 36 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --name bert \
  --output_dir $SQUAD_DIR/results
