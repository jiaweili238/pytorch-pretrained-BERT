export SQUAD_DIR=/data/home/jiawei/pytorch-pretrained-BERT/project/pytorch-pretrained-BERT/squad

python train.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/data/train-v2.0.json \
  --predict_file $SQUAD_DIR/data/dev-v2.0.json \
  --train_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --output_dir $SQUAD_DIR/results
