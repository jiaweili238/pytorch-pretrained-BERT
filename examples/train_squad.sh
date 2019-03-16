export SQUAD_DIR=/data/home/jiawei/project/pytorch-pretrained-BERT/examples/squad

python train.py \
  --bert_model bert-large-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/data/train-v2.0.json \
  --predict_file $SQUAD_DIR/data/dev-v2.0.json \
  --dev_eval_file $SQUAD_DIR/data/dev_eval.json \
  --eval_steps 5000 \
  --train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --name bert_large \
  --gradient_accumulation_steps 2 \
  --output_dir $SQUAD_DIR/results\bert_large
