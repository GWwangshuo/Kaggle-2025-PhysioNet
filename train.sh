torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
main.py \
--batch_size 1 \
--lr 5e-4 \
--epochs 100 --warmup_epochs 5 \
--data_path your-dataset-path
