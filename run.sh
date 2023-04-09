n1=3    # n_dim
n2=200  # n_steps
n3=4    # n_layers
n4=4    # embedD
n5=100  # embedN
n6=256  # hidden_size
n7=1000  # n_epochs
n8=1024 # batch_size
n9=1.0e-5   # lbeta
n10=0.01 # ubeta
n11="linear"   # noise


if [ "$1" == "train" ]; then
    source activate cs726
    rm -rf runs
    python train.py --n_dim $n1 --n_steps $n2  --n_layers $n3 --embedD $n4 --embedN $n5 --hidden_size $n6 --n_epochs $n7 --batch_size $n8 --lbeta $n9 --ubeta $n10 --noise $n11
fi


if [ "$1" == "eval" ]; then
    source activate cs726
    rm -rf results
    python eval.py --ckpt_path runs/results/last.ckpt \
                    --hparams_path runs/results/lightning_logs/version_0/hparams.yaml \
                    --eval_nll --eval_emd \
                    # --vis_diffusion --vis_overlay
fi