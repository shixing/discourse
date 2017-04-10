for H in 128 96 64 48 32;
do
    for K in 128 96 64 48 32;
    do
	python run_seq2seq_latent.py p5_1.cfg K=$K size=$H > K$K_H$H.log
    done
done


