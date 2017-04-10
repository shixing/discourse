head = """#!/bin/bash
#PBS -q isi
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=16:gpus=2:shared

source /home/nlg-05/xingshi/sh/init_tensorflow.sh

cd /home/nlg-05/xingshi/lstm/tensorflow/discourse/py

"""

for h in [128,96,64,48,32]:
    for k in [128,96,64,48,32]:
        fn = "K{}_H{}".format(k,h)
        job = open(fn+".job","w")
        job.write(head)
        cmd = "python run_seq2seq_latent.py p5_1.cfg K={} size={} > K{}_H{}.log\n".format(k,h,k,h)
        job.write(cmd)
        job.close()

