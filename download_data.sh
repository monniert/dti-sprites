set -e
mkdir -p datasets

# Multi-object datasets
gdown --id 1unCVwJbytG1wHZCQfk41otdtmR8yU1AH --proxy http://proxy.enpc.fr:3128/ -O tetrominoes.zip
unzip tetrominoes.zip && rm tetrominoes.zip
mv tetrominoes datasets/
gdown --id 110RrDlhx_fjXymzmTx1nkC3ZMPmMXjsv --proxy http://proxy.enpc.fr:3128/ -O dsprites_gray.zip
unzip dsprites_gray.zip && rm dsprites_gray.zip
mv dsprites_gray datasets/
gdown --id 12c7ZRNWKzqosz5aUOfL0XBmLHrcN240r --proxy http://proxy.enpc.fr:3128/ -O clevr6.zip
unzip clevr6.zip && rm clevr6.zip
mv clevr6 datasets/

# GTSRB
gdown --id 1d5rlgYeH087oT6AnTNAHtwrnRXv3WeRR --proxy http://proxy.enpc.fr:3128/ -O GTSRB.zip
unzip GTSRB.zip && rm GTSRB.zip
mv GTSRB datasets/

# Weizmann Horse
gdown --id 1fQSWQUCwIB6zkA65D4wlbaGmIg8iO6Ja --proxy http://proxy.enpc.fr:3128/ -O weizmann_horse.zip
unzip weizmann_horse.zip && rm weizmann_horse.zip
mv weizmann_horse datasets/

# Instagram collections
gdown --id 1tv5-_Iz-LD6-FqFxF67py9ot97BOZbUc --proxy http://proxy.enpc.fr:3128/ -O santaphoto.zip
unzip santaphoto.zip && rm santaphoto.zip
mkdir -p datasets/instagram/santaphoto
mv santaphoto datasets/instagram/santaphoto/train
gdown --id 1OCLvojYDomLnI6zP6QghgIkZ8PWwmqCD --proxy http://proxy.enpc.fr:3128/ -O weddingkiss.zip
unzip weddingkiss.zip && rm weddingkiss.zip
mkdir -p datasets/instagram/weddingkiss
mv weddingkiss datasets/instagram/weddingkiss/train
