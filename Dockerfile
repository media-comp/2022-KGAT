FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
WORKDIR /2022-KGAT
ADD . / ./
RUN apt-get -qq update \
    && apt-get -qq install -y git python3 python3-pip  \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && pip3 install --quiet torch torchvision torchaudio tensorflow --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip3 install --quiet numpy pandas scipy tqdm scikit-learn \
    && pip3 install --quiet dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
RUN python Main.py --data_name last-fm --n_epoch 1
