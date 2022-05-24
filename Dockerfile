FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
RUN apt-get update \
    && apt-get install -y git python3 python3-pip  \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && pip3 install torch torchvision torchaudio tensorflow --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip3 install numpy pandas scipy tqdm scikit-learn \
    && pip3 install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
CMD cd \
    && git clone https://github.com/tinyrolls/2022-KGAT.git \
    && cd 2022-KGAT \
    && python Main.py --data_name last-fm
