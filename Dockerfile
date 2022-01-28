FROM python:3.8
USER root
WORKDIR /usr/src/app
ENV DEBIAN_FRONTEND noninteractive
# 必要なパッケージを記載
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl && \
    apt-get clean
# 必要なPythonパッケージを記載
RUN python -m pip install --upgrade pip && pip install \ 
    numpy \
    scipy \
    matplotlib \
    ipython \
    scikit-learn \
    pandas \
    pillow \
    mglearn \
    requests \
    pyperclip \
    beautifulsoup4 \
    jupyterlab \
    keras==2.7.0 \
    Keras-Preprocessing==1.1.2

# arm64対応のtensorflowをインストール
RUN pip install tensorflow -f https://tf.kmtea.eu/whl/stable.html

RUN jupyter serverextension enable --py jupyterlab
ENV DEBIAN_FRONTEND dialog