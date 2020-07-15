FROM tensorflow/tensorflow:2.1.0-py3
WORKDIR /usr/src/app
COPY requirements.txt setup.py README.md ./
COPY walter_white ./walter_white
RUN mkdir resources
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd -M -s /bin/sh walter-white && chown -R walter-white:walter-white /usr/src/app
USER walter-white
CMD python walter_white/tran.py
