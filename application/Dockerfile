FROM ubuntu:18.04

WORKDIR /app

RUN apt-get update -y && \
    apt-get -y upgrade && \
    apt-get install -y python3-pip libsm6 libxext6 libxrender-dev cython3 mesa-common-dev qt5-default && \
    rm -rf /var/lib/apt/lists/*

ENV TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
RUN python3 -m pip install --upgrade cython

COPY . ./

RUN cd ./models/blood_cell_identification_model && python3 setup.py build_ext --inplace

EXPOSE 8080
ENTRYPOINT [ "python3" ]
CMD ["./main.py"]



