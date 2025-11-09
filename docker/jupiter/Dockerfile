FROM ubuntu:24.04
LABEL Jupiter with XAI lib
RUN apt-get update	
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install git vim python3.12 python3-pip
RUN (mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.old)
RUN (mkdir /home/xai && cd /home/xai && git clone https://github.com/kdd-lab/XAI-Lib.git)
RUN  /bin/bash -c "cd /home/xai/XAI-Lib && pip install -e . && pip install ucimlrepo && pip install seaborn  && pip install catboost"
#WORKDIR /home/xai
WORKDIR /usr/src/app
RUN python3 -m pip install jupyter
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


