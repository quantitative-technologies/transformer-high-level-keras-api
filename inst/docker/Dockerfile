FROM rocker/ml

RUN sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get -y install texlive-full

#RUN install.r remotes
RUN Rscript -e "install.packages('remotes')"
COPY DESCRIPTION .
RUN Rscript -e "remotes::install_deps()"
