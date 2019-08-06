FROM ubuntu:16.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /app

RUN \
  apt-get update && \
  apt-get install -y python3 python3-dev python3-pip python3-virtualenv && \
  rm -rf /var/lib/apt/lists/*


#RUN apt-get update && apt-get install -y --no-install-recommends apt-utils && apt-get install -y libgtk2.0-dev python python-dev python3 python3-dev python3-pip

#RUN apt-get update && apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

RUN pip3 install setuptools pip --upgrade --force-reinstall


# Install any needed packages specified in requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt

#RUN apt-get update -y

# Install packages
#RUN apt-get install -y curl
#RUN apt-get install -y postgresql
#RUN apt-get install -y postgresql-client
#RUN apt-get install -y python3-numpy python3-opengl python-qt4 python-qt4-gl


# Run app.py when the container launches
#CMD ["python3", "examples/sample.py"]
