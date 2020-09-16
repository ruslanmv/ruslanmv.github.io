---
title: "Create a Flask Regression Plot App Docker Container."
categories: "Linux"

excerpt: "Create a Flask Regression Plot App Docker Container."

header:
    image: "/assets/images/posts/porto.jpg"
    teaser: "/assets/images/posts/genova-container.jpg"
    caption: "Porto Antico - Genova Italy"
---

Today I wanted to recap  Create a Flask Seaborn Regression Plot App Docker Container



In this blog post, we'll see how to create a Docker Container in Mac OS, and how to use  Docker Container to  Create a Flask Seaborn Regression Plot App

# Flask  Regression with  App Docker Container

Now we will work on Docker Container. We will accomplish it by completing each task in the project:

- Install a Linux image into a docker container.
- Use the Pandas package with Seaborn to create a regression plot.
- Use flask to create a web application that returns a plot.
- Build a requirements document with packages needed for the application.
- Build the application in a container using a Dockerfile and test it.



# Install Docker on your Mac OS using Homebrew

Note that `brew install docker` and `brew cask install docker` is different. Some of the instructions about docker installation on Mac OS use the latter code that installs Docker as an Application,

```unix
$ brew install docker docker-machine
$ brew cask install virtualbox
-> need password
-> possibly need to address System Preference setting
$ docker-machine create --driver virtualbox default
$ docker-machine env default
$ eval "$(docker-machine env default)"
$ docker run hello-world
$ docker-machine stop default
```


# List Docker Containers

To list all running Docker containers, enter the following into a terminal window:

```unix
docker ps –a
```

The **`ps`** command provides several columns of information:

- `Container ID` – a unique alphanumeric number for each container
- `Image` – The base operating system image the container is based on
- `Command` – The command that launched the container
- `Created` – How long ago the container was created
- `Status` – Uptime or downtime
- `Ports` – Specifies any ports forwarded to the container for networking
- `Name` – A memorable name assigned by the Docker software

To create a new container from an image and start it, use **`docker run`**:

```unix
docker run [options] image [command] [argument] 
```

To immediately **remove** a docker container without waiting for the grace period to end use:

```output
docker rm  container_id
```

## Install Linux image into container

Verify the docker installation.

After installation, add yourself to the docker group, otherwise use sudo for commands.

Here, you are added to the docker group

Start a Docker container with a Linux Image.

Use Docker command ps-a to view containers.

Enter a Container Linux environment in a bash shell.

Run and remove a named container.

## Use the Pandas package with Seaborn to create a regression plot

Create a Pandas Dataframe

Create a Seaborn regression plot to show possible correlation.

Save the file locally to verify code.

Return the data file.



## Use flask to create a web application that returns a plot.

Import objects and methods to create a flask application.

Use the regression plot to get an image.

Return the image to the browser window.

Run the application and browse to localhost:5000

## Build a requirements document with packages needed for the application.

Create a file called requirements.txt

Determine the versions of packages needed for the application.

pip freeze | grep pandas

pip freeze | grep matplotlib

pip freeze | grep seaborn

 pip freeze | grep matplotlib

python

import flask

flask.__ version __





Add the packages and versions to the requirements.txt.

Create a Dockerfile for use in the next task

## Build the application in a container using a Dockerfile and test it.

Load the base image.

Copy the requirements doc to the container. 

Identify the working directory in the container image.

Identify the port to run on.

Create the command to run the app.

docker build --tag flask-plotting-app .

run -i -t --name flaskpltapp -p5000:5000 flask-plotting-app:latest

