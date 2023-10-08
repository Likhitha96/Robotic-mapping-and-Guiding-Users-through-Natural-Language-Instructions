# Project Abstract
Artificial intelligence has made quite an entrance into our daily lives in the form
of mobile applications, smart appliances, voice assistants giving both an easier and
personalized approach to dealing with technology. In our work, we attempt to establish
smart assistance in indoor spaces by focusing on mapping and navigation in indoor
environments. Although there has been a lot of prior work in simultaneous localization
and mapping, popularly called SLAM, in order to map an environment using sensor
data extending it to navigation is still a novel field of research. Our work contains two
parts, the first part emphasizes on devising an architecture for representing an indoor
environment, houses in our case. We use images from several 2D locations that span
the entire house. Object detection and other feature detection algorithms are used to
construct a map. The second part uses this map to generate natural language instructions
to navigate from one place to another in the environment. The instructions are generated
by first constructing a path between two queried locations within the house, extracting
key landmarks (as objects seen during the navigation) and then using a natural language
generator to construct sentences from the landmarks and navigation actions. Given a
starting point and a destination, the system therefore produces a sequence of simple
natural language instructions to take someone from the starting point to the destination.

# Project highlights
The house is comprised of several 2D locations. The following are the steps in solving this
problem
* Navigability computation between two 2D locations within the house
* Reachability graph for all the nodes in the house
* Path computation given source and destination nodes
* Path to keyword translation
* Keywords to meaningful instructions

# Dataset downloading
The dataset for images is available in https://niessner.github.io/Matterport/ and we worked
on a single scan - '17DRP5sb8fy'
The dataset for instructions and 3D location is available in
https://github.com/peteanderson80/Matterport3DSimulator

# Data preprocessing
Run the images through a MaskRCNN simulator trained on indoor images

# Execution
```
cd code

python3 preprocessing/mapping.py
```
For training,
```
python3 examples/sample.py
```
For inference,
```
python3 examples/sample.py --load_checkpoint epoch_number
```
# Dockerising application
```
cd code

docker build --rm -t project_v2 .

docker run -it  -v $PWD:/app project_v2
```

