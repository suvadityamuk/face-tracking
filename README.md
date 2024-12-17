# Face Tracking in Videos

## Introduction
This specific task perform face-tracking for a given individual in an available video. This is useful for generating face-snippets for a given person as part of a larger video which may involve changes in scenes etc. as well

## Set-up Instructions

Follow the steps below to run the script on your machine.

### Dependencies
- matplotlib
- deepface
- triton
- torch
- torchvision
- torchcodec
- tqdm
- cv2

### Setup & Download
First, clone the repository onto the machine of your choice.
```
git clone https://github.com/suvadityamuk/face-tracking.git
```

#### VS Code
If using VS Code, clone the repository and open it. Then, use the [DevContainers](https://code.visualstudio.com/docs/devcontainers/containers) option (bottom left corner of your VSCode screen) to open up a DevContainer, which will spin up a container based on the Dockerfile present.

1. Move the image and video data into the container either through `docker cp` (detailed instructions given below) or by using the GUI to move data around
2. Change the config.json file to reflect the path of the video and reference image
3. Run `python app.py` to execute the script and generate the clips and metadata

#### General
If using a terminal, use Docker to create a new container based on the Dockerfile available in the repository.

```
docker build -t face-tracking .
docker run -it --rm face-tracking bash
```
Open a new terminal side-by-side. Make use of `docker cp` to move your video and image files into the container. To do so:

1. Find your container ID using `docker ps`
```
CONTAINER ID           IMAGE                   COMMAND     CREATED     STATUS      PORTS       NAMES
<your-container-id>    <random-image-name>     ...         ...         ...         ...         ...
```
2. Use docker cp to move files from your host to your container. Alternatively, mount a host volume to the container to simplify this process.
```
docker cp /path/to/video.mp4 <your-container-id>:/some/other/location/
docker cp /path/to/image.jpg <your-container-id>:/some/other/location/
```
3. Update the `config.json` inside your container to reflect the path of the video and image inside the container
4. Run the script to generate the clips and metadata.
```
python app.py
```

#### Non-Docker
If you want to keep things simple and not use Docker or some other technique altogether, then follow the steps below

1. Change `config.json` to reflect the absolute path of the video and the reference image
2. Create a Virtual environment using `python3 -m venv .venv`
3. Activate the virtual environment using `source .venv/bin/activate`
3. Run `pip3 install -r requirements.txt`
4. Run the script using `python app.py`

## Method

### Step 1
We process the video using `torchcodec` and load the video using it. This allows us to index and traverse through the video not only through numerical indexing but also timestamp-based indexing.

### Step 2
We load the reference image using `cv2.imread()` and then perform a `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` to get a final RGB image.

### Step 3
We iterate over each frame in the video and identify each face using `DeepFace`. Speficically, we make use of the `OpenCV` algorithm, that allows us to performantly extract each face and also maintain relevant metadata for the extraction using Haar Cascades.

### Step 4
We instantiate a VGG-16 model with only the weights upto the feature-module loaded into the model. We discard the classifier and instead make use of a Flatten layer to handle the AdaptiveAvgPool2d output. This acts as a Siamese network, which allows us to extract and compare the embeddings of the reference image and the candidate frame.

### Step 5
We calculate the Euclidean distance between the reference image and against each face detected in the frame, select it, and save it into a list of frames.

### Step 6
We check if there are any changes in the background of the image based on a Mixture of Gaussians architecture through OpenCV. Based on the detection of a background change, we save the current-available frames into a single clip and then, clear out all the variables to free-up some memory.

### Step 7
We record all numerical information like bounding box characteristics, and save it against each video into a metadata file.

### Step 8
We finally complete the full iteration of the frames and save the last available frames into a video along with a metadata file.

## Samples

## Assumptions
- We use mostly free-tier hardware (Colab T4 GPUs and L4 GPUs) to run all scratch experiments and write the code.
- The current solution is light-weight in nature and uses optimized small Deep Learning models as they tend to be heavier at inference-time.
- The GPU use of this current script is minimal (especially since the solution is heavily Computer-Vision oriented), but it can be used through the SIFT Similarity approach too.

## Limitations
- The face-recognition mechanism is difficult to fine-tune as the L1 threshold (instead of L2 to allow for a better decision boundary) has significantly high variance and can misclassify the target face at times.
- The model and my approach is optimized for a lightweight deployment, preferring speed of execution/inference over quality. I have optimized the entire pipeline to upto 20 frames per second, which I believe can be made better through smarter choices for decision boundaries and/or a finetuned model like a YOLO.
- The current system seems to heavily rely on the qualitative characteristics of the reference image and has no baseline reference to accept/reject a frame. For example, Ryan Reynolds with glasses is exponentially more accurate than Ryan Reynolds without his glasses.
- Audio has not been preserved in these clips as the jump in frames led to the audio changes being quite jarring.

## Observations
- Generating embeddings and finding the Euclidean Distance between the reference image and the candidate image does not always guarantee a face match.
- On the other hand, the traditional template-matching method almost guarantees a match except for the case of look-alikes/twins in the video wherein the selection becomes random.
- Using MediaPipe can yield exponentially better performance in terms of speed, but has a significantly lower detection quality for medium-long range distances, thus rendering it ineffective.
- I made attempts to use the descriptors from the SIFT process and send it into a FLANN KNN-based matcher. But it would not work well without fine-tuning of a decision boundary.
- There is still not a strong logic for how we can have the model be confident that the reference-image and candidate-frame match when there is only one face in the frame except for arbitrary bounds that do not scale with the number of videos processed.

## Example scripts to get results

Please navigate to the [`Media`](./media) section to find sample clips and metadata.