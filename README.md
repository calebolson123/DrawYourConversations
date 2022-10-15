# Draw your Conversations
This repo contains the code for the project as seen here: https://youtu.be/rwcyZxLGel8

## Setup

1) Download and unpack Stable Diffusion zip (https://github.com/CompVis/stable-diffusion) within project directory. (TODO: load SD as module in Flask app)
2) Run `conda env create -f environment.yml` to create conda env containing dependencies
3) Run `activate ldm` to jump into created env
4) Run `python server.py` to start the server

## Test it out

On startup, server waits for requests on port 5000. Try sending a POST request to:

```https://localhost:5000/draw```

With a JSON payload which looks like this:

```
{   
    "conversation": "This is your last chance. After this, there is no turning back. You take the blue pill - the story ends, you wake up in your bed and believe whatever you want to believe. You take the red pill - you stay in Wonderland and I show you how deep the rabbit-hole goes."
}
```

The server will respond w/ the generated image including the engineered text embedded onto the image.