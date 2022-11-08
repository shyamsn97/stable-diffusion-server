# Stable Diffusion Google Colab FastAPI Server

[![PyPI version](https://badge.fury.io/py/sd-server.svg)](https://badge.fury.io/py/sd-server)

## Note: This is a pretty hacky server / client interface for generic Stable Diffusion pipelines using diffusers. It's not made for production use and hasn't really been optimized completely.

#### Installation

From pip:

```bash
pip install sd-server
```

From source:

```bash
git clone git@github.com:shyamsn97/stable-diffusion-server.git
cd stable-diffusion-server/
python setup.py install
```

### Stable Diffusion Server + Client Interface -- Simple Usage

Here we create a simple server hosting the standard `StableDiffusionPipeline` from the amazing package [`diffusers`](https://github.com/huggingface/diffusers). The client takes in `**kwargs` that should be the arguments passed into the `__call__` function from the pipeline. For instance, the `StableDiffusionPipeline` `__call__` method can be found: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L185. `host` and `port` can be specified in the `start` method for generic hosting.


```python
from sd_server import DiffusersServer, DiffusersClient
from diffusers import StableDiffusionPipeline

STABLE_DIFFUSION_PATH = "runwayml/stable-diffusion-v1-5" # this can be a local path
pipeline_kwargs = {
    "revision":"fp16",
    "torch_dtype":torch.float16
}
device = torch.device('cuda')

server = DiffusersServer.create(
    pretrained_path = STABLE_DIFFUSION_PATH,
    pipeline_cls = StableDiffusionPipeline,
    pipeline_kwargs = pipeline_kwargs,
    enable_attention_slicing = True,
    device = device
)
url = server.start(host="127.0.0.1", port=8000) # url -- either remote or local

# on another host / terminal
client = DiffusersClient(url)

responses = client(prompt='a photo of an astronaut riding a horse on mars', num_images_per_prompt=4) # this should return a list of images

```

### Serving from Google Colab

Full google colab example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13B7EzuDTAsjd0hpUki_ZiqoQ4wgOyZUM?usp=sharing)

Using the server in Google Colab requires an [ngrok](https://ngrok.com/) account, and needs your ngrok auth token https://dashboard.ngrok.com/get-started/your-authtoken to be passed to the server. Note: Ngrok basically opens a tunnel from a dev machine to the cloud, which means its not really that secure, and should be used at your own risk. Read more here: https://stackoverflow.com/questions/36552950/is-ngrok-safe-to-use-or-can-it-be-compromised.


```python
NGROK_AUTH_KEY = "your auth key from https://dashboard.ngrok.com/get-started/your-authtoken"

... # imports from above

server = DiffusersServer.create(
    pretrained_path = STABLE_DIFFUSION_PATH,
    pipeline_cls = StableDiffusionPipeline,
    pipeline_kwargs = pipeline_kwargs,
    enable_attention_slicing = True,
    device = device
)

url = server.start(ngrok_auth_token=NGROK_AUTH_KEY)

# on another host
client = DiffusersClient(url)

responses = client(prompt='a photo of an astronaut riding a horse on mars', num_images_per_prompt=4) # this should return a list of images
```
