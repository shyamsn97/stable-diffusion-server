from __future__ import annotations

import gc
from io import BytesIO
from typing import Any, Dict, Optional, Type

import nest_asyncio
import torch
import uvicorn
from diffusers import StableDiffusionPipeline

# from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pyngrok import ngrok

# from PIL import Image
# import numpy as np
# from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def clear_cuda_mem():
    """Try to recover from CUDA OOM"""
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception:
            pass

    gc.collect()
    torch.cuda.empty_cache()


def create_app(stable_diffusion_pipeline: StableDiffusionPipeline) -> FastAPI:
    app = FastAPI()
    prompt_image_dict = {}

    @app.get("/")
    async def read_root():
        return {"Hello": "World"}

    @app.get("/get_prompts")
    async def get_prompts():
        return list(prompt_image_dict.keys())

    @app.get("/get_images")
    async def get_images(prompt: str):
        if prompt not in prompt_image_dict:
            return None
        images = prompt_image_dict[prompt]
        if len(images) > 0:
            image = images.pop()
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, "JPEG")
            img_byte_arr.seek(0)
            return StreamingResponse(content=img_byte_arr, media_type="image/jpeg")
        del prompt_image_dict[prompt]
        return None

    @app.post("/pipeline")
    async def pipeline(input: Dict[str, Any]):
        prompt = input["prompt"]
        if prompt not in prompt_image_dict:
            prompt_image_dict[prompt] = []
        prompt_image_dict[prompt].extend(stable_diffusion_pipeline(**input).images)
        return len(prompt_image_dict[prompt])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    return app


class DiffusersServer:
    def __init__(
        self,
        pipeline: StableDiffusionPipeline,
        enable_attention_slicing: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        clear_cuda_mem()
        self.device = device

        self.pipeline = pipeline
        self.pipeline = pipeline.to(device)

        self.enable_attention_slicing = enable_attention_slicing

        if self.enable_attention_slicing:
            self.pipeline.enable_attention_slicing()

        self.app = None

    def to_device(self, device: torch.device):
        self.pipeline = self.pipeline.to(device)

    def create_app(self) -> FastAPI:
        return create_app(self.pipeline)

    @classmethod
    def create(
        cls,
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        pipeline_cls: Type[StableDiffusionPipeline] = StableDiffusionPipeline,
        pipeline_kwargs: Dict[str, Any] = {},
        enable_attention_slicing: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> DiffusersServer:
        pipeline = pipeline_cls.from_pretrained(pretrained_path, **pipeline_kwargs)
        return DiffusersServer(pipeline, enable_attention_slicing, device)

    def start(self, ngrok_auth_token: Optional[str] = None, port: int = 8000, host: str = '127.0.0.1') -> None:
        if self.app is None:
            self.app = self.create_app()

        url = f"http://{host}:{port}"
        if ngrok_auth_token is not None:
            ngrok.set_auth_token(ngrok_auth_token)
            ngrok_tunnel = ngrok.connect(port, bind_tls=True)
            print("Public URL:", ngrok_tunnel.public_url)
            url = ngrok_tunnel.public_url
        nest_asyncio.apply()
        uvicorn.run(self.app, port=port, host=host)
        return url

    def shutdown(self) -> None:
        self.app = None
