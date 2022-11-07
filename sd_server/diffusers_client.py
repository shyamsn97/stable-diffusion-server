import io

import requests
from PIL import Image


def process_images(responses):
    images = []
    for response in responses:
        image = Image.open(io.BytesIO(response.content))
        images.append(image)
    return images


class DiffusersClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def __call__(self, **kwargs):
        prompt = kwargs.get("prompt")
        _ = self.post(**kwargs)
        return self.get(prompt)

    def post(self, **kwargs):
        headers = {
            "accept": "application/json",
            # Already added when you pass json=
        }

        json_data = kwargs

        response = requests.post(
            f"{self.endpoint}/pipeline", headers=headers, json=json_data
        )
        return {"num_images": int(response.content)}

    def get(self, prompt: str):
        headers = {
            "accept": "application/json",
            # Already added when you pass json=
        }
        params = {"prompt": prompt}
        has_images = True
        responses = []

        while has_images:
            response = requests.get(
                f"{self.endpoint}/get_image", params=params, headers=headers
            )
            if response.content == b"null":
                has_images = False
            else:
                responses.append(response)
        return process_images(responses)
