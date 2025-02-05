import cv2
import numpy as np
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from urllib.parse import urlparse
import logging
import json
from io import BytesIO
from dataclasses import dataclass


@dataclass
class TryOnDiffusionAPIResponse:
    status_code: int
    image: np.ndarray = None
    response_data: bytes = None
    error_details: str = None
    seed: int = None


class TryOnDiffusionClient:
    def __init__(self, base_url: str = "https://try-on-diffusion.p.rapidapi.com", api_key: str = "f46338f1a1msh8f27a3a69564667p1c5a31jsnbd2438a5d1c9"):
        self._logger = logging.getLogger("try_on_diffusion_client")
        self._base_url = base_url
        self._api_key = api_key

        if self._base_url[-1] == "/":
            self._base_url = self._base_url[:-1]

        parsed_url = urlparse(self._base_url)
        self._rapidapi_host = parsed_url.netloc if parsed_url.netloc.endswith(".rapidapi.com") else None

        if self._rapidapi_host is not None:
            self._logger.info(f"Using RapidAPI proxy: {self._rapidapi_host}")

    @staticmethod
    def _image_to_upload_file(image: np.ndarray) -> tuple:
        _, jpeg_data = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 99])
        jpeg_data = jpeg_data.tobytes()
        fp = BytesIO(jpeg_data)
        return "image.jpg", fp, "image/jpeg"

    def try_on_file(
        self,
        clothing_image: np.ndarray = None,
        clothing_prompt: str = None,
        avatar_image: np.ndarray = None,
        avatar_prompt: str = None,
        avatar_sex: str = None,
        background_image: np.ndarray = None,
        background_prompt: str = None,
        seed: int = -1,
        raw_response: bool = False,
    ) -> TryOnDiffusionAPIResponse:
        url = "https://try-on-diffusion.p.rapidapi.com/try-on-file"
        print(f"API URL: {url}")  # Debugging URL construction

        request_data = {"seed": str(seed)}

        if clothing_image is not None:
            request_data["clothing_image"] = self._image_to_upload_file(clothing_image)

        if clothing_prompt is not None:
            request_data["clothing_prompt"] = clothing_prompt

        if avatar_image is not None:
            request_data["avatar_image"] = self._image_to_upload_file(avatar_image)

        if avatar_prompt is not None:
            request_data["avatar_prompt"] = avatar_prompt

        if avatar_sex is not None:
            request_data["avatar_sex"] = avatar_sex

        if background_image is not None:
            request_data["background_image"] = self._image_to_upload_file(background_image)

        if background_prompt is not None:
            request_data["background_prompt"] = background_prompt

        multipart_data = MultipartEncoder(fields=request_data)
        headers = {"Content-Type": multipart_data.content_type}

        if self._rapidapi_host is not None:
            headers["X-RapidAPI-Key"] = self._api_key
            headers["X-RapidAPI-Host"] = self._rapidapi_host
        else:
            headers["X-API-Key"] = self._api_key

        try:
            response = requests.post(url, data=multipart_data, headers=headers)
        except Exception as e:
            self._logger.error(e, exc_info=True)
            return TryOnDiffusionAPIResponse(status_code=0)

        if response.status_code != 200:
            self._logger.warning(f"Request failed, status code: {response.status_code}, response: {response.content}")

        result = TryOnDiffusionAPIResponse(status_code=response.status_code)

        if not raw_response and response.status_code == 200:
            try:
                result.image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                self._logger.error(f"Error decoding image: {e}", exc_info=True)
                result.image = None
        else:
            result.response_data = response.content

        if result.status_code == 200:
            if "X-Seed" in response.headers:
                result.seed = int(response.headers["X-Seed"])
        else:
            try:
                response_json = (
                    json.loads(result.response_data.decode("utf-8")) if result.response_data is not None else None
                )
                if response_json is not None and "detail" in response_json:
                    result.error_details = response_json["detail"]
            except Exception as e:
                self._logger.error(f"Error parsing response JSON: {e}", exc_info=True)
                result.error_details = None

        return result
