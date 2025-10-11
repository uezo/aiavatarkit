import base64
from logging import getLogger
from typing import Tuple, Dict, Union
from pathlib import Path
from uuid import uuid4
import aiofiles
from google import genai    # pip install google-genai
from google.genai import types
from aiavatar.sts.llm import Tool

logger = getLogger(__name__)


class NanoBanana:
    def __init__(
        self,
        gemini_api_key: str,
        model: str = "gemini-2.5-flash-image",
        system_instruction: str = None,
        id_prefix: str = None,
        timeout: int = 60000,
        get_image: callable = None,
        save_image: callable = None,
        save_dir: str = "nanobanana_images",
        debug: bool = False
    ):
        self.model = model
        self.system_instruction = system_instruction
        self.client = genai.Client(
            api_key=gemini_api_key,
            http_options=types.HttpOptions(timeout=timeout)
        )
        self.id_prefix =id_prefix or "nanobanana"
        self._get_image = get_image
        self._save_image = save_image
        self.save_dir = save_dir
        debug = debug

    async def generate_image(self, *, image_id: str = None, prompt: str = None, reference_image: Union[bytes, str] = None) -> Tuple[str, bytes]:
        if not image_id:
            image_id = f"{self.id_prefix}_{uuid4()}"

        parts = []
        if prompt:
            parts.append(types.Part.from_text(text=prompt))
        if reference_image:
            if isinstance(reference_image, bytes):
                parts.append(types.Part.from_bytes(data=reference_image, mime_type="image/png"))
            elif isinstance(reference_image, str):
                parts.append(types.Part.from_uri(file_uri=reference_image, mime_type="image/png"))
        contents = [types.Content(role="user", parts=parts)]

        generate_content_config = types.GenerateContentConfig(response_modalities=["IMAGE"])
        if self.system_instruction:
            generate_content_config.system_instruction = self.system_instruction

        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        )

        if resp.candidates and resp.candidates[0].content:
            for p in resp.candidates[0].content.parts:
                if p.inline_data and p.inline_data.data:
                    # Data type depends on environments :(
                    if p.inline_data.data.startswith(b"\x89PNG"):
                        return image_id, p.inline_data.data
                    else:
                        return image_id, base64.b64decode(p.inline_data.data)

        raise Exception("Image is not generated.")

    async def save_image(self, id: str, image_bytes: bytes):
        if self._save_image:
            await self._save_image(id, image_bytes)
        else:
            async with aiofiles.open(Path(self.save_dir) / f"{id}.png", "wb") as f:
                await f.write(image_bytes)

    async def read_image(self, id: str) -> bytes:
        if self._get_image:
            return await self._get_image(id)
        else:
            async with aiofiles.open(Path(self.save_dir) / f"{id}.png", "rb") as f:
                return await f.read()


class NanoBananaTool(Tool):
    def __init__(
        self,
        *,
        gemini_api_key: str,
        model: str = "gemini-2.5-flash-image",
        system_instruction: str = None,
        id_prefix: str = "image",
        timeout: int = 60000,
        reference_image: Union[bytes, str] = None,
        get_image: callable = None,
        save_image: callable = None,
        save_dir: str = "nanobanana_images",
        name: str = None,
        spec: str = None,
        instruction: str = None,
        is_dynamic: bool = False,
        debug: bool = False
    ):
        self.nanobanana = NanoBanana(
            gemini_api_key=gemini_api_key,
            model=model,
            system_instruction=system_instruction,
            id_prefix=id_prefix,
            timeout=timeout,
            get_image=get_image,
            save_image=save_image,
            save_dir=save_dir,
            debug=debug
        )
        self.reference_image = reference_image
        self.debug = debug

        super().__init__(
            name=name or "generate_image",
            spec=spec or {
                "type": "function",
                "function": {
                    "name": name or "generate_image",
                    "description": "Generate image by given prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Prompt to generate image."},
                            "reference_image_uri": {"type": "string", "description": "URI for reference image."},
                        },
                        "required": ["prompt"]
                    },
                }
            },
            func=self.generate_image,
            instruction=instruction,
            is_dynamic=is_dynamic
        )

    async def generate_image(self, prompt: str, reference_image_uri: str = None) -> Dict[str, str]:
        try:
            reference_image = reference_image_uri or self.reference_image
            id, image_bytes = await self.nanobanana.generate_image(
                prompt=prompt, reference_image=reference_image
            )
            await self.nanobanana.save_image(id, image_bytes)
            return {"image_id": id}
        except Exception as ex:
            logger.error(f"Error at generate_image: {ex}")
            return {"error": "Error at generate_image."}

    async def get_image(self, id: str) -> bytes:
        return await self.nanobanana.read_image(id)


class NanoBananaSelfieTool(NanoBananaTool):
    def __init__(
        self,
        *,
        gemini_api_key: str,
        model: str = "gemini-2.5-flash-image",
        system_instruction: str = None,
        id_prefix: str = "selfie",
        timeout: int = 60000,
        reference_image: Union[bytes, str] = None,
        get_image: callable = None,
        save_image: callable = None,
        save_dir: str = "nanobanana_images",
        name: str = None,
        spec: str = None,
        instruction: str = None,
        is_dynamic: bool = False,
        debug: bool = False
    ):
        super().__init__(
            gemini_api_key=gemini_api_key,
            model=model,
            system_instruction=system_instruction or """You are the selfie generator.

Please update the selfie image of the given character in the given situation according to the following rules.

## Rules

- Keep the selfie composition.
- Do not output smartphones or cameras. We want to output an image taken by the character themselves.
""",
            id_prefix=id_prefix,
            timeout=timeout,
            reference_image=reference_image,
            get_image=get_image,
            save_image=save_image,
            save_dir=save_dir,
            name=name or "generate_selfie",
            spec=spec or {
                "type": "function",
                "function": {
                    "name": name or "generate_selfie",
                    "description": "Generate selfie image by given situations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "face_expression": {"type": "string"},
                            "cloth": {"type": "string"},
                            "location_background": {"type": "string"},
                        },
                        "required": ["face_expression", "cloth", "location_background"]
                    },
                }
            },
            instruction=instruction,
            is_dynamic=is_dynamic,
            debug=debug
        )
        self.func = self.generate_selfie

    async def generate_selfie(self, face_expression: str, cloth: str, location_background: str) -> Dict[str, str]:
        try:
            prompt = f"""Modify this selfie image:

            - face expression: {face_expression}
            - cloth: {cloth}
            - location background: {location_background}
            """
            if self.debug:
                logger.info(f"Prompt for selfie: {prompt}")
            result = await self.generate_image(prompt)
            return {"selfie_id": result["image_id"]}
        except Exception as ex:
            logger.error(f"Error at generate_selfie: {ex}")
            return {"error": "Error at generate_selfie."}
