import logging
import gradio as gr
import numpy as np
import cv2
import os
import base64

from try_on_diffusion_client import TryOnDiffusionClient

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(thread)-8s %(name)-16s %(levelname)-8s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "examples")

API_URL = os.getenv("TRY_ON_DIFFUSION_DEMO_API_URL", "https://try-on-diffusion.p.rapidapi.com")
API_KEY = os.getenv("TRY_ON_DIFFUSION_DEMO_API_KEY", "32815c8a07msh6a1cb1815af30cfp10ee85jsn177a64ac277f")

SHOW_RAPIDAPI_LINK = os.getenv("TRY_ON_DIFFUSION_DEMO_SHOW_RAPIDAPI_LINK", "1") == "1"

CONCURRENCY_LIMIT = int(os.getenv("TRY_ON_DIFFUSION_DEMO_CONCURRENCY_LIMIT", "2"))

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

client = TryOnDiffusionClient(base_url=API_URL, api_key=API_KEY)


def get_image_base64(file_name: str) -> str:
    _, ext = os.path.splitext(file_name.lower())

    content_type = "image/jpeg"

    if ext == ".png":
        content_type = "image/png"
    elif ext == ".webp":
        content_type = "image/webp"
    elif ext == ".gif":
        content_type = "image/gif"

    with open(file_name, "rb") as f:
        return f"data:{content_type};base64," + base64.b64encode(f.read()).decode("utf-8")


def get_examples(example_dir: str) -> list[str]:
    file_list = [f for f in os.listdir(os.path.join(EXAMPLE_PATH, example_dir)) if f.endswith(".jpg")]
    file_list.sort()

    return [os.path.join(EXAMPLE_PATH, example_dir, f) for f in file_list]


def try_on(
    clothing_image: np.ndarray = None,
    clothing_prompt: str = None,
    avatar_image: np.ndarray = None,
    avatar_prompt: str = None,
    avatar_sex: str = None,
    background_image: np.ndarray = None,
    background_prompt: str = None,
    seed: int = -1,
) -> tuple:
    result = client.try_on_file(
        clothing_image=cv2.cvtColor(clothing_image, cv2.COLOR_RGB2BGR) if clothing_image is not None else None,
        clothing_prompt=clothing_prompt,
        avatar_image=cv2.cvtColor(avatar_image, cv2.COLOR_RGB2BGR) if avatar_image is not None else None,
        avatar_prompt=avatar_prompt,
        avatar_sex=avatar_sex if avatar_sex in ["male", "female"] else None,
        background_image=cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) if background_image is not None else None,
        background_prompt=background_prompt,
        seed=seed,
    )

    if result.status_code == 200:
        return cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB), f"<h3>Success</h3><p>Seed: {result.seed}</p>"
    else:
        error_message = f"<h3>Error {result.status_code}</h3>"

        if result.error_details is not None:
            error_message += f"<p>{result.error_details}</p>"

        return None, error_message


with gr.Blocks(theme=gr.themes.Soft(), delete_cache=(3600, 3600)) as app:
    gr.HTML(
         f"""
        <div style="width: 100%; background-color: #001537; border-radius: 10px; padding-left: 10px">
            <a href="https://www.ieee.org/" target="_blank">
                <img src="{get_image_base64("images/logo.png")}" title=LOGO" alt="IEEE" style="float: left; margin-right: 10px; margin-left: -10px; border-radius: 10px; max-height: 50px;"/>
            </a>
            <h1 style="margin: 0; margin-right: 10px; line-height: 50px; color: #8cecd5; text-transform: uppercase">Dressify-Virtual Try ON TSYP</h1>
        </div>
        <br/>
        <p>
            Virtual Try On technology is developed by IEEE Dressify Members . We present This prototype  
    """
    )

   

    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <h2>Clothing</h2>
                <p>
                    Clothing may be specified with a reference image or a text prompt. 
                    For more exotic use cases image and prompt can be also used together.
                    If both image and prompt are empty the model will generate random clothing.
                    <br/><br/>
                </p>
                """
            )

            with gr.Tab("Image"):
                clothing_image = gr.Image(label="Clothing Image", sources=["upload"], type="numpy")

                clothing_image_examples = gr.Examples(
                    inputs=clothing_image, examples_per_page=12, examples=get_examples("clothing")
                )

            with gr.Tab("Prompt"):
                clothing_prompt = gr.TextArea(
                    label="Clothing Prompt",
                    info='Compel weighting <a href="https://github.com/damian0815/compel/blob/main/doc/syntax.md">syntax</a> is supported.',
                )

                clothing_prompt_examples = gr.Examples(
                    inputs=clothing_prompt,
                    examples_per_page=8,
                    examples=[
                        "a sheer blue sleeveless mini dress",
                        "a beige woolen sweater and white pleated skirt",
                        "a black leather jacket and dark blue slim-fit jeans",
                        "a floral pattern blouse and leggings",
                        "a paisley pattern purple shirt and beige chinos",
                        "a striped white and blue polo shirt and blue jeans",
                        "a colorful t-shirt and black shorts",
                        "a checked pattern shirt and dark blue cargo pants",
                    ],
                )

        with gr.Column():
            gr.HTML(
                """
                <h2>Avatar</h2>
                <p>
                    Avatar may be specified with a subject photo or a text prompt.
                    Latter can be used, for example, to replace person while preserving clothing. 
                    For more exotic use cases image and prompt can be also used together.
                    If both image and prompt are empty the model will generate random avatars.
                </p>
                """
            )

            with gr.Tab("Image"):
                avatar_image = gr.Image(label="Avatar Image", sources=["upload"], type="numpy")

                avatar_image_examples = gr.Examples(
                    inputs=avatar_image,
                    examples_per_page=12,
                    examples=get_examples("avatar"),
                )

            with gr.Tab("Prompt"):
                avatar_prompt = gr.TextArea(
                    label="Avatar Prompt",
                    info='Compel weighting <a href="https://github.com/damian0815/compel/blob/main/doc/syntax.md">syntax</a> is supported.',
                )

                avatar_prompt_examples = gr.Examples(
                    inputs=avatar_prompt,
                    examples_per_page=8,
                    examples=[
                        "a beautiful blond girl with long hair",
                        "a cute redhead girl with freckles",
                        "a plus size female model wearing sunglasses",
                        "a woman with dark hair and blue eyes",
                        "a fit man with dark beard and blue eyes",
                        "a young blond man posing for a photo",
                        "a gentleman with beard and mustache",
                        "a plus size man walking",
                    ],
                )

            avatar_sex = gr.Dropdown(
                label="Avatar Sex",
                choices=[("Auto", ""), ("Male", "male"), ("Female", "female")],
                value="",
                info="Avatar sex selector can be used to enforce a specific sex of the avatar.",
            )

        with gr.Column():
            gr.HTML(
                """
                <h2>Background</h2>
                <p>
                    Replacing background is optional. 
                    Background may be specified with a reference image or a text prompt.
                    If omitted original avatar background will be preserved.
                    <br/><br/><br/>
                </p>
                """
            )

            with gr.Tab("Image"):
                background_image = gr.Image(label="Background Image", sources=["upload"], type="numpy")

                background_image_examples = gr.Examples(
                    inputs=background_image, examples_per_page=12, examples=get_examples("background")
                )

            with gr.Tab("Prompt"):
                background_prompt = gr.TextArea(
                    label="Background Prompt",
                    info='Compel weighting <a href="https://github.com/damian0815/compel/blob/main/doc/syntax.md">syntax</a> is supported.',
                )

                background_prompt_examples = gr.Examples(
                    inputs=background_prompt,
                    examples_per_page=8,
                    examples=[
                        "in an autumn park",
                        "in front of a brick wall",
                        "near an old tree",
                        "on a busy city street",
                        "in front of a staircase",
                        "on an ocean beach with palm trees",
                        "in a shopping mall",
                        "in a modern office",
                    ],
                )

        with gr.Column():
            gr.HTML(
                """
                <h2>Generation</h2>
                """
            )

            seed = gr.Number(
                label="Seed",
                value=-1,
                minimum=-1,
                info="Seed used for generation, specify -1 for random seed for each generation.",
            )

            generate_button = gr.Button(value="Generate", variant="primary")

            result_image = gr.Image(label="Result", show_share_button=False, format="jpeg")
            result_details = gr.HTML(label="Details")

    generate_button.click(
        fn=try_on,
        inputs=[
            clothing_image,
            clothing_prompt,
            avatar_image,
            avatar_prompt,
            avatar_sex,
            background_image,
            background_prompt,
            seed,
        ],
        outputs=[result_image, result_details],
        api_name=False,
        concurrency_limit=CONCURRENCY_LIMIT,
    )

    app.title = "Virtual Try-On Diffusion by IEEE Dressify"


if __name__ == "__main__":
    app.queue(api_open=False).launch(show_api=False)
