# Virtual Try-On Diffusion API

<!-- TOC -->
* [Virtual Try-On Diffusion API](#virtual-try-on-diffusion-api)
  * [Summary](#summary)
  * [Consuming the API](#consuming-the-api)
  * [Try-On Endpoints](#try-on-endpoints)
  * [Try-On Input Parameters](#try-on-input-parameters)
    * [Clothing image](#clothing-image)
    * [Clothing prompt](#clothing-prompt)
    * [Avatar image](#avatar-image)
    * [Avatar prompt](#avatar-prompt)
    * [Background image](#background-image)
    * [Background prompt](#background-prompt)
    * [Additional notes](#additional-notes)
  * [Try-On Output](#try-on-output)
    * [Response codes](#response-codes)
    * [NSFW content](#nsfw-content)
  * [Use Cases and Recipes](#use-cases-and-recipes)
    * [Image-based virtual try-on](#image-based-virtual-try-on)
    * [Image-based virtual try-on with background](#image-based-virtual-try-on-with-background)
    * [Avatar from a text prompt](#avatar-from-a-text-prompt)
    * [Clothing from a text prompt](#clothing-from-a-text-prompt)
    * [Modifying avatar's body](#modifying-avatars-body)
    * [Txt2Img](#txt2img)
    * [Other creative possibilities](#other-creative-possibilities)
  * [Performance](#performance)
  * [Known Issues and Limitations](#known-issues-and-limitations)
<!-- TOC -->

## Summary

Virtual Try-On Diffusion [VTON-D] by [Texel.Moda](https://texelmoda.com) is a custom diffusion-based pipeline for fast 
and flexible multi-modal virtual try-on. Clothing, avatar and background can be specified by reference images or text 
prompts allowing for clothing transfer, avatar replacement, fashion image generation and other virtual try-on related 
tasks. Check out the [demo on Hugging Face](https://huggingface.co/spaces/texelmoda/try-on-diffusion) to try the API in
a user-friendly way.

## Consuming the API

The API is exposed through the RapidAPI Hub which manages API subscriptions, API keys, payments and other things. Please 
refer to the [RapidAPI Documentation](https://docs.rapidapi.com/docs/consumer-quick-start-guide) to get started.

Generally, in order to use the API you need to perform the following steps:
- Create a RapidAPI.com account.
- [Navigate to the API page](https://rapidapi.com/texelmoda-texelmoda-apis/api/try-on-diffusion) and subscribe to a 
  suitable pricing plan. We also provide a free BASIC plan with 100 API requests per month.
- Use the obtained RapidAPI key to authenticate (via the _X-RapidAPI-Key_ header) and use the API from any programming 
  language or tool you like.

Example API call using cURL:
```shell
curl --request POST \
--url https://try-on-diffusion.p.rapidapi.com/try-on-file \
--header 'Content-Type: multipart/form-data' \
--header 'x-rapidapi-host: try-on-diffusion.p.rapidapi.com' \
--header 'x-rapidapi-key: <RapidAPI Key>' \
--form clothing_image=1.jpg \
--form avatar_image=2.jpg
```

For a simple Python client implementation please see the 
[Hugging Face demo application source](https://huggingface.co/spaces/texelmoda/try-on-diffusion/blob/main/try_on_diffusion_client.py).

## Try-On Endpoints

Try-On API consists of two endpoints that differ only in the method of passing reference images:

- **POST** _/try-on-file_ - takes reference images as uploaded files in the request body (using multipart/form-data).


- **POST** _/try-on-url_ - takes reference images as image URLs in POST parameters.

All image requirements, behavior and status codes are the same for both endpoints, choose the one that best suits your 
application architecture.

## Try-On Input Parameters

All input parameters for the try-on endpoints are currently optional. Images and prompts serve as additional generation 
conditions and can even be used in combination. Below is the short parameter summary with links to extended information 
on certain parameters.

List of input parameters for the **POST** _/try-on-file_ endpoint:

| Parameter                               | Description                                                                                                                                                                          | Required |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| [clothing_image](#clothing-image)       | Clothing reference image in JPEG, PNG or WEBP format, maximum file size is 12 MB.                                                                                                    | No       |
| [clothing_prompt](#clothing-prompt)     | Text prompt for clothing, can be used instead of an image. Compel weighting syntax is supported. Example: _red sleeveless mini dress_                                                | No       |
| [avatar_image](#avatar-image)           | Avatar image in JPEG, PNG or WEBP format, maximum file size is 12 MB.                                                                                                                | No       |
| avatar_sex                              | Avatar sex, either "male" or "female". Will be detected automatically, if left empty or omitted. Will enforce certain avatar sex if specified.                                       | No       |
| [avatar_prompt](#avatar-prompt)         | Text prompt for the avatar, can be used instead of an image or with image to modify the avatar. Compel weighting syntax is supported. Example: _a gentleman with beard and mustache_ | No       |
| [background_image](#background-image)   | Optional background reference image in JPEG, PNG or WEBP format, maximum file size is 12 MB. Original avatar background is preserved if background is not specified.                 | No       |
| [background_prompt](#background-prompt) | Optional background text prompt. Original avatar background is preserved if background is not specified. Example: _in an autumn park_                                                | No       |
| seed                                    | Seed for image generation. Default is -1 (random seed). Actual seed will also be output in the "X-Seed" response header. Example: _42_                                               | No       |

List of input parameters for the **POST** _/try-on-url_ endpoint:

| Parameter                                 | Description                                                                                                                                                                               | Required |
|-------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| [clothing_image_url](#clothing-image)     | Clothing reference image URL. Image should be in JPEG, PNG or WEBP format, maximum file size is 12 MB.                                                                                    | No       |
| [clothing_prompt](#clothing-prompt)       | Text prompt for clothing, can be used instead of an image. Compel weighting syntax is supported. Example: _red sleeveless mini dress_                                                     | No       |
| [avatar_image_url](#avatar-image)         | Avatar image URL. Image should be in JPEG, PNG or WEBP format, maximum file size is 12 MB.                                                                                                | No       |
| avatar_sex                                | Avatar sex, either "male" or "female". Will be detected automatically, if left empty or omitted. Will enforce certain avatar sex if specified.                                            | No       |
| [avatar_prompt](#avatar-prompt)           | Text prompt for the avatar, can be used instead of an image or with image to modify the avatar. Compel weighting syntax is supported. Example: _a gentleman with beard and mustache_      | No       |
| [background_image_url](#background-image) | Optional background reference image URL. Image should be in JPEG, PNG or WEBP format, maximum file size is 12 MB. Original avatar background is preserved if background is not specified. | No       |
| [background_prompt](#background-prompt)   | Optional background text prompt. Original avatar background is preserved if background is not specified. Example: _in an autumn park_                                                     | No       |
| seed                                      | Seed for image generation. Default is -1 (random seed). Actual seed will also be output in the "X-Seed" response header. Example: _42_                                                    | No       |

### Clothing image

For best results clothing reference images should meet a number of requirements:

- File format: **JPEG**, **PNG** or **WEBP**
- Maximum file size: **12 MB**
- Minimum image size: **256x256**
- Recommended image size: **768x1024 and above**
- Clothing should be **dressed on a person**. Some flat lay clothing photos might work, but currently it's not guaranteed
- **Single person** on the image (though multiple persons might also work)
- **Frontal** photo, though some degree of rotation is fine
- **Good lighting** conditions and **high image quality** as it directly affects the result
- **Minimal occlusion** by hair, hands or accessories

To summarize: the better is the clothing image the better is the final result.

Examples of good clothing images:

| <img src="images/clothing_image_01.jpg" width="240"> | <img src="images/clothing_image_02.jpg" width="240"> | <img src="images/clothing_image_03.jpg" width="240"> | <img src="images/clothing_image_04.jpg" width="240"> |
|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|

### Clothing prompt

Instead of a clothing image you can use text prompt to describe the garment. Short and clear prompts work best. 
Additionally, [Compel weighting syntax](https://github.com/damian0815/compel/blob/main/doc/syntax.md) is supported to 
increase or decrease weight of certain tokens. Examples:
- _a sheer blue sleeveless mini dress_
- _a beige woolen sweater and white pleated skirt_
- _a black leather jacket and dark blue slim-fit jeans_
- _a floral pattern blouse and leggings_
- _a colorful+++ t-shirt and black shorts_

### Avatar image

Avatar images should also meet a some requirements:

- File format: **JPEG**, **PNG** or **WEBP**
- Maximum file size: **12 MB**
- Minimum image size: **256x256**
- Recommended image size: **768x1024 and above**
- **Single person** on the image (though multiple persons might also work)
- **Frontal** photo, though some degree of rotation is fine
- **Good lighting** conditions and **high image quality**

Examples of good avatar images:

| <img src="images/avatar_image_01.jpg" width="240"> | <img src="images/avatar_image_02.jpg" width="240"> | <img src="images/avatar_image_03.jpg" width="240"> | <img src="images/avatar_image_04.jpg" width="240"> |
|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|

### Avatar prompt

Instead of an avatar image you can use text prompt to describe the person. Short and clear prompts work best. 
Additionally, [Compel weighting syntax](https://github.com/damian0815/compel/blob/main/doc/syntax.md) is supported to 
increase or decrease weight of certain tokens. Examples:
- _a beautiful blond girl with long hair_
- _a cute redhead girl with freckles_
- _a (plus size)++ female model wearing sunglasses_
- _a fit man with dark beard and blue eyes_
- _a gentleman with beard and mustache_

### Background image

Background images are used to extract high-level background features only and serve as a reference (and not exact 
background). Below are basic image requirements:

- File format: **JPEG**, **PNG** or **WEBP**
- Maximum file size: **12 MB**
- Recommended image size: **256x256 and above**

Examples of background images:

| <img src="images/background_image_01.jpg" width="240"> | <img src="images/background_image_02.jpg" width="240"> | <img src="images/background_image_03.jpg" width="240"> | <img src="images/background_image_04.jpg" width="240"> |
|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|

### Background prompt

Instead of a background image you can use text prompt to describe the background. Short and clear prompts work best. 
Additionally, [Compel weighting syntax](https://github.com/damian0815/compel/blob/main/doc/syntax.md) is supported to 
increase or decrease weight of certain tokens. Examples:
- _in an autumn park_
- _in front of a brick wall_
- _on an ocean beach with (palm trees)++_
- _in a shopping mall_
- _in a modern office_

### Additional notes

We use the "same-crop" approach for clothing and avatar images: images will be cropped roughly the same way (using pose 
estimation), so we don't have to add too much new information (e.g. assume lower body clothing). So, if you use only a 
photo of an upper body clothing the result will also be cropped the same way regardless of the avatar image (and the 
other way around):

| Clothing Image                                       | Avatar Image                                        | Result Image                                           |
|------------------------------------------------------|-----------------------------------------------------|--------------------------------------------------------|
| <img src="images/clothing_image_02.jpg" width="240"> | <img src="images/avatar_image_02.jpg" width="240">  | <img src="images/same_crop_result_01.jpg" width="240"> |
| <img src="images/clothing_image_03.jpg" width="240"> | <img src="images/avatar_image_03.jpg" width="240">  | <img src="images/same_crop_result_02.jpg" width="240"> |

## Try-On Output

### Response codes

HTTP status code is used as a high-level response status. In case of a successful API call HTTP code 200 will be 
returned and response body will contain a resulting JPEG image with the maximum size of 768x1024 pixels. Response
will also have the "X-Seed" header set that should contain the actual seed used for image generation (for 
reproducibility). Other status codes (not 200) indicate unsuccessful request, see the table below for additional 
details:

| Response Code |    Content-Type    |    Headers     | Description                                                                                                                       |                                                    Example                                                    |
|:-------------:|:------------------:|:--------------:|-----------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------:|
|    **200**    |     image/jpeg     | X-Seed: {seed} | Successful API call. Response body contains the resulting image in JPEG format.                                                   |                            <img src="images/same_crop_result_01.jpg" width="160">                             |
|    **400**    |  application/json  |                | Bad request: at least one of request parameters is invalid. Response body should contain additional error details in JSON format. |                    { "detail": "Invalid upload file type: application/x-zip-compressed" }                     |
|    **403**    |  application/json  |                | Indicates authentication issue (e.g. invalid API key).                                                                            |                                                                                                               |
|    **422**    |  application/json  |                | Request validation error. Response body should contain error details in JSON format.                                              |                { "detail": [ { "loc": [ "string", 0], "msg": "string", "type": "string" } ] }                 |
|    **429**    |                    |                | Too many requests. Might be triggered by the RapidAPI proxy in case of reaching maximum request rate or API call limit.           |                                                                                                               |
|    **500**    |                    |                | Indicates an internal server error, might not have any details.                                                                   |                                                                                                               |

### NSFW content

We use NSFW content checker to ensure we don't output inappropriate images. If potential NSFW content is detected in the
generated image, the API will return HTTP status code 400 with a corresponding error message in JSON response.

## Use Cases and Recipes

Our Virtual Try-On API offers a flexible way to specify clothing, avatar and background, which makes it possible to not
only perform a classic task of virtual try-on, but also generate entirely new images or alter existing images in some
interesting aspects. Feel free to try and explore!

In all the examples below all unmentioned inputs are assumed to be empty.

### Image-based virtual try-on

The most common use case is to transfer clothing from one photo (e.g. from a product page) to another photo (e.g. 
user avatar) while maintaining the avatar and the background.

| Clothing Image                                       | Avatar Image                                       | Result Image                                             |
|------------------------------------------------------|----------------------------------------------------|----------------------------------------------------------|
| <img src="images/clothing_image_01.jpg" width="240"> | <img src="images/avatar_image_02.jpg" width="240"> | <img src="images/image_based_result_01.jpg" width="240"> |

### Image-based virtual try-on with background

Additionally, it's possible to replace the avatar background with a reference image or a text prompt.

| Clothing Image                                       | Avatar Image                                       | Background Image                                       | Result Image                                                        |
|------------------------------------------------------|----------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------|
| <img src="images/clothing_image_04.jpg" width="240"> | <img src="images/avatar_image_03.jpg" width="240"> | <img src="images/background_image_01.jpg" width="240"> | <img src="images/image_based_background_result_01.jpg" width="240"> |

And with a text prompt for the background:

| Clothing Image                                       | Avatar Image                                       | Background Prompt            | Result Image                                                        |
|------------------------------------------------------|----------------------------------------------------|------------------------------|---------------------------------------------------------------------|
| <img src="images/clothing_image_04.jpg" width="240"> | <img src="images/avatar_image_03.jpg" width="240"> | in front of a snowy mountain | <img src="images/image_based_background_result_02.jpg" width="240"> |

### Avatar from a text prompt

It's possible to replace the person on the clothing image with an avatar, described in a text prompt. Background will be 
changed as well and will be a random one if not specified:

| Clothing Image                                       | Avatar Prompt                              | Background Prompt  | Result Image                                               |
|------------------------------------------------------|--------------------------------------------|--------------------|------------------------------------------------------------|
| <img src="images/clothing_image_02.jpg" width="240"> | a beautiful blond girl with long hair      |                    | <img src="images/avatar_prompt_result_01.jpg" width="240"> |
| <img src="images/clothing_image_03.jpg" width="240"> | a gentleman with a long beard and mustache | near a fireplace   | <img src="images/avatar_prompt_result_02.jpg" width="240"> |

You may also experiment with avatar prompts for more interesting results:

| Clothing Image                                       | Avatar Prompt       | Background Prompt     | Result Image                                               |
|------------------------------------------------------|---------------------|-----------------------|------------------------------------------------------------|
| <img src="images/clothing_image_03.jpg" width="240"> | (iron man mask)+++  | in the Sahara Desert  | <img src="images/avatar_prompt_result_03.jpg" width="240"> |

### Clothing from a text prompt

Similarly, you can specify clothing with a text prompt while providing an avatar image:

| Clothing Prompt                     | Avatar Image                                       | Result Image                                                 |
|-------------------------------------|----------------------------------------------------|--------------------------------------------------------------|
| a sheer blue sleeveless mini dress  | <img src="images/avatar_image_02.jpg" width="240"> | <img src="images/clothing_prompt_result_01.jpg" width="240"> |
| a colorful t-shirt and black shorts | <img src="images/avatar_image_03.jpg" width="240"> | <img src="images/clothing_prompt_result_02.jpg" width="240"> |

### Modifying avatar's body

If you specify clothing and avatar images to be the same while providing an avatar prompt it's possible to change
avatar's body proportions. Note that it may require using additional term weighting to achieve stronger changes.

| Clothing Image                                       | Avatar Image                                         | Avatar Prompt                 | Result Image                                                     |
|------------------------------------------------------|------------------------------------------------------|-------------------------------|------------------------------------------------------------------|
| <img src="images/clothing_image_01.jpg" width="240"> | <img src="images/clothing_image_01.jpg" width="240"> | a (plus size)+ woman          | <img src="images/avatar_modification_result_01.jpg" width="240"> |
| <img src="images/clothing_image_03.jpg" width="240"> | <img src="images/clothing_image_03.jpg" width="240"> | a (muscular bodybuilder)+++++ | <img src="images/avatar_modification_result_02.jpg" width="240"> |

### Txt2Img

As our diffusion model was fine-tuned to produce people wearing various clothing, it can better follow a clothing prompt 
and output realistic people and garments:

| Clothing Prompt                                 | Avatar Prompt                  | Background Prompt      | Result Image                                         |
|-------------------------------------------------|--------------------------------|------------------------|------------------------------------------------------|
| a paisley pattern purple shirt and beige chinos | a fit man with dark beard      | plain white background | <img src="images/txt2img_result_01.jpg" width="240"> |
| a white polka dot pattern dress                 | a beautiful petite blond woman | on a yacht             | <img src="images/txt2img_result_02.jpg" width="240"> |

### Other creative possibilities

If you specify the same image for clothing and avatar while providing a background prompt (or background image) you can
replace the background in a creative way:

| Clothing Image                                     | Avatar Image                                       | Background Prompt       | Result Image                                                |
|----------------------------------------------------|----------------------------------------------------|-------------------------|-------------------------------------------------------------|
| <img src="images/avatar_image_02.jpg" width="240"> | <img src="images/avatar_image_02.jpg" width="240"> | on a snowy mountain top | <img src="images/new_background_result_01.jpg" width="240"> |

It's also possible to use a combination of clothing image, clothing prompt, avatar image and a background to add some
accessories:

| Clothing Image                                       | Clothing Prompt          | Avatar Image                                         | Background Image                                       | Result Image                                                     |
|------------------------------------------------------|--------------------------|------------------------------------------------------|--------------------------------------------------------|------------------------------------------------------------------|
| <img src="images/avatar_image_02.jpg" width="240">   | a (light brown purse)+++ | <img src="images/avatar_image_02.jpg" width="240">   | <img src="images/background_image_03.jpg" width="240"> | <img src="images/accessory_result_01.jpg" width="240">           |

## Performance

Typically, one try-on request is processed in 5-10 seconds (depending on type of conditions) excluding network latency.
In order to reduce network overhead you might want to compress your images before feeding to the API (e.g. using JPEG). 
Please note that in case of a high demand processing time might increase due to request being queued, though we 
constantly monitor our GPU cluster capacity and perform scaling as needed.

## Known Issues and Limitations

As any generative model, our models are not perfect (though we constantly work on improvements):
- Currently, we do not fully support flat lay clothing images. Some might work, but that's not guaranteed.
- Prompt following might not be perfect, especially in case of long and sophisticated prompts. Prefer simpler and more
  straightforward prompts whenever possible. Also be pretty verbose (e.g. use the word "plain" if you need something of
  solid color). Additionally, Compel weighting might be used to increase weight of certain tokens.
- As usual, generative models struggle with hands, fingers and toes, though we try to mitigate it to a certain extent.
- Currently, we do not support trying on a single garment, only the full look.
- Hats and sunglasses are not currently transferred, but we are working on it.
- Backgrounds might lack some clarity as currently we focus more on clothing.
- In case of a specified background a hairstyle might change.
- Body shape of the avatar might change towards smaller sizes.
