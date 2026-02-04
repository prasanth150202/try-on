import logging
import os
import cv2
import argparse
from pathlib import Path
from datetime import datetime
import sys
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import shutil
import requests
import numpy as np

from try_on_diffusion_client import TryOnDiffusionClient

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(name)-16s %(levelname)-8s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# API Configuration
API_URL = os.getenv("TRY_ON_DIFFUSION_DEMO_API_URL", "https://try-on-diffusion.p.rapidapi.com")
API_KEY = os.getenv("TRY_ON_DIFFUSION_DEMO_API_KEY", "32815c8a07msh6a1cb1815af30cfp10ee85jsn177a64ac277f")

# Initialize client
client = TryOnDiffusionClient(base_url=API_URL, api_key=API_KEY)

# Initialize FastAPI app
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

app = FastAPI(
    title="Virtual Try-On API",
    description="API for virtual clothing try-on using diffusion models",
    version="1.0.0"
)

# Serve results folder
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


def validate_image(image_path: str) -> bool:
    """Validate if image file exists and is readable."""
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return False
    
    if not os.path.isfile(image_path):
        logger.error(f"Path is not a file: {image_path}")
        return False
    
    return True


def load_image(image_path: str):
    """Load image using OpenCV."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        logger.info(f"Successfully loaded image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def load_image_from_url(url: str):
    """Download and load image from URL."""
    try:
        logger.info(f"Downloading image from URL: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Convert to numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Failed to decode image from URL: {url}")
            return None
        
        logger.info(f"Successfully loaded image from URL")
        return image
    except Exception as e:
        logger.error(f"Error loading image from URL {url}: {e}")
        return None


def create_output_folder(output_folder: str) -> bool:
    """Create output folder if it doesn't exist."""
    try:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder ready: {output_folder}")
        return True
    except Exception as e:
        logger.error(f"Failed to create output folder {output_folder}: {e}")
        return False


def save_image(image, output_path: str) -> bool:
    """Save image to specified path."""
    try:
        success = cv2.imwrite(output_path, image)
        if success:
            logger.info(f"Image saved successfully: {output_path}")
            return True
        else:
            logger.error(f"Failed to save image: {output_path}")
            return False
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False


def process_try_on(
    clothing_image_path: str,
    avatar_image_path: str,
    output_folder: str = None,
    clothing_prompt: str = None,
    avatar_prompt: str = None,
    avatar_sex: str = None,
    background_image_path: str = None,
    background_prompt: str = None,
    seed: int = -1,
    output_filename: str = None,
) -> dict:
    """
    Process virtual try-on and save result.
    
    Args:
        clothing_image_path: Path to clothing/product image
        avatar_image_path: Path to avatar image
        output_folder: Folder to save results (default: ./results)
        clothing_prompt: Optional text description for clothing
        avatar_prompt: Optional text description for avatar
        avatar_sex: Avatar sex ("male", "female", or None for auto)
        background_image_path: Optional path to background image
        background_prompt: Optional text description for background
        seed: Random seed (-1 for random)
        output_filename: Custom output filename (without extension)
    
    Returns:
        dict: Result dictionary with status, output_path, and details
    """
    
    logger.info("=" * 60)
    logger.info("Starting Try-On Processing")
    logger.info("=" * 60)
    
    result = {
        "status": "failed",
        "output_path": None,
        "error": None,
        "seed": None,
    }
    
    # Validate input images
    logger.info("Validating input images...")
    if not validate_image(clothing_image_path):
        result["error"] = f"Invalid clothing image: {clothing_image_path}"
        logger.error(result["error"])
        return result
    
    if not validate_image(avatar_image_path):
        result["error"] = f"Invalid avatar image: {avatar_image_path}"
        logger.error(result["error"])
        return result
    
    # Load images
    logger.info("Loading images...")
    clothing_image = load_image(clothing_image_path)
    avatar_image = load_image(avatar_image_path)
    background_image = None
    
    if clothing_image is None or avatar_image is None:
        result["error"] = "Failed to load one or more images"
        logger.error(result["error"])
        return result
    
    if background_image_path and validate_image(background_image_path):
        background_image = load_image(background_image_path)
    
    # Create output folder
    if output_folder is None:
        output_folder = str(RESULTS_DIR)
    if not create_output_folder(output_folder):
        result["error"] = f"Failed to create output folder: {output_folder}"
        logger.error(result["error"])
        return result
    
    # Call API
    logger.info("Calling Try-On API...")
    try:
        api_result = client.try_on_file(
            clothing_image=clothing_image,
            clothing_prompt=clothing_prompt,
            avatar_image=avatar_image,
            avatar_prompt=avatar_prompt,
            avatar_sex=avatar_sex if avatar_sex in ["male", "female"] else None,
            background_image=background_image,
            background_prompt=background_prompt,
            seed=seed,
        )
        
        if api_result.status_code != 200:
            result["error"] = f"API Error {api_result.status_code}: {api_result.error_details}"
            logger.error(result["error"])
            return result
        
        logger.info(f"API call successful! Seed: {api_result.seed}")
        result["seed"] = api_result.seed
        
        # Generate output filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"tryon_result_{timestamp}"
        
        output_path = os.path.join(output_folder, f"{output_filename}.jpg")
        
        # Save result image
        logger.info(f"Saving result image...")
        if save_image(api_result.image, output_path):
            result["status"] = "success"
            result["output_path"] = os.path.abspath(output_path)
            logger.info("=" * 60)
            logger.info(f"✓ Try-On Processing Completed Successfully!")
            logger.info(f"  Output: {result['output_path']}")
            logger.info(f"  Seed: {result['seed']}")
            logger.info("=" * 60)
        else:
            result["error"] = f"Failed to save image to {output_path}"
            logger.error(result["error"])
        
    except Exception as e:
        result["error"] = f"Exception during API call: {str(e)}"
        logger.error(result["error"], exc_info=True)
    
    return result


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Virtual Try-On Image Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api.py --clothing product.jpg --avatar avatar.jpg
  python api.py --clothing product.jpg --avatar avatar.jpg --output ./my_results
  python api.py --clothing product.jpg --avatar avatar.jpg --seed 42
  python api.py --clothing product.jpg --avatar avatar.jpg --clothing-prompt "red dress" --avatar-prompt "woman"
        """
    )
    
    parser.add_argument(
        "--clothing",
        required=True,
        help="Path to clothing/product image",
        metavar="PATH"
    )
    parser.add_argument(
        "--avatar",
        required=True,
        help="Path to avatar image",
        metavar="PATH"
    )
    parser.add_argument(
        "--output",
        default="./results",
        help="Output folder (default: ./results)",
        metavar="PATH"
    )
    parser.add_argument(
        "--output-filename",
        help="Custom output filename (without extension)",
        metavar="NAME"
    )
    parser.add_argument(
        "--clothing-prompt",
        help="Text description for clothing",
        metavar="TEXT"
    )
    parser.add_argument(
        "--avatar-prompt",
        help="Text description for avatar",
        metavar="TEXT"
    )
    parser.add_argument(
        "--avatar-sex",
        choices=["male", "female"],
        help="Avatar sex (male/female)",
        metavar="SEX"
    )
    parser.add_argument(
        "--background",
        help="Path to background image",
        metavar="PATH"
    )
    parser.add_argument(
        "--background-prompt",
        help="Text description for background",
        metavar="TEXT"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random, default: -1)",
        metavar="INT"
    )
    
    args = parser.parse_args()
    
    # Process try-on
    result = process_try_on(
        clothing_image_path=args.clothing,
        avatar_image_path=args.avatar,
        output_folder=args.output,
        clothing_prompt=args.clothing_prompt,
        avatar_prompt=args.avatar_prompt,
        avatar_sex=args.avatar_sex,
        background_image_path=args.background,
        background_prompt=args.background_prompt,
        seed=args.seed,
        output_filename=args.output_filename,
    )
    
    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)


# ==================== FastAPI Endpoints ====================

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Virtual Try-On API",
        "endpoints": {
            "GET /process": "Process try-on with image URLs",
            "POST /process": "Process try-on with uploaded images",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/process")
async def process_tryon_url(
    request: Request,
    clothing_image: str = Query(..., description="URL to clothing/product image"),
    avatar_image: str = Query(..., description="URL to avatar image"),
    clothing_prompt: Optional[str] = Query(None, description="Text description for clothing"),
    avatar_prompt: Optional[str] = Query(None, description="Text description for avatar"),
    avatar_sex: Optional[str] = Query(None, description="Avatar sex: 'male' or 'female'"),
    background_image: Optional[str] = Query(None, description="URL to background image (optional)"),
    background_prompt: Optional[str] = Query(None, description="Text description for background"),
    seed: int = Query(-1, description="Random seed (-1 for random)"),
):
    """
    Process virtual try-on with image URLs.
    
    Returns: Generated try-on image
    """
    
    logger.info("=" * 60)
    logger.info("API Request: Processing Try-On (URL-based)")
    logger.info("=" * 60)
    
    try:
        # Download images from URLs
        logger.info("Downloading images from URLs...")
        
        clothing_img = load_image_from_url(clothing_image)
        if clothing_img is None:
            raise HTTPException(status_code=400, detail=f"Failed to load clothing image from URL: {clothing_image}")
        
        avatar_img = load_image_from_url(avatar_image)
        if avatar_img is None:
            raise HTTPException(status_code=400, detail=f"Failed to load avatar image from URL: {avatar_image}")
        
        background_img = None
        if background_image:
            background_img = load_image_from_url(background_image)
            if background_img is None:
                logger.warning(f"Failed to load background image, continuing without it")
        
        # Call API directly with images
        logger.info("Calling Try-On API...")
        api_result = client.try_on_file(
            clothing_image=clothing_img,
            clothing_prompt=clothing_prompt,
            avatar_image=avatar_img,
            avatar_prompt=avatar_prompt,
            avatar_sex=avatar_sex if avatar_sex in ["male", "female"] else None,
            background_image=background_img,
            background_prompt=background_prompt,
            seed=seed,
        )
        
        if api_result.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"API Error {api_result.status_code}: {api_result.error_details}"
            )
        
        logger.info(f"API call successful! Seed: {api_result.seed}")
        
        # Save result
        output_folder = "./results"
        create_output_folder(output_folder)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"tryon_result_{timestamp}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        
        if not save_image(api_result.image, output_path):
            raise HTTPException(status_code=500, detail="Failed to save result image")
        
        logger.info(f"Returning image: {output_path}")
        
        base_url = str(request.base_url).rstrip("/")
        return {
            "result_image": f"{base_url}/results/{output_filename}",
            "seed": api_result.seed
        }

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/process")
async def process_tryon_api(
    request: Request,
    clothing_image: UploadFile = File(..., description="Clothing/Product image"),
    avatar_image: UploadFile = File(..., description="Avatar image"),
    clothing_prompt: Optional[str] = Form(None, description="Text description for clothing"),
    avatar_prompt: Optional[str] = Form(None, description="Text description for avatar"),
    avatar_sex: Optional[str] = Form(None, description="Avatar sex: 'male' or 'female'"),
    background_image: Optional[UploadFile] = File(None, description="Background image (optional)"),
    background_prompt: Optional[str] = Form(None, description="Text description for background"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
):
    """
    Process virtual try-on with uploaded images.
    
    Returns: Generated try-on image
    """
    
    logger.info("=" * 60)
    logger.info("API Request: Processing Try-On")
    logger.info("=" * 60)
    
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = "./temp_api_uploads"
        Path(temp_dir).mkdir(exist_ok=True)
        
        # Save uploaded files temporarily
        logger.info("Processing uploaded files...")
        
        clothing_path = os.path.join(temp_dir, "clothing_" + clothing_image.filename)
        avatar_path = os.path.join(temp_dir, "avatar_" + avatar_image.filename)
        
        with open(clothing_path, "wb") as f:
            f.write(await clothing_image.read())
        
        with open(avatar_path, "wb") as f:
            f.write(await avatar_image.read())
        
        background_path = None
        if background_image:
            background_path = os.path.join(temp_dir, "background_" + background_image.filename)
            with open(background_path, "wb") as f:
                f.write(await background_image.read())
        
        # Process try-on
        result = process_try_on(
            clothing_image_path=clothing_path,
            avatar_image_path=avatar_path,
            output_folder="./results",
            clothing_prompt=clothing_prompt,
            avatar_prompt=avatar_prompt,
            avatar_sex=avatar_sex,
            background_image_path=background_path,
            background_prompt=background_prompt,
            seed=seed,
        )
        
        if result["status"] != "success":
            logger.error(f"Processing failed: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Return the generated image file
        output_path = result["output_path"]
        
        logger.info(f"Returning image: {output_path}")
        
        output_filename = os.path.basename(output_path)
        base_url = str(request.base_url).rstrip("/")
        return {
            "result_image": f"{base_url}/results/{output_filename}",
            "seed": result["seed"]
        }

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")


# ==================== Server Launch ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run as FastAPI server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        logger.info(f"Starting FastAPI server on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Run CLI
        main()
