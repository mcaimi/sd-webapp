#!/usr/bin/env python
"""
API Client for Stable Diffusion Backend

Handles all communication with the FastAPI backend for image generation.
"""

import base64
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

import requests
from PIL import Image


class SDAPIClient:
    """Client for communicating with the Stable Diffusion API backend."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def health_check(self) -> Dict[str, str]:
        """Check API health status.

        Returns:
            Health status dict
        """
        response = self._make_request("GET", "/api/v1/system/health")
        return response.json()

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.

        Returns:
            System info including GPU, cache, and queue stats
        """
        response = self._make_request("GET", "/api/v1/system/info")
        return response.json()

    def list_models(self, model_type: str, resource_type: str) -> Dict[str, Any]:
        """List available models.

        Args:
            model_type: "sd15" or "sdxl"
            resource_type: "checkpoints", "loras", or "vaes"

        Returns:
            List of models with names and paths
        """
        response = self._make_request(
            "GET", f"/api/v1/models/{model_type}/{resource_type}"
        )
        return response.json()

    def get_model_metadata(
        self, model_type: str, resource_type: str, model_name: str
    ) -> Dict[str, Any]:
        """Get metadata for a specific model.

        Args:
            model_type: "sd15" or "sdxl"
            resource_type: "checkpoints", "loras", or "vaes"
            model_name: Model filename

        Returns:
            Model metadata
        """
        response = self._make_request(
            "GET",
            f"/api/v1/models/{model_type}/{resource_type}/{model_name}/metadata",
        )
        return response.json()

    def list_schedulers(self) -> List[str]:
        """List available noise schedulers.

        Returns:
            List of scheduler names
        """
        response = self._make_request("GET", "/api/v1/info/schedulers")
        return response.json()

    def generate_image(
        self,
        model_type: str,
        positive_prompt: str,
        negative_prompt: str = "",
        model_checkpoint: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.0,
        seed: int = -1,
        scheduler: str = "DPM++ 2M",
        loras: Optional[List[Dict[str, Any]]] = None,
        custom_vae: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a single image (async job).

        Args:
            model_type: "sd15" or "sdxl"
            positive_prompt: Positive prompt
            negative_prompt: Negative prompt
            model_checkpoint: Model checkpoint filename
            width: Image width
            height: Image height
            steps: Inference steps
            cfg_scale: CFG scale
            seed: Random seed (-1 for random)
            scheduler: Noise scheduler name
            loras: List of LoRA configs
            custom_vae: Custom VAE checkpoint

        Returns:
            Tuple of (job_id, job_response)
        """
        payload = {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "model_checkpoint": model_checkpoint,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "scheduler": scheduler,
        }

        if loras:
            payload["loras"] = loras
        if custom_vae:
            payload["custom_vae"] = custom_vae

        response = self._make_request(
            "POST", f"/api/v1/{model_type}/generate", json=payload
        )

        job_response = response.json()
        return job_response["job_id"], job_response

    def batch_generate(
        self,
        model_type: str,
        requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate multiple images in a batch.

        Args:
            model_type: "sd15" or "sdxl"
            requests: List of generation requests

        Returns:
            List of job responses
        """
        payload = {"requests": requests}
        response = self._make_request(
            "POST", f"/api/v1/{model_type}/generate/batch", json=payload
        )
        return response.json()

    def compare_models(
        self,
        model_type: str,
        positive_prompt: str,
        models: List[str],
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.0,
        seed: int = 42,
        scheduler: str = "DPM++ 2M",
    ) -> List[Dict[str, Any]]:
        """Generate images from multiple models for comparison.

        Args:
            model_type: "sd15" or "sdxl"
            positive_prompt: Positive prompt
            models: List of model checkpoints to compare
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            steps: Inference steps
            cfg_scale: CFG scale
            seed: Fixed seed for fair comparison
            scheduler: Noise scheduler name

        Returns:
            List of job responses
        """
        payload = {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "models": models,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "scheduler": scheduler,
        }
        response = self._make_request(
            "POST", f"/api/v1/{model_type}/generate/compare", json=payload
        )
        return response.json()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status and result.

        Args:
            job_id: Job ID

        Returns:
            Job status response
        """
        response = self._make_request("GET", f"/api/v1/jobs/{job_id}")
        return response.json()

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs.

        Returns:
            List of job status responses
        """
        response = self._make_request("GET", "/api/v1/jobs/")
        return response.json()

    def cancel_job(self, job_id: str) -> None:
        """Cancel a job.

        Args:
            job_id: Job ID to cancel
        """
        self._make_request("DELETE", f"/api/v1/jobs/{job_id}")

    def wait_for_job(
        self, job_id: str, poll_interval: float = 1.0, timeout: float = 300.0
    ) -> Dict[str, Any]:
        """Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait

        Returns:
            Final job status response

        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails
        """
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            status = self.get_job_status(job_id)

            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                error = status.get("error", "Unknown error")
                raise RuntimeError(f"Job {job_id} failed: {error}")
            elif status["status"] in ("pending", "running"):
                time.sleep(poll_interval)
            else:
                raise RuntimeError(f"Unknown job status: {status['status']}")

    def decode_image_from_result(self, result: Dict[str, Any]) -> Image.Image:
        """Decode a PIL Image from job result.

        Args:
            result: Job result containing base64 image

        Returns:
            PIL Image
        """
        if "image_base64" in result:
            image_data = base64.b64decode(result["image_base64"])
            return Image.open(BytesIO(image_data))

        raise ValueError("No image found in result")

    def generate_and_wait(
        self,
        model_type: str,
        positive_prompt: str,
        negative_prompt: str = "",
        model_checkpoint: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.0,
        seed: int = -1,
        scheduler: str = "DPM++ 2M",
        loras: Optional[List[Dict[str, Any]]] = None,
        custom_vae: Optional[str] = None,
        timeout: float = 300.0,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate an image and wait for completion.

        Args:
            model_type: "sd15" or "sdxl"
            positive_prompt: Positive prompt
            negative_prompt: Negative prompt
            model_checkpoint: Model checkpoint filename
            width: Image width
            height: Image height
            steps: Inference steps
            cfg_scale: CFG scale
            seed: Random seed (-1 for random)
            scheduler: Noise scheduler name
            loras: List of LoRA configs
            custom_vae: Custom VAE checkpoint
            timeout: Maximum seconds to wait

        Returns:
            Tuple of (PIL Image, generation parameters)
        """
        job_id, _ = self.generate_image(
            model_type=model_type,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            model_checkpoint=model_checkpoint,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            scheduler=scheduler,
            loras=loras,
            custom_vae=custom_vae,
        )

        job_status = self.wait_for_job(job_id, timeout=timeout)
        result = job_status["result"]
        image = self.decode_image_from_result(result)
        
        return image, result.get("metadata", {})

    def merge_models(
        self,
        model_type: str,
        base_model: str,
        target_model: str,
        method: str = "linear",
        alpha: float = 0.5,
        output_name: str = "",
        preserve_metadata: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Submit a merge job to the API.

        Args:
            model_type: "sd15" or "sdxl"
            base_model: Base model checkpoint filename
            target_model: Target model checkpoint filename
            method: Merge method (linear, slerp, additive, subtract)
            alpha: Merge strength/alpha (0.0-1.0)
            output_name: Output checkpoint filename
            preserve_metadata: Preserve model metadata

        Returns:
            Tuple of (job_id, job_response)
        """
        payload = {
            "model_type": model_type,
            "base_model": base_model,
            "target_model": target_model,
            "method": method,
            "alpha": alpha,
            "output_name": output_name,
            "preserve_metadata": preserve_metadata,
        }
        response = self._make_request("POST", "/api/v1/merge/merge", json=payload)
        job_response = response.json()
        return job_response["job_id"], job_response

    def batch_merge_models(
        self,
        model_type: str,
        base_model: str,
        target_models: List[str],
        method: str = "linear",
        alpha: float = 0.5,
        output_subdir: str = "batch_merged",
        preserve_metadata: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Submit a batch merge job to the API.

        Args:
            model_type: "sd15" or "sdxl"
            base_model: Base model checkpoint filename
            target_models: List of target model checkpoint filenames
            method: Merge method (linear, slerp, additive, subtract)
            alpha: Merge strength/alpha (0.0-1.0)
            output_subdir: Output subdirectory name
            preserve_metadata: Preserve model metadata

        Returns:
            Tuple of (job_id, job_response)
        """
        payload = {
            "model_type": model_type,
            "base_model": base_model,
            "target_models": target_models,
            "method": method,
            "alpha": alpha,
            "output_subdir": output_subdir,
            "preserve_metadata": preserve_metadata,
        }
        response = self._make_request("POST", "/api/v1/merge/batch", json=payload)
        job_response = response.json()
        return job_response["job_id"], job_response

    def recipe_merge_models(
        self,
        model_type: str,
        base_model: str,
        steps: List[Dict[str, Any]],
        output_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Submit a recipe merge job to the API.

        Args:
            model_type: "sd15" or "sdxl"
            base_model: Base model checkpoint filename
            steps: List of dicts with keys: target_model, method, alpha
            output_name: Output checkpoint filename

        Returns:
            Tuple of (job_id, job_response)
        """
        payload = {
            "model_type": model_type,
            "base_model": base_model,
            "steps": steps,
            "output_name": output_name,
        }
        response = self._make_request("POST", "/api/v1/merge/recipe", json=payload)
        job_response = response.json()
        return job_response["job_id"], job_response

    def inpaint_image(
        self,
        model_type: str,
        positive_prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: str = "",
        model_checkpoint: Optional[str] = None,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        scheduler: str = "DPM++ 2M",
        loras: Optional[List[Dict[str, Any]]] = None,
        custom_vae: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Inpaint an image (async job).

        Args:
            model_type: "sd15" or "sdxl"
            positive_prompt: Positive prompt
            image: Input PIL Image
            mask_image: Mask PIL Image (white = inpaint, black = preserve)
            negative_prompt: Negative prompt
            model_checkpoint: Model checkpoint filename
            steps: Inference steps
            cfg_scale: CFG scale
            seed: Random seed (-1 for random)
            scheduler: Noise scheduler name
            loras: List of LoRA configs
            custom_vae: Custom VAE checkpoint

        Returns:
            Tuple of (job_id, job_response)
        """
        # Convert images to base64
        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        image_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        mask_buffer = BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")

        payload = {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "model_checkpoint": model_checkpoint,
            "image_base64": image_base64,
            "mask_base64": mask_base64,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "scheduler": scheduler,
        }

        if loras:
            payload["loras"] = loras
        if custom_vae:
            payload["custom_vae"] = custom_vae

        response = self._make_request(
            "POST", f"/api/v1/{model_type}/inpaint", json=payload
        )

        job_response = response.json()
        return job_response["job_id"], job_response

    def inpaint_and_wait(
        self,
        model_type: str,
        positive_prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: str = "",
        model_checkpoint: Optional[str] = None,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        scheduler: str = "DPM++ 2M",
        loras: Optional[List[Dict[str, Any]]] = None,
        custom_vae: Optional[str] = None,
        timeout: float = 300.0,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Inpaint an image and wait for completion.

        Args:
            model_type: "sd15" or "sdxl"
            positive_prompt: Positive prompt
            image: Input PIL Image
            mask_image: Mask PIL Image
            negative_prompt: Negative prompt
            model_checkpoint: Model checkpoint filename
            steps: Inference steps
            cfg_scale: CFG scale
            seed: Random seed (-1 for random)
            scheduler: Noise scheduler name
            loras: List of LoRA configs
            custom_vae: Custom VAE checkpoint
            timeout: Maximum seconds to wait

        Returns:
            Tuple of (PIL Image, generation parameters)
        """
        job_id, _ = self.inpaint_image(
            model_type=model_type,
            positive_prompt=positive_prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            model_checkpoint=model_checkpoint,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            scheduler=scheduler,
            loras=loras,
            custom_vae=custom_vae,
        )

        job_status = self.wait_for_job(job_id, timeout=timeout)
        result = job_status["result"]
        output_image = self.decode_image_from_result(result)

        return output_image, result.get("parameters", {})
