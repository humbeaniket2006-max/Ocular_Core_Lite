# src/ocular_core/generator.py
import torch
from optimum.intel.openvino import OVStableDiffusionPipeline

def generate_iris(output_path="./output.png", prompt="extreme macro photo, human eye, blue iris, 8k"):
    """
    Generates a synthetic iris using the OpenVINO optimized pipeline.
    """
    print("Loading Neural Engine...")
    model_id = "OpenVINO/stable-diffusion-v1-5-fp16-ov"
    
    # Load model
    pipe = OVStableDiffusionPipeline.from_pretrained(model_id, compile=False)
    pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
    pipe.to("cpu")
    pipe.compile()

    print("Generating...")
    image = pipe(
        prompt=prompt,
        negative_prompt="blurry, drawing, painting, illustration",
        num_inference_steps=20,
        height=512,
        width=512
    ).images[0]

    image.save(output_path)
    print(f"Saved to {output_path}")
    return output_path