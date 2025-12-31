# src/ocular_core/stress.py
import time
from .generator import generate_iris # Re-use the generator we just made!

def stress_test(output_dir="./experiments"):
    """
    Runs the thermal throttling stress test protocol.
    """
    prompts = [
        ("dilated", "extreme macro photo, human eye, blue iris, pupil fully dilated, wide open"),
        ("constricted", "extreme macro photo, human eye, green iris, pupil constricted, bright light")
    ]
    
    for i, (name, prompt) in enumerate(prompts):
        print(f"--- Running Stress Test: {name} ---")
        generate_iris(f"{output_dir}/stress_{name}.png", prompt)
        
        if i < len(prompts) - 1:
            print("COOLING PROTOCOL: Sleeping 10s...")
            time.sleep(10)