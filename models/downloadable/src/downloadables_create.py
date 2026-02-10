import argparse
import json
import os
import subprocess
import dtlpy as dl



def create_manifest(model_name: str, image_version: str = "0.1.13", manifest_path: str = "embeddings/nvidia/nvclip") -> dict:
    """
    Create a manifest from the template by replacing placeholders.
    
    Args:
        model_name: Name of the model (e.g., 'nvclip')
        image_version: Docker image version tag
        manifest_path: Path to save the manifest
    
    Returns:
        Manifest dictionary
    """
    template_path = os.path.join(os.path.dirname(__file__), 'manifest_template.json')
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Create display name from model name
    display_name = model_name.replace('-', ' ').title()
    
    # Replace placeholders
    manifest_content = template_content.replace('{{MODEL_NAME}}', model_name)
    manifest_content = manifest_content.replace('{{MODEL_DISPLAY_NAME}}', display_name)
    manifest_content = manifest_content.replace('{{IMAGE_VERSION}}', image_version)
    
    manifest_path = os.path.join("models/downloadable", manifest_path)
    
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(os.path.join(manifest_path, 'dataloop.json'), 'w') as f:
        json.dump(json.loads(manifest_content), f, indent=4)



def build_docker_image(model_name: str, image_version: str = "1.0.0"):
    """
    Build and push Docker image directly using docker commands.
    
    Args:
        model_name: NIM model name (e.g. 'nvclip' or 'baai/bge-m3'). Slashes are
            replaced with hyphens for the NGC image path (nvcr.io/nim/nvidia/<name>:latest).
        image_version: Docker image version tag
    """
    # NGC expects a single segment; slashes cause "Incorrect Repository Format"
    docker_image_name = model_name.replace("/", "-")
    target_image = f"gcr.io/viewo-g/piper/agent/runner/gpu/{docker_image_name}:{image_version}"
    work_dir = os.path.dirname(__file__) or '.'
    
    print("=" * 60)
    print(f"Building Docker image: {target_image}")
    print("=" * 60)
    
    # Build the image (IMAGE_NAME for Dockerfile FROM nvcr.io/nim/nvidia/${IMAGE_NAME}:latest)
    build_cmd = [
        'docker', 'build',
        '--build-arg', f'IMAGE_NAME={model_name}',
        '-f', 'Dockerfile.template',
        '-t', target_image,
        '.'
    ]
    
    print(f"Running: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, cwd=work_dir, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed with exit code {result.returncode}")
    
    print(f"\n✓ Built {target_image}")
    
    # Push the image
    print(f"\nPushing {target_image}...")
    push_cmd = ['docker', 'push', target_image]
    
    result = subprocess.run(push_cmd, cwd=work_dir, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Docker push failed with exit code {result.returncode}")
    
    print(f"✓ Pushed {target_image}")

def _extract_version():
    # extract version from .bumpversion.cfg
    with open('.bumpversion.cfg', 'r') as f:
        for line in f:
            if line.startswith('current_version = '):
                version = line.split(' = ')[1].strip()
                break
    return version

def build_downloadable_nim(model_name: str, manifest_path: str):
    version = _extract_version()
    build_docker_image(model_name=model_name, image_version=version)
    create_manifest(model_name=model_name, image_version=version, manifest_path=manifest_path)