"""
Build and create manifests for downloadable NIM models.

This module handles:
1. Building Docker images from the Dockerfile.template
2. Creating manifests in the models/downloadable directory structure
   that mirrors the models/api structure

Usage:
    from downloadables_create import build_downloadable_nim
    
    # Build a downloadable NIM model with manifest at:
    # models/downloadable/embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v2/dataloop.json
    build_downloadable_nim(
        model_name="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
        manifest_path="embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v2"
    )
"""

import json
import subprocess
from pathlib import Path


def _get_agent_dir() -> Path:
    """Get the agent directory path."""
    return Path(__file__).parent


def _get_repo_root() -> Path:
    """Get the repository root directory."""
    return _get_agent_dir().parent


def _extract_version() -> str:
    """Extract version from .bumpversion.cfg in repo root."""
    bumpversion_path = _get_repo_root() / '.bumpversion.cfg'
    with open(bumpversion_path, 'r') as f:
        for line in f:
            if line.startswith('current_version = '):
                return line.split(' = ')[1].strip()
    raise ValueError("Could not find version in .bumpversion.cfg")


def _get_nim_entrypoint(model_name: str) -> str:
    """
    Get the NIM container's entrypoint (start script) using docker inspect.
    
    Args:
        model_name: NIM model name (e.g., 'nvidia/nvclip')
    
    Returns:
        Path to the start script (e.g., '/opt/nim/start_server.sh')
    """
    image_name = f"nvcr.io/nim/{model_name}:latest"
    
    try:
        result = subprocess.run(
            ['docker', 'inspect', image_name, '--format', '{{json .Config.Entrypoint}}'],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse JSON output like ["/opt/nim/start_server.sh"]
        entrypoint = json.loads(result.stdout.strip())
        if entrypoint and isinstance(entrypoint, list) and len(entrypoint) > 0:
            return entrypoint[0]
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as e:
        print(f"Warning: Could not get entrypoint via docker inspect: {e}")
    
    # Fallback to default
    return "/opt/nim/start_server.sh"


def create_manifest(
    model_name: str,
    manifest_path: str,
    image_version: str = "0.1.13"
) -> dict:
    """
    Create a manifest from the template by replacing placeholders.
    
    The manifest is saved to models/downloadable/<manifest_path>/dataloop.json,
    mirroring the structure of models/api/.
    
    Args:
        model_name: NIM model name (e.g., 'nvidia/llama-3.2-nemoretriever-300m-embed-v2')
        manifest_path: Relative path for the manifest (e.g., 'embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v2')
        image_version: Docker image version tag
    
    Returns:
        Manifest dictionary
    """
    template_path = _get_agent_dir() / 'manifest_template.json'
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Normalize model name for Docker (replace / with -)
    docker_model_name = model_name.replace("/", "-")
    
    # Create display name from model name
    display_name = model_name.split('/')[-1].replace('-', ' ').title()
    
    # Replace placeholders
    manifest_content = template_content.replace('{{MODEL_NAME}}', docker_model_name)
    manifest_content = manifest_content.replace('{{MODEL_DISPLAY_NAME}}', display_name)
    manifest_content = manifest_content.replace('{{IMAGE_VERSION}}', image_version)
    
    manifest = json.loads(manifest_content)
    
    # Save to models/downloadable/<manifest_path>/dataloop.json
    output_dir = _get_repo_root() / 'models' / 'downloadable' / manifest_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'dataloop.json'
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    print(f"✓ Created manifest at: {output_path}")
    return manifest


def build_docker_image(model_name: str, image_version: str = "1.0.0") -> str:
    """
    Build and push Docker image using docker commands.
    
    Uses docker inspect to get the NIM container's entrypoint and passes it
    as a build arg to set NIM_START_SCRIPT env var.
    
    Args:
        model_name: NIM model name (e.g. 'nvidia/nvclip' or 'baai/bge-m3').
            Slashes are replaced with hyphens for the Docker image name.
        image_version: Docker image version tag
    
    Returns:
        Target image name
    """
    # Normalize model name for Docker (replace / with -)
    docker_image_name = model_name.replace("/", "-")
    # target_image = f"gcr.io/viewo-g/piper/agent/runner/gpu/{docker_image_name}:{image_version}"
    target_image = f"hub.dataloop.ai/dataloop/piper/agent/runner/gpu/{docker_image_name}:{image_version}"
    agent_dir = _get_agent_dir()
    
    # Get the entrypoint from the base NIM image
    nim_start_script = _get_nim_entrypoint(model_name)
    
    print("=" * 60)
    print(f"Building Docker image: {target_image}")
    print(f"Base NIM image: nvcr.io/nim/{model_name}:latest")
    print(f"NIM start script: {nim_start_script}")
    print("=" * 60)

    # Build the image with IMAGE_NAME and NIM_START_SCRIPT
    build_cmd = [
        'docker', 'build',
        '--build-arg', f'IMAGE_NAME={model_name}',
        '--build-arg', f'NIM_START_SCRIPT={nim_start_script}',
        '-f', str(agent_dir / 'Dockerfile.template'),
        '-t', target_image,
        str(agent_dir)
    ]
    
    print(f"Running: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed with exit code {result.returncode}")
    
    print(f"\n✓ Built {target_image}")
    
    # Push the image
    print(f"\nPushing {target_image}...")
    push_cmd = ['docker', 'push', target_image]
    
    result = subprocess.run(push_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Docker push failed with exit code {result.returncode}")
    
    print(f"✓ Pushed {target_image}")
    
    # Clean up Docker images to free disk space
    base_image = f"nvcr.io/nim/{model_name}:latest"
    print(f"\nCleaning up Docker images...")
    
    # Remove the built image
    subprocess.run(['docker', 'rmi', '-f', target_image], check=False)
    print(f"  ✓ Removed {target_image}")
    
    # Remove the base NIM image
    subprocess.run(['docker', 'rmi', '-f', base_image], check=False)
    print(f"  ✓ Removed {base_image}")
    
    return target_image


def build_downloadable_nim(model_name: str, manifest_path: str) -> dict:
    """
    Build a downloadable NIM model: Docker image and manifest.
    
    The manifest will be created at models/downloadable/<manifest_path>/dataloop.json,
    mirroring the structure of models/api/.
    
    Args:
        model_name: NIM model name (e.g., 'nvidia/llama-3.2-nemoretriever-300m-embed-v2')
        manifest_path: Relative path for manifest (e.g., 'embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v2')
    
    Returns:
        Manifest dictionary
    
    Example:
        # This creates:
        # - Docker image: gcr.io/viewo-g/piper/agent/runner/gpu/nvidia-llama-3.2-nemoretriever-300m-embed-v2:<version>
        # - Manifest at: models/downloadable/embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v2/dataloop.json
        build_downloadable_nim(
            model_name="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
            manifest_path="embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v2"
        )
    """
    version = _extract_version()
    print(f"\nBuilding downloadable NIM: {model_name}")
    print(f"Version: {version}")
    print(f"Manifest path: models/downloadable/{manifest_path}/dataloop.json\n")
    
    build_docker_image(model_name=model_name, image_version=version)
    manifest = create_manifest(
        model_name=model_name,
        manifest_path=manifest_path,
        image_version=version
    )
    
    print(f"\n✓ Successfully built downloadable NIM: {model_name}")
    return manifest


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build downloadable NIM models")
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="NIM model name (e.g., 'nvidia/llama-3.2-nemoretriever-300m-embed-v2')"
    )
    parser.add_argument(
        "--path", "-p",
        required=True,
        help="Manifest path relative to models/downloadable/ (e.g., 'embeddings/nvidia/llama_3_2_nemoretriever_300m_embed_v2')"
    )
    
    args = parser.parse_args()
    build_downloadable_nim(model_name=args.model, manifest_path=args.path)
