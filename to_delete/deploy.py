"""
Deploy script for downloadable NIM models.

This script handles:
1. Building Docker images (optional)
2. Creating dynamic manifests from template
3. Publishing and installing to Dataloop

Usage:
    python deploy.py --model nvclip --project "COCO ors"
    python deploy.py --model nvclip --project "COCO ors" --build
    python deploy.py --model nvclip --project "COCO ors" --clean
    python deploy.py --model nvclip --project "COCO ors" --integration <integration-id>
"""

import argparse
import json
import os
from pathlib import Path

import dtlpy as dl

from downloadables_create import build_docker_image, _get_agent_dir, _get_repo_root


def clean(dpk_name: str):
    """Remove all installed apps and DPK revisions for the given dpk_name."""
    try:
        dpk = dl.dpks.get(dpk_name=dpk_name)
    except dl.exceptions.NotFound:
        print(f"DPK {dpk_name} not found")
        return

    apps_filters = dl.Filters(field='dpkName', values=dpk.name, resource='apps')
    for app in dl.apps.list(filters=apps_filters).all():
        print(f"Uninstalling app: {app.name} from project: {app.project.name}")
        app.uninstall()

    revisions = list(dpk.revisions.all())
    for revision in revisions:
        revision.delete()
    print(f"Cleaned {len(revisions)} revisions for {dpk_name}")


def create_manifest(model_name: str, image_version: str = "0.1.13") -> dict:
    """
    Create a manifest from the template by replacing placeholders.
    
    Args:
        model_name: Name of the model (e.g., 'nvclip')
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
    display_name = docker_model_name.upper() if docker_model_name.lower() == 'nvclip' else docker_model_name.replace('-', ' ').title()
    
    # Replace placeholders
    manifest_content = template_content.replace('{{MODEL_NAME}}', docker_model_name)
    manifest_content = manifest_content.replace('{{MODEL_DISPLAY_NAME}}', display_name)
    manifest_content = manifest_content.replace('{{IMAGE_VERSION}}', image_version)
    
    return json.loads(manifest_content)


def publish_and_install(project: dl.Project, manifest: dict, integration_id: str = None) -> dl.App:
    """
    Publish the DPK and install/update the app in the project.
    
    Args:
        project: Dataloop project
        manifest: Manifest dictionary
        integration_id: Optional integration ID for NGC API key
    
    Returns:
        Installed/updated App
    """
    env = dl.environment()
    app_name = manifest['name']
    app_version = manifest['version']
    print(f'Publishing {app_name} v{app_version} to project {project.name} in {env}')

    dpk = dl.Dpk.from_json(manifest)
    
    # Pack from repo root to include models/downloadable/main.py
    repo_root = _get_repo_root()
    dpk.codebase = project.codebases.pack(
        directory=str(repo_root),
        name=dpk.display_name,
        extension='dpk',
        ignore_directories=['.venv', 'output', 'test_results', '.vscode', '.github', 'to_delete', '__pycache__'],
        ignore_max_file_size=True,
    )
    
    dpk = project.dpks.publish(dpk=dpk)
    print(f'Published successfully! DPK: {dpk.name}, version: {dpk.version}, id: {dpk.id}')
    
    # Prepare integrations if provided
    integrations = None
    if integration_id:
        integrations = [{'key': 'dl-ngc-api-key', 'value': integration_id}]
        print(f'Using integration: dl-ngc-api-key = {integration_id}')
    
    try:
        app = project.apps.get(app_name=dpk.display_name)
        print('App already installed, updating...')
        app.dpk_version = dpk.version
        app.update()
        print(f'Updated! App id: {app.id}')
    except dl.exceptions.NotFound:
        print('Installing new app...')
        app = project.apps.install(dpk=dpk, app_name=dpk.display_name, integrations=integrations)
        print(f'Installed! App id: {app.id}')
    
    return app


def main():
    parser = argparse.ArgumentParser(description='Deploy downloadable NIM models to Dataloop')
    parser.add_argument('--model', '-m', required=True, help='Model name (e.g., nvclip)')
    parser.add_argument('--project', '-p', required=True, help='Dataloop project name')
    parser.add_argument('--version', '-v', default='0.1.13', help='Docker image version (default: 0.1.13)')
    parser.add_argument('--build', '-b', action='store_true', help='Build Docker image before deploying')
    parser.add_argument('--clean', '-c', action='store_true', help='Clean existing installations before deploying')
    parser.add_argument('--env', '-e', default='prod', choices=['prod', 'dev', 'rc'], help='Dataloop environment')
    parser.add_argument('--integration', '-i', default='54945abd-a5c7-448f-a58a-c986445d1203', help='Integration ID for NGC API key (dl-ngc-api-key)')
    
    args = parser.parse_args()
    
    # Setup Dataloop
    dl.setenv('prod')
    
    print(args.project)
    project = dl.projects.get(project_name=args.project)
    dpk_name = f"nim-{args.model.replace('/', '-')}-downloadable"
    
    # Clean if requested
    if args.clean:
        print(f"\nCleaning existing installations for {dpk_name} in project {args.project}...")
        clean(dpk_name)
    
    # Build Docker image if requested
    if args.build:
        print(f"\nBuilding Docker image for {args.model}...")
        build_docker_image(args.model, args.version)
    
    # Create manifest and deploy
    print(f"\nCreating manifest for {args.model}...")
    manifest = create_manifest(args.model, args.version)
    
    print(f"\nDeploying to project {args.project}...")
    app = publish_and_install(project=project, manifest=manifest, integration_id=args.integration)
    
    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"App URL: https://{app.name.replace('_', '-')}-{app.id}.apps.dataloop.ai")
    print("=" * 60)


if __name__ == "__main__":
    main()
