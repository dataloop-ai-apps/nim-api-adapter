#!/bin/bash

# Build script for downloadable NIM Docker images
# Usage: bash build_downloadable_nim.sh <image_name>
# Example: bash build_downloadable_nim.sh nvclip

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to display usage
usage() {
    echo "Usage: $0 <image_name>"
    echo ""
    echo "Example:"
    echo "  $0 nvclip"
    echo ""
    echo "This will build and push:"
    echo "  gcr.io/viewo-g/piper/agent/runner/gpu/<image_name>:1.0.0"
    echo ""
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 1 ]; then
    echo -e "${RED}Error: Invalid number of arguments${NC}"
    usage
fi

IMAGE_NAME="$1"
BASE_IMAGE="nvcr.io/nim/nvidia/${IMAGE_NAME}:latest"
TARGET_IMAGE="gcr.io/viewo-g/piper/agent/runner/gpu/${IMAGE_NAME}:1.0.0"

echo -e "${GREEN}Building downloadable NIM Docker image...${NC}"
echo -e "Base image: ${YELLOW}${BASE_IMAGE}${NC}"
echo -e "Target image: ${YELLOW}${TARGET_IMAGE}${NC}"
echo ""

# Step 1: Build the Docker image
echo -e "${GREEN}[1/2] Building Docker image...${NC}"
if docker build \
    --build-arg IMAGE_NAME="${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile.template" \
    -t "${TARGET_IMAGE}" \
    "${SCRIPT_DIR}"; then
    echo -e "${GREEN}✓ Successfully built ${TARGET_IMAGE}${NC}"
else
    echo -e "${RED}✗ Failed to build Docker image${NC}"
    exit 1
fi
echo ""

# Step 2: Push the image to the registry
echo -e "${GREEN}[2/2] Pushing image to registry...${NC}"
if docker push "${TARGET_IMAGE}"; then
    echo -e "${GREEN}✓ Successfully pushed ${TARGET_IMAGE}${NC}"
else
    echo -e "${RED}✗ Failed to push ${TARGET_IMAGE}${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}Build and push completed successfully!${NC}"
echo -e "Image available at: ${YELLOW}${TARGET_IMAGE}${NC}"
