if ! command -v pre-commit 2>&1 >/dev/null
then
    pip install pre-commit
fi

export HTTPS_PROXY=http://proxy.ics.muni.cz:3128

uv python install 3.11
uv venv
uv sync

# Create necessary directories
mkdir -p ./cache
mkdir -p ./explanations
mkdir -p ./logs

echo "✅ Setup complete!"
echo ""
echo "To run CIAO explanations:"
echo "  python -m ciao"
echo ""
echo "To run with different config:"
echo "  python -m ciao variant=colorectal"
echo "  python -m ciao ciao/segmentation=hexagonal"
echo "  python -m ciao ciao/obfuscation=interlacing"