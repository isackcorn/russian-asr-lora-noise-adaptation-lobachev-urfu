set -e
echo "create environment"
conda create -n whisper-lora python=3.10 -y
echo "activate environment"
eval "$(conda shell.bash hook)"
conda activate whisper-lora
echo "install torch"
pip install torch==2.1.2 torchaudio==2.1.2
echo "install requirements"
pip install -r requirements.txt
echo ""
echo "check environment"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
if torch.backends.mps.is_available():
    print('MPS (Apple Silicon) available')
"
echo ""
echo "done"
echo "conda activate whisper-lora"
echo "pip freeze > requirements_locked.txt"
