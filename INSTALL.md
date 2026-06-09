# Installation

This release is tested for the CODA-LM + LLaVA-1.5-7B path.

## 1. Create Environment

```bash
conda create -n seeing-llava python=3.10 -y
conda activate seeing-llava
```

Install PyTorch for your CUDA version. On most CUDA 12.x machines, start with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If this wheel does not match your machine, install `torch` and `torchvision` from the official PyTorch selector, then continue below.

## 2. Clone This Repository

```bash
git clone https://github.com/XinHu98/seeing.git
cd seeing
```

## 3. Install LLaVA

The CODA-LM LLaVA scripts import the external `llava` package. Install LLaVA or LLaVA-NeXT in editable mode:

```bash
mkdir -p external
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git external/LLaVA-NeXT
pip install -e external/LLaVA-NeXT --no-deps
```

If you already have a working LLaVA checkout, use it instead:

```bash
export PYTHONPATH=/path/to/LLaVA-or-LLaVA-NeXT:${PYTHONPATH}
```

## 4. Install Seeing Dependencies

Run this after installing LLaVA so the CODA-LM release controls the tested `transformers` version and the small LLaVA-side helper packages:

```bash
pip install -r requirements.txt
```

## 5. Check Installation

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import llava; print('llava', llava.__file__)"
PYTHONPATH=src:${PYTHONPATH:-} python -m seeing.cli.eval_llava15_coda --help
```

The final command should print the CODA-LM evaluation arguments.
