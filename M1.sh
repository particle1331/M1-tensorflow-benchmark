source /Users/particle1331/miniforge3/bin/activate

conda activate ml
python run.py "M1 (GPU)" mlp
python run.py "M1 (GPU)" vgg

# consolidate results dir in one plot
python plot_results.py