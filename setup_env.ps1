# Setup Environment Script

echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing PyTorch Geometric..."
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

echo "Installing remaining dependencies..."
pip install pandas numpy scikit-learn matplotlib seaborn tqdm

echo "Environment setup complete!"
