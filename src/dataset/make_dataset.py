import argparse
from timesnet_pipeline_data_loader import load_dataset

def verify_dataset(args):
    """
    Function to verify the dataset loading and provide basic statistics.
    """
    for flag in ['train', 'val', 'test']:
        data_loader, scaler = load_dataset(args, flag)
        
        print(f"\n{flag.upper()} Dataset:")
        print(f"Number of batches: {len(data_loader)}")
        
        # Get a sample batch
        sample_batch = next(iter(data_loader))
        if isinstance(sample_batch, (list, tuple)):
            print(f"Batch structure: {type(sample_batch)}")
            print(f"Number of tensors in batch: {len(sample_batch)}")
            for i, tensor in enumerate(sample_batch):
                print(f"  Tensor {i} shape: {tensor.shape}")
        else:
            print(f"Batch shape: {sample_batch.shape}")
        
        if scaler:
            print("Scaler is provided for this dataset")
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                print(f"Scaler mean: {scaler.mean_[:5]}...")
                print(f"Scaler scale: {scaler.scale_[:5]}...")
        else:
            print("No scaler is provided for this dataset")
        
        # Basic statistics
        if isinstance(sample_batch, (list, tuple)):
            data = sample_batch[0]  # Assume the first tensor is the main data
        else:
            data = sample_batch
        print(f"Data type: {data.dtype}")
        print(f"Data range: [{data.min().item():.4f}, {data.max().item():.4f}]")
        print(f"Data mean: {data.mean().item():.4f}")
        print(f"Data std: {data.std().item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TimesNet Dataset Verification')
    
    # Add all necessary arguments here
    parser.add_argument('--data', type=str, required=True, help='dataset name')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--scale', action='store_true', help='whether to scale the data')
    parser.add_argument('--timeenc', type=int, default=0, help='time encoding method')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    
    args = parser.parse_args()
    
    verify_dataset(args)