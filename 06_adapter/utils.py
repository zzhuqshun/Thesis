def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def create_dataloaders(datasets, seq_len, batch_size):
    """Create PyTorch DataLoaders from processed datasets"""
    loaders = {}
    for k, df in datasets.items():
        if not df.empty and any(x in k for x in ['train', 'val', 'test']):
            ds = BatteryDataset(df, seq_len)
            loaders[k] = DataLoader(ds, batch_size=batch_size, shuffle=('train' in k))
    return loaders

def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    log_path = log_dir / 'train.log'
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) 
              for h in logger.handlers):
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)
    
    return logger