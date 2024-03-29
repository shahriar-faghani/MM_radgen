import math
import os
os.environ['WANDB_API_KEY'] = 'your_api_key_here'  # Replace 'your_api_key_here' 
os.environ['WANDB_SILENT'] = 'true'
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai import monai as mn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import wandb

if __name__ == '__main__':
    # Standard library imports
    
    # Third-party library imports
    
    
    #---
    
    # Set your API key here
    
    def seed_all(seed: int) -> None:
        """
        Set the seed for all necessary libraries to ensure reproducibility.
        
        Args:
        seed (int): The seed value to use for all applicable libraries.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Ensure deterministic behavior in PyTorch (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # MONAI-specific seed setting
        mn.utils.misc.set_determinism(seed=seed)
    
    # Example of seeding
    seed_all(123)
    
    #---
    
    # Hyperparameters
    bs = 2
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1000
    fold_num = None  # TODO: Replace None with the actual fold number
    variable_number = None  # TODO: Replace None with the actual variable number
    
    # Load clinical data
    df_clinical = pd.read_csv(
        'Multiple_Myeloma/df_clinical_skeletal_survey_complete.csv'
    )  # Complete CSV
    column = list(df_clinical.columns)[variable_number]
    
    # Informative print statement
    print(f'This notebook will train and log a model for: {column}')
    
    #---
    
    total_data_list = []
    nii_idx = os.listdir('/Multiple_Myeloma/niftis')
    
    for idx in nii_idx:
        try:
            # Initialize the data dictionary with file path
            data_dict = {'img': f"/Multiple_Myeloma/niftis/{idx}"}
            
            # Extract the file ID by removing the file extension
            file_id = idx.split('.')[0]
            
            # Assign the corresponding label from 'df_clinical'
            data_dict['Label'] = int(df_clinical[df_clinical['id'] == int(file_id)][column].item())
            
            # Append the data dictionary to the list
            total_data_list.append(data_dict)
        except Exception as e:
            # Ideally, handle specific exceptions or log them for debugging
            print(f"Error processing {idx}: {e}")
    
    # Create a DataFrame from the list and print its length
    df = pd.DataFrame(total_data_list)
    print(len(total_data_list))
    
    #---
    
    # Initialize the StratifiedKFold object
    cv = StratifiedKFold(n_splits=5, random_state=1211, shuffle=True)
    
    # Initialize 'fold' counter
    fold = 0
    
    # Splitting the dataset
    for train_idxs, test_idxs in cv.split(X=df[['img']], y=df['Label']):
        if fold == fold_num:
            # Select the rows for training and validation sets based on the indexes
            df_train = df.iloc[train_idxs]
            df_val = df.iloc[test_idxs]
            break  # Exit the loop once the desired fold is processed
        fold += 1  # Increment 'fold' counter
    
    # Calculating the ratios of labels in the training set
    ratios = df_train['Label'].value_counts(normalize=True)  # Normalize=True to get proportions
    
    # If you wish to print or use 'ratios', you can do so here
    print(ratios)
    
    #create list of dictionaries
    train_data_list = df_train.to_dict('records')
    val_data_list = df_val.to_dict('records')
    
    #---
    
    def apply_window(img, ww=500, wl=100):
        """
        Apply windowing on the CT-scan.
    
        Parameters:
        img (ndarray): The input image array.
        ww (int): Window width.
        wl (int): Window level.
    
        Returns:
        ndarray: The windowed image array.
        """
        U = 1
        W = U / ww
        b = (-U / ww) * (wl - ww / 2)
        img = W * img + b
        img = np.clip(img, 0, U)  # Simplified clipping step
        return img
    
    class Window(mn.transforms.Transform):
        def __init__(self, key: str = "img") -> None:
            super().__init__()
            self.key = key
    
        def __call__(self, data):
            data_copy = copy.deepcopy(data)
            file = data_copy[self.key]
            data_copy[self.key] = apply_window(file)
            return data_copy
    
    def my_transform(data):
        """
        Apply transformations to the data.
    
        Parameters:
        data (dict): The input data with 'img' and 'Label'.
    
        Returns:
        dict: The transformed data.
        """
        img = data['img']
        label = data['Label']
        label = mn.utils.convert_to_tensor(label, dtype=None)  # Specify dtype if needed
        data = {'img': img, 'Label': label}
        return data
    
    #---
    
    train_transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys = ['img']),
        mn.transforms.AddChannelD(keys = 'img'),
        window(),
        mn.transforms.ScaleIntensityd(keys=['img'], minv=-1, maxv=1),
        mn.transforms.Resized(keys = ['img'], spatial_size =(256,256,800)),  
        mn.transforms.RandFlipd(keys = ['img'], prob = 0.5, spatial_axis = [0,1,2]),
        mn.transforms.RandAffineD(
           keys = ['img'],
           translate_range = (15,15,25),
           scale_range = (0.05,0.05,0.05),
            rotate_range = (math.pi/12,math.pi/12,math.pi/12),
            padding_mode = 'zeros',
            prob = 0.5),
        mn.transforms.RandGaussianNoised(keys = ['img'], prob=0.5, mean=0.0, std=0.2),
        mn.transforms.ToTensord(keys = ['img',"Label"]),
        my_transform,
    ])
    val_transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys = ['img']),
        mn.transforms.AddChannelD(keys = 'img'),
        window(),
        mn.transforms.ScaleIntensityd(keys=['img'], minv=-1, maxv=1),
        mn.transforms.Resized(keys = ['img'], spatial_size =(256,256,800)),
        mn.transforms.ToTensord(keys = ['img',"Label"]),
        my_transform,
    ])
    
    #---
    
    train_ds = mn.data.PersistentDataset(
        data=train_data_list, 
        transform=train_transforms, 
        cache_dir='path to your data'
    )
    
    val_ds = mn.data.PersistentDataset(
        data=val_data_list, 
        transform=val_transforms, 
        cache_dir='path to your data'
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=bs, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available(), 
        prefetch_factor=1, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=bs, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available(), 
        prefetch_factor=1
    )
    
    
    #---
    
    model = mn.networks.nets.DenseNet121(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=2
    ).to(device)
    
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor([float(ratios[1]), float(ratios[0])]).to(device)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs
    )
    
    #---
    
    def multiclass_auroc(preds, labels):
        """
        Compute the multiclass AUROC given the predicted probabilities and true labels.
    
        Args:
        - preds (tensor): Tensor of shape (N, C) containing the predicted probabilities for each class, where N is the number of instances and C is the number of classes.
        - labels (tensor): Tensor of shape (N,) containing the true labels for each instance.
    
        Returns:
        - auroc (float): The multiclass AUROC.
        """
    
        # Convert predictions and labels to numpy arrays
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
    
        # Compute AUROC for each class
        aurocs = []
        for c in range(preds.shape[1]):
            try:
                auroc = roc_auc_score(labels_np == c, preds_np[:, c])
            except ValueError:
                auroc = 0.5  # If there is only one class, set AUROC to 0.5
            aurocs.append(auroc)
    
        # Compute the average AUROC across all classes
        auroc = np.mean(aurocs)
    
        return auroc
    
    #---
    
    # Best metrics initialization
    best_metric_acc_epoch = -1
    best_metric_acc = 0
    best_metric_auc_epoch = -1
    best_metric_auc = 0
    
    # Initialize wandb
    wandb.init(project='MM_skeletal_survey', entity='your_entity')
    
    # Define your wandb configuration
    config = wandb.config
    config.learning_rate = lr
    config.batch_size = bs
    config.mode = '3D'
    config.backbone = 'DenseNet121'
    config.optimizer = 'AdamW'
    config.normalization = 'Per patient'
    config.epochs = epochs
    
    # Naming the run for clarity
    wandb.run.name = (f'MM_skeletal_survey_3D_Densenet121_{lr}_bs{bs}_'
                      f'{column}_fold{fold_num}')
    
    # Training loop
    for i, epoch in enumerate(tqdm(range(epochs))):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        val_epoch_loss = 0
        step = 0
        val_step = 0
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
    
        # Training step
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['img'].to(device), batch_data['Label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = (len(train_ds) // train_loader.batch_size) + 1
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            wandb.log({"loss": loss.item()})
            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, labels], dim=0)
    
        # Logging and scheduling
        wandb.log({'lr': lr_scheduler.get_lr()[0]})
        lr_scheduler.step()
        epoch_loss /= step
        wandb.log({'epoch_loss': epoch_loss})
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        # Training metrics
        acc_value_train = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric_train = acc_value_train.sum().item() / len(acc_value_train)
        train_auc = multiclass_auroc(y_pred.softmax(dim=1), y)
        wandb.log({'train_AUC': train_auc, 'train_acc': acc_metric_train})
    
        # Validation step
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            val_step = 0
            for val_data in val_loader:
                val_step += 1
                val_images, val_labels = val_data['img'].to(device), val_data['Label'].to(device)
                val_outputs = model(val_images)
                val_loss = loss_function(val_outputs, val_labels)
                val_epoch_loss += val_loss.item()
                val_epoch_len = (len(val_ds) // val_loader.batch_size) + 1
                wandb.log({"val_loss": val_loss.item()})
                y_pred = torch.cat([y_pred, val_outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)
    
            # Validation metrics and model saving
            val_epoch_loss /= val_step
            wandb.log({'val_epoch_loss': val_epoch_loss})
            acc_value_val = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric_val = acc_value_val.sum().item() / len(acc_value_val)
            val_auc = multiclass_auroc(y_pred.softmax(dim=1), y)
            wandb.log({'val_auc': val_auc, 'val_acc': acc_metric_val})
            print(f'preds:{y_pred},\n labels:{y}')
            if acc_metric_val > best_metric_acc:
                best_metric_acc = acc_metric_val
                best_metric_acc_epoch = epoch
                torch.save(model.state_dict(), 
                           f'MM_skeletal_survey_highest_acc_{column}_fold{fold_num}.pth')
                print(f'New model
    


