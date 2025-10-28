import os
import sys
import json
from argparse import ArgumentParser
import torch
from code.kandev.utils import create_dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import logging
import numpy as np
from time import time
from code.kandev.kan import *
from matplotlib import pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU

def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Log all uncaught exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)  # Allow Ctrl+C to work normally
        return

    logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

# yapf: disable
parser = ArgumentParser()
parser.add_argument(
    'model_path', type=str,
    help='Directory where the model will be stored.'
)
parser.add_argument(
    'params_filepath', type=str,
    help='Path to the parameter file.'
)
parser.add_argument(
    'training_name', type=str,
    help='Name for the training.'
)
# yapf: enable



def train(model, device, train_loader, valid_loader, optimizer, loss_fn, epoch):
    global preds, labels

    model.train()

    total_train_loss = 0
    preds = []
    labels = []
    for _, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        x,y = x.to(device),y.to(device)
        output =  model(x)
        loss = loss_fn(output, y) 
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_train_loss += loss.item()               
        preds.append(output.detach().cpu())
        labels.append(y.detach().cpu())
    # Concatenate all batches
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    # tolerance = 0.01
    # train_accuracy = torch.mean((torch.abs(output - y) < tolerance).float())
    train_accuracy=torch.mean(
        ((torch.round(preds*100)/100)==(torch.round(labels*100)/100)).float()
        )
        
    model.eval()
    preds = []
    labels = []
    total_loss_val = 0.0
    with torch.no_grad():
        for _, (x,y) in enumerate(valid_loader):
            x,y = x.to(device),y.to(device)
            output = model(x)
            loss = loss_fn(output, y)      
            total_loss_val += loss.item()
            preds.append(output.detach().cpu())
            labels.append(y.detach().cpu())
        # Concatenate all batches
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
      
        # val_accuracy = torch.mean((torch.abs(output - y) < tolerance).float())
        val_accuracy=torch.mean(
            ((torch.round(preds*100)/100)==(torch.round(labels*100)/100)).float()
            )
    return total_train_loss, total_loss_val, train_accuracy.item(), val_accuracy.item()

def main(
    model_path,
    params_filepath,
    training_name
):
    global preds, labels
    
    # Read parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    # Create model directory and dump files
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)

    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logger = logging.getLogger(f"{training_name}")
    logger.setLevel(logging.INFO)

    # Suppress Matplotlib DEBUG logs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info('The code uses GPU...')
    else:
        device = torch.device('cpu')
        logger.info('The code uses CPU!!!')
  
    # Create a file handler to save logs to a file
    file_handler = logging.FileHandler(os.path.join(model_dir, "results", 'logfile.log'))
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    #Add the handlers to the logger
    logger.addHandler(file_handler)

    # Set the global exception handler
    sys.excepthook = global_exception_handler

    with open(os.path.join(model_dir, "model_params.json"), "w") as fp:
        json.dump(params, fp, indent=4)
   
    save_top_model = os.path.join(model_dir, "weights/{}_{}_{}.pt")
    # Helper functions
    def update_info():
        return {
            "best_val_accuracy": str(val_acc),
            "predictions": [float(p) for p in preds],
            "labels": [float(p) for p in labels],
                }

    def save(path, metric, typ, val=None):
        torch.save(model.state_dict(), save_top_model.format("best", "accuracy", training_name))
        with open(os.path.join(model_dir, "results", metric + ".json"), "w") as f:
            json.dump(info, f)
        if typ == "best":
            logger.info(
                f'\t New best performance in "{metric}"'
                f" with value : {val:.7f} in epoch: {epoch}"
            )

    # Prepare the dataset
    logger.info("Start creating data...")
    f = eval(params["data_fn"], {"torch": torch})
    assert type(params["KAN_layers_nodes"])==list, "KAN_layers_nodes should be a list"    
    kan_layers_input=params["KAN_layers_nodes"]
    if type(kan_layers_input[0]) == list:
        num_input = kan_layers_input[0][0]
    else:
        num_input = kan_layers_input[0]
    dataset = create_dataset(f, n_var=num_input, train_num=int(params["num_samples_train"]), test_num=int(params["num_samples_val"]), device=device)
    # dataset1 = create_dataset(f, n_var=num_input, train_num=1000, test_num=int(params["num_samples_test"]), device=device)

    train_ds = TensorDataset(dataset['train_input'], dataset['train_label'])
    val_ds = TensorDataset(dataset['test_input'], dataset['test_label'])
    # test_ds = TensorDataset(dataset1['test_input'], dataset1['test_label'])
    train_loader = DataLoader(train_ds, batch_size=int(params["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(params["batch_size"]), shuffle=False)
    # test_loader = DataLoader(test_ds, batch_size=int(params["batch_size"]), shuffle=False)
    logger.info(
        f"Training dataset has {len(train_ds)} samples, validation set has "
        f"{len(val_ds)}."
    )
    
    model=KAN(kan_layers_input,int(params["num_grids"]),
              params["KAN_activation_layers"],
              params["addbias"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["optimizer_weight_decay"])
    scheduler = StepLR(optimizer, step_size=params["scheduler_step_size"], gamma=params["scheduler_gamma"])    
    loss_fn = torch.nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters {num_params}")
    logger.info(model)
    
    # Start training
    logger.info("Training about to start...\n")
    train_loss_history=[]
    val_loss_history=[]
    train_accuracy_history = []
    val_accuracy_history = []
    
    torch.cuda.reset_peak_memory_stats()
    st=time()
    best_acc = 0
    NUM_EPOCHS=params["epochs"]
    patience = params["patience_for_early_stop"]  # stop if validation loss does not decrease for 5 consecutive epochs
    best_val_loss = float('inf')
    no_improve_count = 0
    for epoch in range(NUM_EPOCHS):
        train_loss, val_loss, train_acc, val_acc = train(model, 
                                                         device, 
                                                         train_loader, 
                                                         val_loader, 
                                                         optimizer, 
                                                         loss_fn, 
                                                         epoch + 1)
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Vali Loss: {val_loss:.6f} | Train Acc: {train_acc:.2f} | Vali Acc: {val_acc:.2f}")
        scheduler.step()
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_accuracy_history.append(train_acc)
        val_accuracy_history.append(val_acc)
             
        if val_acc > best_acc:
            best_acc = val_acc
            # print(f'New best Accuracy : {best_acc:.5f}')
            logger.info(f"New best Accuracy : {best_acc:.5f}")
            info = update_info()
            save(save_top_model, "val accuracy", "best", best_acc)
            
        if val_acc == 1:
            break
        # --- Early stopping logic ---
        if val_loss < best_val_loss - 1e-6:  # small threshold to avoid floating noise
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            logger.info(f"No improvement in validation loss for {no_improve_count} consecutive epochs.")

        if no_improve_count >= patience:
            logger.info(f"Stopping early at epoch {epoch+1} (no validation loss improvement in {patience} steps).")
            break
    logger.info("Training completed.")
    logger.info(f"Training time (sec): {time()-st:.2f}")
    logger.info(f"Best validation accuracy= {best_acc*100:.2f} %")
    
    plt.clf()
    f,ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].plot(train_loss_history,label='Train mse loss')
    ax[0].plot(val_loss_history,label='Validation mse loss')
    ax[0].set_xlabel("Epochs")
    ax[0].legend()
    
    ax[1].plot(train_accuracy_history,label='Train accuracy')
    ax[1].plot(val_accuracy_history,label='Validation accuracy')
    ax[1].set_xlabel("Epochs")
    ax[1].legend()
    plt.savefig(model_dir+'/'+training_name+'_plots.png')
    print(f'Plots saved to {model_dir}/{training_name}_plots.png')
    plt.close()
    
if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()

    # reset seeds
    set_seed(42)  
    preds = []
    labels = []


    # run the training
    main(
        args.model_path,
        args.params_filepath,
        args.training_name
    )