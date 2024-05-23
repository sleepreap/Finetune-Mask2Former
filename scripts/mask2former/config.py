from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import sys
import os

#Training hyperparmeters
LEARNING_RATE=0.001
EPOCHS=180
PRECISION='16-mixed"
DEVICES=[2,3]
CHECKPOINT_CALLBACK = ModelCheckpoint(save_top_k=1, 
                                      monitor="valLoss", 
                                      every_n_epochs=1,  # Save the model at every epoch 
                                      save_on_train_epoch_end=True  # Ensure saving happens at the end of a training epoch
                                     )
LOGGER = CSVLogger("outputs", name="lightning_logs_csv")
#lr Scheduler
FACTOR=0.1
PATIENCE=30

#Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataset')
DATASET_DIR=dataset_path
NUM_WORKERS=4
BATCH_SIZE=2
ID2LABEL={
    0: 'Background',
    1: 'Metal Lines'
}

