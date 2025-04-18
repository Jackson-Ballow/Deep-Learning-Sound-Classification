# xgboost_model.py
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader

class AccuracyTracker(xgb.callback.TrainingCallback):
    """Custom callback to track accuracy during XGBoost training"""
    
    def __init__(self, dtrain, dval=None):
        self.dtrain = dtrain
        self.dval = dval
        self.train_labels = dtrain.get_label()
        self.val_labels = dval.get_label() if dval is not None else None
        self.train_accs = []
        self.val_accs = []
        
    def after_iteration(self, model, epoch, evals_log):
        # Get training predictions
        train_preds = model.predict(self.dtrain)
        if len(train_preds.shape) > 1:  # multi-class case
            train_preds = np.argmax(train_preds, axis=1)
        else:  # binary case
            train_preds = (train_preds > 0.5).astype(int)
        
        # Calculate training accuracy
        train_acc = accuracy_score(self.train_labels, train_preds)
        self.train_accs.append(train_acc)
        
        # Calculate validation accuracy if validation data is provided
        if self.dval is not None:
            val_preds = model.predict(self.dval)
            if len(val_preds.shape) > 1:  # multi-class case
                val_preds = np.argmax(val_preds, axis=1)
            else:  # binary case
                val_preds = (val_preds > 0.5).astype(int)
            
            val_acc = accuracy_score(self.val_labels, val_preds)
            self.val_accs.append(val_acc)
        
        return False  # Continue training

class AudioXGBoost:
    def __init__(self, num_classes=2, **kwargs):
        """
        Initialize XGBoost model for audio classification
        
        Args:
            num_classes: Number of classes for classification
            **kwargs: Additional parameters for XGBoost
        """
        self.num_classes = num_classes
        
        # Set default parameters
        params = {
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'eta': 0.1,
            'max_depth': 6,
            # 'min_child_weight': 3,
            # 'gamma': 0.2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            # 'lambda': 2.0,
            # 'alpha': 0.5,
            'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
            'tree_method': 'hist',  # For faster training
        }
        
        # Update with user params
        for key, value in kwargs.items():
            if key in params and value is not None:
                params[key] = value
                
        # Remove None values
        self.params = {k: v for k, v in params.items() if v is not None}
        self.model = None
        
    def to(self, device):
        """Mock method to match PyTorch API"""
        return self
        
    def train(self):
        """Mock method to match PyTorch API"""
        return self
        
    def eval(self):
        """Mock method to match PyTorch API"""
        return self
        
    def _extract_features(self, dataloader):
        """
        Extract flattened features from spectrograms in dataloader
        
        Args:
            dataloader: PyTorch DataLoader containing spectrograms
            
        Returns:
            X: numpy array of features
            y: numpy array of labels
        """
        all_features = []
        all_labels = []
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            for X, y in dataloader:
                # Get data from the dataloader
                batch_spectrograms = X.numpy()
                batch_labels = y.numpy()
                
                # Flatten the spectrograms (convert 3D to 2D)
                # Shape: (batch_size, channels, height, width) -> (batch_size, channels*height*width)
                batch_size = batch_spectrograms.shape[0]
                flattened = batch_spectrograms.reshape(batch_size, -1)
                
                all_features.append(flattened)
                all_labels.append(batch_labels)
                
        return np.vstack(all_features), np.concatenate(all_labels)
        
    def fit(self, train_loader, val_loader=None, num_boost_round=100, early_stopping_rounds=15):
        """
        Train the XGBoost model
        
        Args:
            train_loader: DataLoader with training data
            val_loader: Optional DataLoader with validation data
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Dictionary with training history
        """
        # Extract features
        X_train, y_train = self._extract_features(train_loader)
        
        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Setup validation if provided
        evals = [(dtrain, 'train')]
        dval = None
        if val_loader:
            X_val, y_val = self._extract_features(val_loader)
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'validation'))
        
        # Create accuracy tracking callback
        accuracy_tracker = AccuracyTracker(dtrain, dval)
        
        # Train the model
        history = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True,
            evals_result=history,
            callbacks=[accuracy_tracker]
        )
        
        # Convert history format to match PyTorch training
        formatted_history = {
            "train_loss": history['train']['mlogloss' if self.num_classes > 2 else 'logloss'],
            "train_acc": accuracy_tracker.train_accs,
        }
        
        if val_loader:
            formatted_history["val_loss"] = history['validation']['mlogloss' if self.num_classes > 2 else 'logloss']
            formatted_history["val_acc"] = accuracy_tracker.val_accs
        else:
            formatted_history["val_loss"] = []
            formatted_history["val_acc"] = []
            
        return formatted_history
        
    def predict(self, dataloader):
        """
        Make predictions using the trained model
        
        Args:
            dataloader: DataLoader with data to predict
            
        Returns:
            numpy array of predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        X, _ = self._extract_features(dataloader)
        dtest = xgb.DMatrix(X)
        
        if self.num_classes > 2:
            # For multiclass, get the class with highest probability
            probs = self.model.predict(dtest)
            preds = np.argmax(probs, axis=1)
        else:
            # For binary classification, threshold at 0.5
            probs = self.model.predict(dtest)
            preds = (probs > 0.5).astype(int)
            
        return preds
        
    def __call__(self, X):
        """
        Make predictions on a batch (to match PyTorch API)
        
        Args:
            X: batch of spectrograms
            
        Returns:
            model output compatible with cross-entropy loss
        """
        # Convert PyTorch tensor to numpy
        batch_spectrograms = X.cpu().numpy()
        batch_size = batch_spectrograms.shape[0]
        
        # Flatten the spectrograms
        flattened = batch_spectrograms.reshape(batch_size, -1)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(flattened)
        
        if self.num_classes > 2:
            # For multiclass, return class probabilities
            probs = self.model.predict(dmatrix)
            # Convert to PyTorch tensor with shape (batch_size, num_classes)
            output = torch.from_numpy(probs).float()
        else:
            # For binary, return logits for both classes
            probs = self.model.predict(dmatrix)
            # Convert to PyTorch tensor with shape (batch_size, 2)
            probs_tensor = torch.from_numpy(probs).float().view(-1, 1)
            output = torch.cat([1 - probs_tensor, probs_tensor], dim=1)
            
        # Move to same device as input
        if X.is_cuda:
            output = output.cuda()
            
        return output