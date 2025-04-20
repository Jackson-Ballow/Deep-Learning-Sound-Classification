import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import torch

class AccuracyTracker(xgb.callback.TrainingCallback):
    
    def __init__(self, dtrain, dval):
        self.dtrain = dtrain
        self.dval = dval
        self.train_labels = dtrain.get_label()
        self.val_labels = dval.get_label()
        self.train_accs = []
        self.val_accs = []
        
    def after_iteration(self, model, epoch, evals_log):
        train_preds = model.predict(self.dtrain)
        if len(train_preds.shape) > 1:
            train_preds = np.argmax(train_preds, axis=1)
        else:
            train_preds = (train_preds > 0.5).astype(int)
        
        train_acc = accuracy_score(self.train_labels, train_preds)
        self.train_accs.append(train_acc)

        if self.dval is not None:
            val_preds = model.predict(self.dval)
            if len(val_preds.shape) > 1:
                val_preds = np.argmax(val_preds, axis=1)
            else:
                val_preds = (val_preds > 0.5).astype(int)

            val_acc = accuracy_score(self.val_labels, val_preds)
            self.val_accs.append(val_acc)
        
        return False

class AudioXGBoost:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

        params = {
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'eta': 0.3,
            'max_depth': 3,
            'min_child_weight': 6,
            'gamma': 0.3,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'lambda': 5.0,
            'alpha': 1.0,
            'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
            'tree_method': 'hist',
            'device': "cuda",
        }
        print(params)

        self.params = {k: v for k, v in params.items() if v is not None}
        self.model = None
        
    def _extract_features(self, dataloader):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for X, y in dataloader:
                batch_spectrograms = X.numpy()
                batch_labels = y.numpy()
                batch_size = batch_spectrograms.shape[0]
                flattened = batch_spectrograms.reshape(batch_size, -1)
                all_features.append(flattened)
                all_labels.append(batch_labels)

        return np.vstack(all_features), np.concatenate(all_labels)
        
    def fit(self, train_loader, val_loader, num_boost_round=100, early_stopping_rounds=10):
        X_train, y_train = self._extract_features(train_loader)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        X_val, y_val = self._extract_features(val_loader)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, 'validation'))

        accuracy_tracker = AccuracyTracker(dtrain, dval)
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

        formatted_history = {
            "train_loss": history['train']['mlogloss' if self.num_classes > 2 else 'logloss'],
            "train_acc": accuracy_tracker.train_accs,
            "val_loss": history['validation']['mlogloss' if self.num_classes > 2 else 'logloss'],
            "val_acc": accuracy_tracker.val_accs,
        }
        return formatted_history
        
    def predict(self, dataloader):
        X, _ = self._extract_features(dataloader)
        dtest = xgb.DMatrix(X)

        if self.num_classes > 2:
            probs = self.model.predict(dtest)
            preds = np.argmax(probs, axis=1)
        else:
            probs = self.model.predict(dtest)
            preds = (probs > 0.5).astype(int)
            
        return preds
        
    def __call__(self, X):
        batch_spectrograms = X.cpu().numpy()
        batch_size = batch_spectrograms.shape[0]

        flattened = batch_spectrograms.reshape(batch_size, -1)
        dmatrix = xgb.DMatrix(flattened)
        
        if self.num_classes > 2:
            probs = self.model.predict(dmatrix)
            output = torch.from_numpy(probs).float()
        else:
            probs = self.model.predict(dmatrix)
            probs_tensor = torch.from_numpy(probs).float().view(-1, 1)
            output = torch.cat([1 - probs_tensor, probs_tensor], dim=1)
            
        if X.is_cuda:
            output = output.cuda()
            
        return output