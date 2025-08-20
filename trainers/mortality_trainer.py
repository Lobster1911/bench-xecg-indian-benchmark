from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
import numpy as np
import torch
import trainers.common as common
from trainers.common_trainer import CommonTrainerDownstream
from utils.train_utils import focal_loss
from lifelines.utils import concordance_index
from torchmetrics import Metric



class TrainerMortality(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.loss_fn = CoxProportionalHazardsLoss()

        self.train_ci = ConcordanceIndexMetric()
        self.val_ci = ConcordanceIndexMetric()
        self.test_ci = ConcordanceIndexMetric()

    def training_step(self, batch, _):
        loss, logits, batch = self.predict_batch(batch)
        self.log("train_loss", loss, prog_bar=True)

        self.train_ci.update(logits, batch["timey"], batch["death"])
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_ci", self.train_ci.compute(), prog_bar=True)
        self.train_ci.reset()
        return super().on_train_epoch_end()
    
    def validation_step(self, batch, _):
        loss, logits, batch = self.predict_batch(batch)
        self.log("val_loss", loss, prog_bar=True)

        self.val_ci.update(logits, batch["timey"], batch["death"])
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_ci", self.val_ci.compute(), prog_bar=True)
        self.val_ci.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch, _):
        loss, logits, batch = self.predict_batch(batch)
        self.log("test_loss", loss)

        self.test_ci.update(logits, batch["timey"], batch["death"])
        return loss
    
    def on_test_epoch_end(self):
        self.log("test_ci", self.test_ci.compute(), prog_bar=True)
        self.test_ci.reset()
        return super().on_test_epoch_end()

    def sort_batch(self, batch, return_ind=False):
        """Sort batch by follow up time (descending)"""
        fu_time = batch["timey"]
        ind = np.argsort(fu_time.cpu())
        ind = torch.flip(ind, dims=[0])
        out = {k: v[ind] for k, v in batch.items()}
        if return_ind:
            return out, ind
        else:
            return out
            
    
    def predict_batch(self, batch):
        batch = self.sort_batch(batch)

        x = batch["signals"]
        # if i have nan print
        if torch.isnan(x).any():
            print("NaN found in input signals")
            nan_count = torch.sum(torch.isnan(x)).item()
            print(f"Number of NaNs in input signals: {nan_count}")

        death = batch["death"]
        if torch.isnan(death).any():
            print("NaN found in death labels")

        if torch.isnan(batch['timey']).any():
            print("NaN found in timey labels")

        # timey = batch["timey"]
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x)

        if torch.isnan(logits).any():
            print("NaN found in logits")

        loss = self.loss_fn(death, logits)

        return loss, logits, batch



class CoxProportionalHazardsLoss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        super(CoxProportionalHazardsLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def _negative_log_likelihood(self, censor_status, risk):
        """
        Define Cox PH partial likelihood function loss.

        Taken from "https://github.com/UK-Digital-Heart-Project/4Dsurvival/blob/master/survival4D/nn.py"
        Arguments: censor_status (censoring status) bool, risk (risk [log hazard ratio] predicted by network) for batch of input subjects
        As defined, this function requires that all subjects in input batch must be sorted in descending order of
        followup time
        """
        risk = risk.squeeze()
        risk = torch.sigmoid(risk)
        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=-1))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * censor_status
        neg_likelihood = -censored_likelihood
        return neg_likelihood

    def forward(self, censor_status, risk):
        neg_likelihood = self._negative_log_likelihood(censor_status, risk)

        if self.reduction == "mean":
            loss = torch.mean(neg_likelihood)
        elif self.reduction == "sum":
            loss = torch.sum(neg_likelihood)
        elif self.reduction == "none":
            loss = neg_likelihood

        return loss
    

class ConcordanceIndexMetric(Metric):
    """
    TorchMetrics-compatible metric for concordance index (C-index)
    using lifelines.utils.concordance_index.


    Stores predictions/targets across all batches and computes
    the C-index at the end of an epoch.
    """


    full_state_update = True # ensures we aggregate across batches


    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)


        # Add states for predictions, durations, and events
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("durations", default=[], dist_reduce_fx=None)
        self.add_state("events", default=[], dist_reduce_fx=None)


    def update(self, preds: torch.Tensor, durations: torch.Tensor, events: torch.Tensor):
        """
        Args:
        preds: risk scores or predicted survival times (higher means higher risk).
        durations: observed survival times.
        events: event indicators (1 if event occurred, 0 if censored).
        """
        self.preds.append(preds.detach().cpu())
        self.durations.append(durations.detach().cpu())
        self.events.append(events.detach().cpu())


    def compute(self):
        preds = torch.cat(self.preds).numpy()
        durations = torch.cat(self.durations).numpy()
        events = torch.cat(self.events).numpy()
        c_index = concordance_index(durations, -preds, events)
        return torch.tensor(c_index, dtype=torch.float32)