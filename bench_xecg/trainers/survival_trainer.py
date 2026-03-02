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
from utils.loss_utils import focal_loss
from lifelines.utils import concordance_index
from torchmetrics import Metric



class TrainerSurvival(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None, evaluate_music=False):
        super().__init__(model, config,  len_train_dataset, weights)

        self.loss_fn = CoxProportionalHazardsLoss()

        self.train_ci = ConcordanceIndexMetric()
        self.val_ci = ConcordanceIndexMetric()
        self.test_ci = ConcordanceIndexMetric()

        self.evaluate_music = evaluate_music
        if self.evaluate_music:
            self.test_ci_cardiac = ConcordanceIndexMetric(multiple_preds_for_same_ecg=True)
            self.test_ci = ConcordanceIndexMetric(multiple_preds_for_same_ecg=True)


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

        if self.evaluate_music:
            self.test_ci_cardiac.update(logits, batch["timey"], batch["cardiac_death"], batch['subj'])
            self.test_ci.update(logits, batch["timey"], batch["death"], batch['subj'])
        else:
            self.test_ci.update(logits, batch["timey"], batch["death"])
        return loss
    
    def on_test_epoch_end(self):
        if self.evaluate_music:
            self.log("test_ci_music_cardiac", self.test_ci_cardiac.compute(), prog_bar=True)
            self.log("test_ci_music", self.test_ci.compute(), prog_bar=True)
            self.test_ci_cardiac.reset()
            self.test_ci.reset()
        else: 
            self.log("test_ci", self.test_ci.compute(), prog_bar=True)
            self.test_ci.reset()

        return super().on_test_epoch_end()

    def sort_batch(self, batch, return_ind=False):
        """Sort batch by follow up time (descending)"""
        fu_time = batch["timey"]
        ind = np.argsort(fu_time.cpu())
        ind = torch.flip(ind, dims=[0])

        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                # already a tensor → can index directly
                out[k] = v[ind]
            elif isinstance(v, np.ndarray):
                # numpy → convert index to numpy
                out[k] = v[ind.numpy()]
            elif isinstance(v, list):
                # python list → gather manually
                out[k] = [v[i] for i in ind.tolist()]
            else:
                raise TypeError(f"Unsupported type for key {k}: {type(v)}")
        # out = {k: v[ind.tolist()] for k, v in batch.items()}
        if return_ind:
            return out, ind
        else:
            return out
            
    
    def predict_batch(self, batch):
        batch = self.sort_batch(batch)

        x = batch["signals"]
        death = batch["death"]

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x)
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


    def __init__(self, dist_sync_on_step=False, multiple_preds_for_same_ecg=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)


        # Add states for predictions, durations, and events
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("durations", default=[], dist_reduce_fx=None)
        self.add_state("events", default=[], dist_reduce_fx=None)

        self.multiple_preds_for_same_ecg = multiple_preds_for_same_ecg
        if multiple_preds_for_same_ecg:
            self.add_state("subjs", default=[], dist_reduce_fx=None)


    def update(self, preds: torch.Tensor, durations: torch.Tensor, events: torch.Tensor, subjs = None):
        """
        Args:
        preds: risk scores or predicted survival times (higher means higher risk).
        durations: observed survival times.
        events: event indicators (1 if event occurred, 0 if censored).
        """
        self.preds.append(preds.detach().cpu())
        self.durations.append(durations.detach().cpu())
        self.events.append(events.detach().cpu())

        if self.multiple_preds_for_same_ecg:
            assert subjs is not None, "subjs must be provided if multiple_preds_for_same_ecg is True"
            self.subjs.extend(subjs)


    def compute(self):
        if not self.multiple_preds_for_same_ecg:
            preds = torch.cat(self.preds).numpy()
            durations = torch.cat(self.durations).numpy()
            events = torch.cat(self.events).numpy()
            c_index = concordance_index(durations, -preds, events)
        else:
            # i need to aggregate and take the mean for every subject
            preds = torch.cat(self.preds).numpy()
            durations = torch.cat(self.durations).numpy()
            events = torch.cat(self.events).numpy()

            # subjs contain as keys the subject ids and as values the indexes where the subject id was appearing
            subjects = {}
            for i, subj in enumerate(self.subjs):
                if subj not in subjects.keys():
                    subjects[subj] = []
                subjects[subj].append(i)

            # preds is average over the indexes for each subject
            preds_agg = []
            durations_agg = []
            events_agg = []
            for subj, indexes in subjects.items():
                preds_agg.append(np.mean(preds[indexes]))
                durations_agg.append(durations[indexes[0]])
                events_agg.append(events[indexes[0]])

            c_index = concordance_index(durations_agg, -np.array(preds_agg), events_agg)

        return torch.tensor(c_index, dtype=torch.float32)