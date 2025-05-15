# MIT-BIH_ecg_arrhytmia


### Remove an undesired file from git history
```bash
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch checkpoints/epoch=0-step=170.ckpt" --prune-empty --tag-name-filter cat -- --all
```


## removed file from code:
Removing 1080 records (1460876, ...)
Original length: 345779
New length: 344699

## Observation on hyperparameters

- *ema_0*: 0.99 seems to work well, need to understand if higher values leads to better result (currently running)
- *lr*: in combination with 0.99 ema_0 is good but need to understan if higher values are better (lower are not)
- *drop_path*: smaller values seems to work better

THE PARAMETERS ABOVE ARE GOOD FOR TRAINING BUT FOR THE KNN THESE OBSERVATION ARE USELESS