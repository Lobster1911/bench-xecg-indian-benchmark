# MIT-BIH_ecg_arrhytmia


### Remove an undesired file from git history
```bash
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch checkpoints/epoch=0-step=170.ckpt" --prune-empty --tag-name-filter cat -- --all
```