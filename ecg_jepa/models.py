import torch
from ecg_jepa.ecg_jepa import ecg_jepa, ECGJepaClassifier, ECGJepaFeatureClassifierMIT_BIH

def load_encoder(ckpt_dir, num_classes=5, leads=None, drop_path_rate=0.0, feature_classification=False, linear_probing=False):

    if leads is None:
        leads = [0,1,2,3,4,5,6,7]

    params = {
        'encoder_embed_dim': 768,
        'encoder_depth': 12,
        'encoder_num_heads': 16,
        'predictor_embed_dim': 384,
        'predictor_depth': 6,
        'predictor_num_heads': 12,
        'c': 8,
        'pos_type': 'sincos',
        'mask_scale': (0, 0),
        'leads': leads,
        'drop_path_rate': drop_path_rate
    }
    encoder = ecg_jepa(**params).encoder
    ckpt = torch.load(ckpt_dir)
    encoder.load_state_dict(ckpt['encoder'])
    # check if all params require grad

    if feature_classification:
        model = ECGJepaFeatureClassifierMIT_BIH(encoder, num_classes, patch_size=75, linear_probing=linear_probing)
    else:
        model = ECGJepaClassifier(encoder, num_classes, patch_size=75, linear_probing=linear_probing)

    return model