import torch
from ecg_jepa.ecg_jepa import ecg_jepa, ECGJepaClassifier, ECGJepaFeatureClassifier

def load_encoder(ckpt_dir, config, feature_classification=False, minute_aggregation=False):

    if config.leads is None:
        config.leads = [0,1,2,3,4,5,6,7]

    num_patches = config.max_length_signal // config.patch_size

    params = {
        'encoder_embed_dim': 768,
        'encoder_depth': 12,
        'encoder_num_heads': 16,
        'predictor_embed_dim': 384,
        'predictor_depth': 6,
        'predictor_num_heads': 12,
        'c': 8,
        'p': num_patches,
        'pos_type': 'sincos',
        'mask_scale': (0, 0),
        'leads': config.leads,
        'drop_path_rate': config.drop_path_prob,
    }
    encoder = ecg_jepa(**params).encoder
    ckpt = torch.load(ckpt_dir)
    # drop the pos embedding
    ckpt['encoder'].pop('pos_embed', None)
    msg = encoder.load_state_dict(ckpt['encoder'], strict=False)
    print(msg)
    # check if all params require grad

    if feature_classification:
        model = ECGJepaFeatureClassifier(
            encoder, 
            config.num_classes, 
            patch_size=config.patch_size, 
            linear_probing=config.linear_probing, 
            r_peaks_detection=config.r_peaks_detection,
            minute_aggregation=minute_aggregation,
        )
    else:
        model = ECGJepaClassifier(
            encoder, 
            config.num_classes, 
            patch_size=config.patch_size, 
            linear_probing=config.linear_probing, 
            use_batch_norm_jepa=config.use_batch_norm_jepa
        )

    return model