import glob


def load_weights(weights_dir):
    weights_files = glob.glob('{}/*.h5'.format(weights_dir))
    p_net_weight = None
    r_net_weight = None
    o_net_weight = None
    for wf in weights_files:
        if 'p_net' in wf:
            p_net_weight = wf
        elif 'r_net' in wf:
            r_net_weight = wf
        elif 'o_net' in wf:
            o_net_weight = wf
        else:
            raise ValueError('No valid weights files found, should be p_net*.h5, r_net*.h5, o_net*.h5')

    if p_net_weight is None and r_net_weight is None and o_net_weight is None:
        raise ValueError('No valid weights files found, please specific the correct weights file directory')

    return p_net_weight, r_net_weight, o_net_weight
