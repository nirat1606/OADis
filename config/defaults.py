from yacs.config import CfgNode as CN


_C = CN()
_C.config_name = 'OADIS'

# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
_C.DATASET = CN(new_allowed=True)
_C.DATASET.name = 'mitstates'
_C.DATASET.root_dir = '/CODE/LOCATION'
_C.DATASET.splitname = 'compositional-split-natural'

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
_C.MODEL = CN(new_allowed=True)
_C.MODEL.name = 'OADIS'
_C.MODEL.load_checkpoint = False
_C.MODEL.weights = ''
_C.MODEL.optim_weights = ''

# -----------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------
_C.TRAIN = CN(new_allowed=True)

_C.TRAIN.log_dir = '/Code/'
_C.TRAIN.checkpoint_dir = '/Code/checkpoints'
_C.TRAIN.seed = 124
_C.TRAIN.num_workers = 4

_C.TRAIN.test_batch_size = 32
_C.TRAIN.batch_size = 256
_C.TRAIN.lr = 0.001

_C.TRAIN.disp_interval = 100
_C.TRAIN.save_every_epoch = 1
_C.TRAIN.eval_every_epoch = 1

# -----------------------------------------------------------------------
# Eval
# -----------------------------------------------------------------------
_C.EVAL = CN(new_allowed=True)
_C.EVAL.topk = 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()