from lib import implicit_funcitions
from .map3d import *


def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = max(current_metadata.get('render_width', current_metadata['gen_width']),
                       current_metadata.get('render_height', current_metadata['gen_height']))
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        curriculum_size = max(curriculum[curriculum_step].get('render_width', 512),
                              curriculum[curriculum_step].get('render_height', 512))
        if curriculum_step > current_step and  curriculum_size > current_size:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = max(current_metadata.get('render_height', current_metadata['gen_width']),
                       current_metadata.get('render_width', current_metadata['gen_height']))
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        curriculum_size = max(curriculum[curriculum_step].get('render_width', current_metadata['gen_width']),
                              curriculum[curriculum_step].get('render_height', current_metadata['gen_height']))
        if curriculum_step <= current_step and curriculum_size == current_size:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


def get_config(opt):

    config = globals()[opt.config]
    config['neural_field_cls'] = getattr(implicit_funcitions, config['neural_field_cls'])

    if len(opt.tune) == 0:
        pass

    elif opt.tune == 'lr':
        variants = [
            (1e-4, 4e-4), (2e-4, 2e-4),
            (1e-4, 2e-4), (1e-4, 1e-4),
        ]
        gen_lr, disc_lr = variants[opt.variant]
        for key in config.keys():
            if isinstance(key, int):
                config[key]['gen_lr'] = gen_lr
                config[key]['disc_lr'] = disc_lr
        config['name'] = "{}_G_lr={}_D_lr={}".format(config["name"], gen_lr, disc_lr)
    elif opt.tune == 'map3d_mode':
        variants = ["isolated", "mixed", "all"]
        map3d_mode = variants[opt.variant]
        config['map3d_mode'] = map3d_mode
        config['name'] = "{}_map3d_mode={}".format(config["name"], map3d_mode)
    else:
        raise NotImplementedError

    return config