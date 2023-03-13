import os

import torch


UNIGRAM1000_LIST = (
    ["<blank>"]
    + [
        _.split()[0]
        for _ in open(
            os.path.join(os.path.dirname(__file__), "labels", "unigram1000_units.txt")
        )
        .read()
        .splitlines()
    ]
    + ["<eos>"]
)


# Writes list of objects (anything that can be converted to str) to a txt file, separated by "\n"s
def write_to_txt(obj_list, path):
    f = open(path, "w")
    for obj in obj_list:
        f.write(str(obj) + "\n")
    f.close()


def ids_to_str(token_ids, char_list):
    tokenid_as_list = list(map(int, token_ids))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    return "".join(token_as_list).replace("<space>", " ")


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def average_checkpoints(last):
    avg = None
    for path in last:
        states = torch.load(path)["state_dict"]
        states = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]

    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= len(last)
            else:
                avg[k] //= len(last)
    return avg


def get_param_groups(
    model,
    num_blocks,
    base_lr_enc,
    base_lr_other,
    lr_decay_rate,
    ctc_equals_other=True,
    min_lr=1e-6,
):
    param_groups = {}
    layer_scales = list(
        lr_decay_rate ** (num_blocks - i - 1) for i in range(num_blocks)
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("encoder.after_norm"):
            group_name = "after_norm"
            base_lr = max(base_lr_enc, min_lr)
        elif name.startswith("encoder.embed"):
            group_name = "embed"
            base_lr = max(layer_scales[0] * base_lr_enc, min_lr)
        elif name.startswith("encoder.frontend"):
            group_name = "frontend"
            base_lr = max(layer_scales[0] * base_lr_enc, min_lr)
        elif name.startswith("encoder.encoders"):
            group_id = int(name.split(".")[2])
            group_name = f"block_{group_id}"
            base_lr = max(layer_scales[group_id] * base_lr_enc, min_lr)
        elif name.startswith("ctc"):
            group_name = "ctc"
            base_lr = (
                max(base_lr_other, min_lr)
                if ctc_equals_other
                else max(base_lr_enc, min_lr)
            )
        else:
            assert not name.startswith("encoder")
            group_name = "other"
            base_lr = max(base_lr_other, min_lr)

        if group_name not in param_groups:
            param_groups[group_name] = {"name": group_name, "lr": base_lr, "params": []}
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())
