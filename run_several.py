import subprocess
import yaml
import re
import os
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True, help='GPU number')
    parser.add_argument('--opts', type=str, required=True, help='path to json with pathes to yamls????')
    args = parser.parse_args()
    with open(args.opts, 'r') as f:
        opts = json.load(f)
    for opt in opts:
        opt_name = f"options/iseebetter/{opt}"
        with open(opt_name, "r") as file:
            loader = yaml.SafeLoader
            loader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(u'''^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
                list(u'-+0123456789.'))
            data = yaml.load(file, Loader=loader)
            data['gpu_ids'] = [args.gpu]
        opt_name = f"options/iseebetter/tmp_{opt}"
        with open(opt_name, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
        
        subprocess.run(["python", "train_model.py", "--opt", opt_name])
        os.remove(opt_name)