from typing import *

import os
import json
import jsonpickle


def json_dump(data, path, fname):
    fpath = os.path.join(path, fname)
    with open(fpath, 'w+') as f:
        frozen = jsonpickle.encode(data)
        json.dump(frozen, f, sort_keys=True)

def json_load(fpath):
    with open(fpath, 'r') as f:
        frozen = json.load(f)
        data = jsonpickle.decode(frozen)
    
    return data

