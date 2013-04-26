# -*- coding: utf-8 -*-

import cPickle
import os
import StringIO
import tarfile


def dump_tar_bz2(obj, path):
    """
    Save object to a .tar.bz2 file.

    The file stored within the .tar.bz2 has the same basename as 'path', but
    ends with '.pkl' instead of '.tar.bz2'.

    :param obj: Object to be saved.

    :param path: Path to the file (must end with '.tar.bz2').
    """
    assert path.endswith('.tar.bz2')
    pkl_name = os.path.basename(path)[0:-8] + '.pkl'
    # We use StringIO to avoid having to write to disk a temporary
    # pickle file.
    obj_io = None
    f_out = tarfile.open(path, mode='w:bz2')
    try:
        obj_str = cPickle.dumps(obj)
        obj_io = StringIO.StringIO(obj_str)
        tarinfo = tarfile.TarInfo(name=pkl_name)
        tarinfo.size = len(obj_str)
        f_out.addfile(tarinfo=tarinfo, fileobj=obj_io)
    finally:
        f_out.close()
        if obj_io is not None:
            obj_io.close()


def get_dataset_base_path():
    return os.environ.get('ML_DATA_PATH')


def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval


def load_tar_bz2(path):
    """
    Load a file saved with `dump_tar_bz2`.
    """
    assert path.endswith('.tar.bz2')
    name = os.path.split(path)[-1].replace(".tar.bz2", ".pkl")
    f = tarfile.open(path).extractfile(name)
    try:
        data = f.read()
    finally:
        f.close()
    return cPickle.loads(data)
