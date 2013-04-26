# -*- coding: utf-8 -*-

import os
import shutil
import sys
import numpy

from ml_code import util

def main():
    data_dir = util.get_dataset_base_path()
    path = os.path.join(data_dir, 'mnist_images')
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.png')]
    possible_targets = numpy.arange(10)
    temp_data_dir = os.path.join(data_dir, 'temp')
    if not os.path.exists(temp_data_dir):
        print 'creating directory %s' %temp_data_dir
        os.makedirs(temp_data_dir)
    image_count = {}
    # image_count = {0: 6903, 1: 7877, 2: 6990, 3: 7141, 4: 6824, 5: 6313, 6: 6876, 7: 7293, 8: 6825, 9: 6958}
    for idx, file in enumerate(files):
        sys.stdout.write('\r%2d%%' % int(idx / float(len(files)) * 100 + 0.5))
        sys.stdout.flush()
        target = int(file.split('number_')[-1][0])
        assert target in possible_targets
        image_count.setdefault(target, 0)
        image_count[target] += 1
        out_path = os.path.join(temp_data_dir, 'target_%s'%target)
        if not os.path.exists(out_path):
            print 'creating directory %s' %out_path
            os.makedirs(out_path)
        out_path = os.path.join(out_path, 'image_%s.png'%image_count[target])
        shutil.copyfile(file, out_path)
    import pdb; pdb.set_trace()
if __name__=='__main__':
    sys.exit(main())