import numpy as np
import matplotlib.pylab as plt
import re
import pdb

filename = 'kddcup_allinone_3000_nolog.log'
with open(filename, 'r') as logfile:
    data = logfile.read()
    training_err = []

    for m in re.finditer('Train error.*:\s+(0.[0-9]*)', data):
        print '%02d-%02d: %s' % (m.start(), m.end(), m.group(1))
        training_err.append(float(m.group(1)))
    
