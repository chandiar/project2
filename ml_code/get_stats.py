# -*- coding: utf-8 -*-
import sys, os
from jobman.tools import DD
from jobman import parse
import argparse
import matplotlib.pyplot as plt
import numpy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('files', type=str, nargs='*')
args = parser.parse_args()

files = []

for fname in args.files:
    d = DD(parse.filemerge(fname))
    
    if d['jobman.status'] == 5:
        continue
    if not hasattr(d,'best_test_combined'):
        continue
    if numpy.isnan(d.best_test_combined).any():
        continue
    if d.valid_diff > 0.474:
        continue

    d.filename  =   fname
    #idx = numpy.argmin(d.valid_err_list)
    #d.best_valid = d.valid_err_list[idx]

    files.append(d)    

print 'nb of successful exps.' , len(files)

idx_diff    =   numpy.argmin([f.valid_diff for f in files])
idx_fun     =   numpy.argmin([f.valid_fun for f in files])
idx_comb    =   numpy.argmin([numpy.mean(f.best_valid_combined) for f in files])


print 'Valid diff   :   ', files[idx_diff].valid_diff
print 'Valid fun    :   ', files[idx_diff].valid_fun
print 'Valid comb   :   ', files[idx_diff].best_valid_combined

print 'Test diff    :   ', files[idx_diff].test_diff
print 'Test fun     :   ', files[idx_diff].test_fun
print 'Test comb    :   ', files[idx_diff].best_test_combined


# Varied HParams
import random_sampling
to_plot = filter(lambda x:type(random_sampling.standard_config[x])==type(()),random_sampling.standard_config)

print
print '######################'
print 'Interval for best Diff'
k       =   5
top_k = [files[idx] for idx in numpy.argsort([f.valid_diff for f in files])][:k]
for p in to_plot:
    print p+'\t:', numpy.min([f[p] for f in top_k]), numpy.max([f[p] for f in top_k])


def load_costs(index):
    return numpy.load(os.path.join(top_k[index].filename.split('/')[0], 'costs.npy'))

def show_learning_curves(index):
    diff_costs = [x[:,0] for x in load_costs(index)]
    fun_costs = [x[:,1] for x in load_costs(index)]
    [plt.plot(range(len(x)), x) for x in diff_costs]
    plt.show()

import pdb; pdb.set_trace()

import sys

best        = files[idx_diff]
best_params = [best[key] for key in to_plot]

print
for (p,v) in zip(to_plot,best_params):
  print p+'\t'+`v`





x_plots = [[f[hp] for f in files] for hp in to_plot]
y_plots = [f.valid_diff for f in files]

fig = plt.figure()
i   = 1
for (x,hpname) in zip(x_plots,to_plot):
  plt.subplot(330+i)
  plt.scatter(numpy.log(x),y_plots)
  plt.title(hpname)
  i += 1
plt.show()




#files=sorted(files,key=lambda x:x[0].best_valid,reverse=True)
#files=sorted(files,key=lambda x:x[0].test_err_list[0],reverse=True)
#files=sorted(files,key=lambda x:x[0].suggestions[5],reverse=True)


files=sorted(files,key=lambda x:x.best_valid,reverse=True)
#files=sorted(files,key=lambda x:x.suggestions[-1])


#files=files[-50:]
files=files[-20:]
#print files[0].suggestions
#print files[0].best_valid
#print files[0].test_err_list

e0=sorted(map(lambda x:x.e0,files))
h=sorted(map(lambda x:x.h,files))
L1=sorted(map(lambda x:x.L1_reg,files))
L2=sorted(map(lambda x:x.L2_reg,files))
tau=sorted(map(lambda x:x.tau,files))




###
v=map(lambda x:x.best_valid,files)
#fivestar=map(lambda x:x.suggestions[5],files)

#print 'correlation'

#print v

#print numpy.cov(zip(v,fivestar))



#print numpy.correlate(fivestar,v)


###

try:
  alpha=sorted(map(lambda x:x.alpha,files))
  beta=sorted(map(lambda x:x.beta,files))  
except:
  nothing=0
try:
  bal=sorted(map(lambda x:x.bal,files))
except:
  nothing=0
print 'hyperp intervals for best'
print 'e0',minmax(e0)
print 'h',minmax(h)
print 'L1',minmax(L1)
print 'L2',minmax(L2)
print 'tau',minmax(tau)
try:
  print 'alpha',minmax(alpha)
  print 'beta',minmax(beta)
except:
  nothing=0
try:
  print 'bal',minmax(bal)
except:
  nothing=0
fig = plt.figure()
x=map(lambda x:x.best_valid,files)
#x=map(lambda x:x.suggestions[-1],files)

y1=map(lambda x:numpy.log(x.L1_reg),files)
y2=map(lambda x:numpy.log(x.L2_reg),files)
y3=map(lambda x:x.h,files)
y4=map(lambda x:numpy.log(x.e0),files)
y5=map(lambda x:numpy.log(x.tau),files)
#y6=map(lambda x:x.suggestions[-1],files)


#plt.scatter(y6,x)
#plt.title('MSE valid vs %5*')
#plt.xlabel('%5*')
#plt.ylabel('MSE valid')
#plt.show()



try:
  y7=map(lambda x:x.alpha,files)
  y8=map(lambda x:x.beta,files)
except:
  nothing=0  
try:
  y9=map(lambda x:x.bal,files)
except:
  nothing=0
#plt.figure(1)
#plt.subplot(331)
#plt.scatter(y1,x)
#plt.title('L1')

#plt.subplot(332)
#plt.scatter(y2,x)
#plt.title('L2')

plt.subplot(333)
plt.scatter(y3,x)
plt.title('h')

plt.subplot(334)
plt.scatter(y4,x)
plt.title('e0')

plt.subplot(335)
plt.scatter(y5,x)
plt.title('tau')

#plt.subplot(336)
#plt.scatter(y6,x)
#plt.title('%5*')

try:
  plt.subplot(337)
  plt.scatter(y7,x)
  plt.title('alpha')

  plt.subplot(338)
  plt.scatter(y8,x)
  plt.title('beta')
  
except:
  nothing=0
try:
  plt.subplot(339)
  plt.scatter(y9,x)
  plt.title('bal')
except:
  nothing=0
plt.show()
