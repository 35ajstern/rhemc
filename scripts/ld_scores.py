import numpy as np
import glob
from matplotlib import pyplot as plt
import scipy.stats as stats
from numba import njit,jit
import pandas as pd
import sys

locusDir = sys.argv[1]

# eps is correlation in noise across traits
eps = 0.1 
n = 2000 

def posns_filtered(posns,freqs,start=1000000,end=3000000,maf=0.00):
    ifilt =  ( (posns > start) & (posns < end) & (freqs > maf) & (freqs < 1-maf) ) 
    posnsFilt = posns[ifilt]
    return posnsFilt

def freq(genoMat):
    n = genoMat.shape[1]
    return np.sum(genoMat,axis=1)/n

def freqs_filtered(posns,freqs,start=1000000,end=3000000,maf=0.00):
    ifilt = ( (posns > start) & (posns < end) & (freqs > maf) & (freqs < 1-maf) ) 
    freqsFilt = freqs[ifilt]
    return freqsFilt 

@njit(cache=True)
def r2(genoMat,posnFocal,posns,freqs,start=1000000,end=3000000,maf=0.00):
    ifilt = ( (posns > start) & (posns < end) & (freqs > maf) & (freqs < 1-maf) )
    posnsFilt = posns[ifilt]
    ifiltfocal = list(posnsFilt).index(posnFocal)
    genoMatFilt = genoMat[ifilt,:]
    l = genoMatFilt.shape[0]
    r2vec = np.zeros(l)
    n = genoMatFilt.shape[1]
    rowa = genoMatFilt[ifiltfocal,:]
    for j in range(genoMatFilt.shape[0]):
            rowb = genoMatFilt[j,:] 
            pab = (rowa & rowb).sum()/n
            pa = rowa.sum()/n
            pb = rowb.sum()/n
            #print(pab,pa,pb)
            r2el = ((pab - pa*pb)/np.sqrt(pa*(1-pa)*pb*(1-pb)))
            r2vec[j] = r2el
    return r2vec

@njit(cache=True)
def r2mat(genoMatCpy,posns,freqs,start=1000000,end=3000000,maf=0.00):
    genoMat = genoMatCpy
    ifilt = ( (posns > start) & (posns < end) & (freqs > maf) & (freqs < 1-maf) )
    posnsFilt = posns[ifilt]
    genoMatFilt = genoMat[ifilt,:]
    l = genoMatFilt.shape[0]
    r2 = np.zeros((l,l))
    n = genoMatFilt.shape[1]
    for i in range(l):
        rowa = genoMatFilt[i,:]
        for j in range(i+1,l):
                rowb = genoMatFilt[j,:]
                pab = np.sum((rowa * rowb))/n
                pa = np.sum(rowa)/n
                pb = np.sum(rowb)/n
                #print(pab,pa,pb)
                r2el = ((pab - pa*pb)/np.sqrt(pa*(1-pa)*pb*(1-pb)))**2
                r2[i,j] = r2el
                r2[j,i] = r2el
    for i in range(l):
        r2[i,i] = 1.0
    return r2

traitDir = '/'.join(locusDir.split('/')[:-2])
causal_df = pd.read_csv(traitDir+'/metadata.tsv',sep='\t',index_col=0)
columns = causal_df.columns
marginal_df = pd.DataFrame(columns=['DAF','LD_score','NCP']) 

sitesFile = locusDir+'relate.haps'
ancAllele = '0'
derAllele = '1'
genoMat = []
posns = []
freqs = []

focalPosn = 2000000
k = int(locusDir.split('_')[-1].split('/')[0])
key = 'ld_%d'%(k)
beta = causal_df.loc[key]['beta']
pval = causal_df.loc[key]['pval']
se = causal_df.loc[key]['se']
freqCausal = float(causal_df.loc[key]['minor_AF'])
if causal_df.loc[key]['minor_allele'] != 'T':
	freqCausal = 1.0-freqCausal

SITESFILE = open(sitesFile,'r')
SITESFILELINES = SITESFILE.readlines()

for ll,line in enumerate(SITESFILELINES):
	print('parsing line %d of %d...'%(ll,len(SITESFILELINES)))
	if line[0] == 'N' or line[0] == 'R':
		continue

	cols = line.rstrip().split(' ')
	posn = int(cols[2])
	if posn == focalPosn:
		iFocal = len(posns)
	alleles = ''.join(cols[5:])

	if alleles == ancAllele*len(alleles) or alleles == derAllele*len(alleles):
		continue
	genoMat.append([0 if char == ancAllele else 1 for char in alleles])
	posns.append(posn)
genoMat = np.array(genoMat)
SITESFILE.close()
freqs = freq(genoMat)
freqs = np.array(freqs)
posns = np.array(posns)

r_vec_causal = r2(genoMat,2000000,posns,freqs)
#r2matrix = r2mat(genoMat,posns,freqs)
posnsFilt = posns_filtered(posns,freqs)
freqsFilt = freqs_filtered(posns,freqs)
L = len(posnsFilt) 
assert(L==len(posnsFilt))

causalZ = beta/se * (10**5)**-0.5

for l in range(L):
	freq_tag = freqsFilt[l]
	posn_tag = posnsFilt[l]
	if posn_tag < 1e6 or posn_tag > 3e6:
		continue	

	in_window = np.abs(posnsFilt - posn_tag) < 1e6
	r_vec_tag = r2(genoMat,posn_tag,posns,freqs)	
	ld_score = np.sum(r_vec_tag[in_window]**2)
	
	z_tag = causalZ*r_vec_causal[l]
	marginal_df.loc[posn_tag] = [freq_tag,ld_score,z_tag]
	if freq_tag > 0.01:
		print(posn_tag,freq_tag,ld_score,z_tag)

marginal_df.to_csv(locusDir + 'ld_scores.tsv', sep='\t')
