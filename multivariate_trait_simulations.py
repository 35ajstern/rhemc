import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import gamma
import os
import subprocess
import pandas as pd
import argparse

#### negative selection functions ###

def P(f,Ne=10**4):
    val = (np.log(f) + np.log(2*Ne))/np.log(2*Ne)
    return val

def Pinv(g,Ne=10**4):
    val = (2*Ne)**(g-1)
    return val

def sfs_sel(p,s,Ne=10**4):   
    return 1/(s*p*(1-p))*np.exp(-2*Ne*s*p)

def expected_chi2_neu(Ne=10**4):
    valNeu = 1/np.log(2*Ne)
    return valNeu

def expected_chi2(tau,a,sbar,Ne=10**4):
    b = a/sbar
    valSel = gamma(2*tau+a)/gamma(a) / (2*Ne*(2*tau+a-1)) * (1/b**(2*tau-1)-b**a/((b+2*Ne)**(2*tau+a-1)))
    return valSel

def rejection_sampling(s,n,Ne=10**4):
    ndone = 0
    M = s * 10**4
    samples = []
    while ndone < n:
        pi = Pinv(np.random.uniform())
        u = np.random.uniform()
        if u < sfs_sel(pi,s)/(M*1/pi/np.log(2*Ne)):
            samples.append(pi)
            ndone = len(samples)
    return samples

def sample_freq_and_beta(tau,scale,s,h2,M,Ne=10**4,size=1):
    freqs = []
    betas = []
    ss = []
    for i in range(size):
        if tau == 0.0: 
            # just sample neutral SFS and indy beta
            c = 1/(2*Ne)
            freq = random_neutral_freq(Ne,c,size=1)[0]
            V = 1/np.log(2*Ne)
            #print(np.sqrt(h2/(V*M)),V)
            beta = np.random.normal(0,scale=np.sqrt(h2/(V*M)))
            srand = 0
        else:
            # draw s from gamma distribution
            sbar = s/(2*Ne) #/(M**(1/(2*tau)))
            srand = np.random.gamma(scale,sbar/scale)
            freq = rejection_sampling(srand,1,Ne=Ne)[0]
            beta2 = gamma(2*tau+scale)/gamma(scale) * (2*Ne)**(-2*tau) * (freq + scale/(2*Ne*sbar))**(-2*tau)
            beta = np.random.normal(0,np.sqrt(beta2)) 

            # calculate c
            #x2 = expected_chi2(tau,scale,2*Ne*sbar,Ne=Ne)
            #c = h2/(2*Ne*x2)

            # draw draw freq | s
            #print(srand)
            #freq = rejection_sampling(srand,1,Ne=Ne)[0]

            # compute beta from s,c
            #rademacher = 2*np.random.binomial(1,0.5)-1
            #beta = rademacher * np.sqrt(c * srand**(2*tau))
        freqs.append(freq)
        betas.append(beta)
        ss.append(srand)
    return np.array(freqs), np.array(betas), np.array(ss)

########


def random_neutral_freq(N,c,size=1):
	probs = 1/np.arange(np.ceil(2*N*c),np.floor(2*N*(1-c))+1)
	probs *= 1/np.sum(probs)
	choice = np.random.choice(np.arange(np.ceil(2*N*c),np.floor(2*N*(1-c))+1), size, p=probs)/(2*N)
	return choice

def simulate_beta_neutral(h2,V,M):
	beta = np.random.normal(0.0,np.sqrt(h2/(V*M)))
	return beta

def simulate_selected_backwards(p0,s_locus,N=10000):
    delta = 1/(2*N)
    traj = [p0]
    a = s_locus*N*2
    while traj[-1] != 1 and traj[-1] != 0:
        curr = traj[-1]
        nextFreq = np.random.normal(-a*curr*(1-curr)/np.tanh(a*curr)*delta + curr, np.sqrt(delta) * np.sqrt(curr*(1-curr)) )
        if nextFreq > 1:
            nextFreq = 1
        if nextFreq < 0:
            nextFreq = 0
        traj.append(nextFreq)
    return traj[1:]

def simulate_selected_forwards(p0,s_locus,tOn,tOff,timeBins,N):
	#Generate an allele frequency trajectory forward in time under selection.
	#p0 is the frequency of the allele when selection starts.
	#N is the (constant, diploid) population size.
	#delta is the interval between time steps.
	traj = [p0]
	time = tOn
	currTime = tOn
	while time > 0:
		Nt = N[np.digitize(currTime,timeBins)-1]
		delta = 1/(2*Nt)
		#change s depending on pulse timing
		if time > tOff:
			s_t = s_locus
		else:
			s_t = 0
		currFreq = traj[-1]
		if(currFreq > 0 and currFreq < 1):
			nextFreq = np.random.normal(currFreq + delta*2*Nt*s_t*currFreq*(1-currFreq), np.sqrt(delta) * np.sqrt(currFreq*(1-currFreq)) )
			if nextFreq > 1:
				nextFreq = 1
			if nextFreq < 0:
				nextFreq = 0
			traj.append(nextFreq)
		else:
			traj.append(currFreq)

		time -= delta * (2*Nt)
		#if time < 1e-8:
		#   print(time,traj[-1])
		currTime -= 1 
	return traj

def simulate_neutral_backwards(p0,s_locus,tOn,timeBins,N):
	traj = [p0]
	currTime = tOn
	maxNe = np.max(N)
	accepted = 0 
	while not accepted:
		while traj[-1] != 1 and traj[-1] != 0:
			Nt = N[np.digitize(currTime,timeBins)-1]
			delta = 1/(2*Nt)
			curr = traj[-1]
			nextFreq = np.random.normal(curr*(1 - delta), np.sqrt(delta) * np.sqrt(curr*(1-curr)) )
			if nextFreq > 1:
				nextFreq = 1
			if nextFreq < 0:
				nextFreq = 0
			traj.append(nextFreq)
			currTime += 1 
		# rejection sampling to account for variable popsize
		accepted = np.random.binomial(1,Nt/maxNe)
	return traj[1:]

def simulate_trajectory(p0,s_locus,tOn,tOff,timeBins,N,c=0.01):
	#Simulate and write n.loci allele-frequency trajectories. None of these should have fixed,
	#so reject and do it again if fixed. Reject if minor allele frequency is < 0.01.

	fixed = True
	while fixed:
		traj_sel_fwd = simulate_selected_forwards(p0,s_locus,tOn,tOff,timeBins,N)
		if traj_sel_fwd[-1] <= 1-c and traj_sel_fwd[-1] >= c:
			fixed = False
	traj_neu_bwd = simulate_neutral_backwards(p0,s_locus,tOn,timeBins,N)
	full_traj = traj_sel_fwd[::-1] + traj_neu_bwd
	return full_traj

def save_mssel_input(root,traj,N=10000):
	f = open('%smssel.traj'%(root),'w')
	f.write('ntraj: 1\nnpop: 1\nn: %d\n'%(len(traj)))
	delta = 1/(4*N)
	for (i,x) in enumerate(traj):
		f.write('%.9f %.6f\n'%(i*delta,traj[i]))
	f.close()
	return

def save_mssel_call_sheet(nder,nanc,root_locus):
	n = nanc+nder
	cmd = 'module load r; %s %d 1 %d %d %s 2000000 -r 1600 4000000 -t 1600 -T -L > %s'%(PATH_TO_SCRIPTS+'msseldir/mssel',n,nanc,nder,'%smssel.traj'%(root_locus),'%smssel.out'%(root_locus))
	f = open(PATH_TO_CMDS,'a')

	cmd1 = 'python '+PATH_TO_SCRIPTS+'parse_mssel.py %smssel.out 4000000 4000000 %d %strue'%(root_locus,nder+nanc,root_locus)
	cmd2 = 'Rscript '+PATH_TO_SCRIPTS+'ms2haps_mod.R %smssel.out %srelate 4000000 %d'%(root_locus,root_locus,nder + nanc)
	f = open(PATH_TO_CMDS,'a')
	f.write(cmd+'; '+cmd1+';' + cmd2 + '\n')
	return

######################
##### set params
######################
parser = argparse.ArgumentParser(description=
			'simulation method for polygenic traits'+
			'with SNPs under negative selection.')
parser.add_argument('S',type=float,help='average value of -2*Ne*s across causal SNPs')
parser.add_argument('h2',type=float,help='trait heritability')
parser.add_argument('M', type=int, help='number of causal loci/SNPs')
parser.add_argument('N', type=int, help='study size (currently assumed constant '+
			'across traits)')
parser.add_argument('pathToTrait',type=str,help='folder where simulations under this particular model will go.')
parser.add_argument('--tau',type=float,help='beta/selection coupling parameter (default 1/2)',default=0.5)
parser.add_argument('--Mmult', type=int, default=1, help='multiplies M to get the number of causal loci to SIMULATE (if you want to simulate a sampling distn on traits, we recommend 10)'+
			'(you can resample M loci [w/o replacement] from this larger sample)')
parser.add_argument('--n',type=int,default=2000, help='number of HAPLOID samples')
parser.add_argument('--Ne',type=int,default=10**4, help='Diploid effective popn size')
parser.add_argument('--coal',type=str,default=None,help='use a Relate .coal file to specify demographic history.')

args = parser.parse_args()

PATH_TO_SCRIPTS = '/global/home/users/ajstern/rhemc/scripts/'
PATH_TO_TRAIT = args.pathToTrait 
try:
	os.mkdir(PATH_TO_TRAIT)
except:
	pass	

PATH_TO_CMDS = PATH_TO_TRAIT.rstrip('/') + '.cmds'

# load genetic correlation and heritability to make genetic covariance matrix
S = args.S
tau = args.tau
h2 = args.h2 
E = 1.0

N=args.N


M=args.M # number of loci (causal for �~I�1 trait)
Mboot=args.Mmult*M # total number of loci to simulate (make this greater than or equal to M)

if args.coal == None:
	timeBins = np.array([0])
	NeTraj = np.array([args.Ne])
else:
	raise NotImplementedError
	timeBins = np.genfromtxt(args.coal,skip_header=1,skip_footer=1,delimiter=' ')[:-2]
	NeTraj = 0.5/np.genfromtxt(args.coal,skip_header=2,delimiter=' ')[2:-1]	

n = args.n
Ne = args.Ne
p = 1/np.arange(1,2*Ne)
p *= 1/np.sum(p)

eps = 1/(2*Ne)
V = 2/np.log(2*Ne)

columns = ['variant','derived_allele','minor_allele','minor_AF','p0','pcurr','ntot','nder','s']
columns += ['beta','betaHat','se','pval']

df = pd.DataFrame(columns=columns)

# Simulating causal loci
nLoci = 0
snpHer = 0

l = 0
#while snpHer < args.Mmult * h2:
while nLoci < M:	
	# Sample allele frequency and beta at start of selection (say, 50 gens ago)
	if S == 0.0:
		f = np.random.choice(np.arange(1,2*Ne)/(2*Ne),p=p)
		beta = simulate_beta_neutral(h2,V,M)	
		s = 0.0
	else:
		scale = 1
		freqs,betas,ss = sample_freq_and_beta(tau,scale,S,h2,M,Ne=Ne,size=1)
		f = freqs[0]
		beta = betas[0]
		s = ss[0]	

	# simulate Z-score
	ncp = np.sqrt(2*N*f*(1-f))*beta
	z = np.random.normal(ncp,1)

	snpHer += ncp**2/N	
	if l%100 == 0:
		print('h2 is %.2f%% simulated...'%(100*snpHer/(h2*args.Mmult)))
		print(f,beta,z,snpHer,s)
	nder = np.random.binomial(n,f)
	l += 1
	nLoci += 1

	# also specify that >1% of chroms must carry the minor allele
	# no point in wasting simulations on alleles with MAF<1%, they are not informative
	if (nder/n == 0.0 ) or (nder/n == 1.0):
		continue
	se = 1/np.sqrt(2*N*f*(1-f))
	betaHat = z*se
	pval = stats.chi2.sf(z**2,df=1)

	if True: 
		if S == 0:	
			trajBwd = simulate_neutral_backwards(f,0,0,timeBins,NeTraj)
		else:
			trajBwd = simulate_selected_backwards(f,ss[0],N=Ne)

		fullTraj = trajBwd

		# create filesystem for storing the simulations
		try:
			os.mkdir(PATH_TO_TRAIT+'ld_%d/'%(nLoci))
		except:
			pass
		
		save_mssel_input(PATH_TO_TRAIT+'ld_%d/'%(nLoci),fullTraj,N=NeTraj[0])

	# add SNP to dataframe of ascertained SNPs 
	#'variant','derived_allele','minor_allele','minor_AF','p0','pcurr','ntot','nder','s'
	dat = [
		'1:2000000:A:T',
		'T',
		['T','A'][int(f>0.5)],
		np.min([f,1-f]),
		f,
		np.nan,
		n,
		nder,
		s,
	] + [beta,betaHat,se,pval]
	df.loc['ld_%d'%(nLoci)] = dat

	# write commands for running mssel, Relate, etc:
	save_mssel_call_sheet(nder,n-nder,PATH_TO_TRAIT+'ld_%d/'%(nLoci))
df.to_csv(PATH_TO_TRAIT + 'metadata.tsv', sep='\t')
