import numpy as np
import argparse
'''
generates three different files: 
1) a .vcf file TODO 
2) a .trees file for the site of interest 
'''

def convert(filename,length,numIndividuals,outfilename,focalPosn):
	
        SUPPRESS_TREES = False 
	
        gmax = 2*10**5
        n = 0
        num_iter = 0
        f = open(filename,'r')
        treeFlag = False
        for line in f: 
                if line[0] == '[' and not treeFlag:
                        print('hi')
                        prev_p = -999
                        treeDict = {} 
                        treeLine = line
                        prev_posn = 0
                        if SUPPRESS_TREES:
                                continue
                        while treeLine[:4] != 'time':
                                posn = int(treeLine.split('[')[1].split(']')[0]) + prev_posn
                                prev_posn = posn
                                if posn >= focalPosn:
                                    print(posn, focalPosn)
                                    outTrees = open(outfilename+'.tree','w')
                                    outTrees.write(treeLine.rstrip().split(']')[1])
                                    outTrees.close()
                                    treeFlag = True
                                    break
				
                                treeLine = f.readline() 
                elif line[:8] == 'segsites':
                        #continue
                        #TODO

                        out = open(outfilename+'.sites','w')
                        out.write('NAMES\t'+'\t'.join([str(i) for i in range(numIndividuals)])+'\n')
                        out.write('REGION\tchr\t1\t'+str(length)+'\n')

                        s = int(line.rstrip().split(' ')[1])
                        n = numIndividuals
                        genotypes = np.zeros((n,s))
                        continue
                if n != 0:
                        #continue
                        #TODO

                        if line[0] == 'p':
                                positions = [int(float(length) * float(p)) for p in line.rstrip().split(' ')[1:]]
                                #print(positions)
                                if SUPPRESS_TREES:
                                        continue
                                
                                for p in positions:
                                	for key in treeDict.keys():
                                		a = key[0]
                                		b = key[1]
                                		if a <= p and p < b:
                                			#outTrees = open(outfilename+'.i_'+str(num_iter)+'.posn_'+str(p)+'.trees','w') 
                                			#outTrees.write(treeDict[key])
                                			continue
                        else:
                                try:
                                        genotypes[numIndividuals-n,:] = [0 if x == '0' else 1 for x in line.rstrip()]
                                        #print(genotypes[numIndividuals-n,:])
                                except:
                                        #print(s,len(line),line)
                                        raise ValueError
                                n -= 1
                                if n == 0:
                                        #print(genotypes[:,0])
                                        for (j,p) in enumerate(positions):
                                                if np.sum(genotypes[:,j]) > 0 and np.sum(genotypes[:,j]) < numIndividuals and prev_p != p:
                                                        #print('hi')
                                                        out.write(str(p)+'\t'+''.join(['T' if x == 1 else 'A' for x in genotypes[:,j]])+'\n')
                                                prev_p = p
                                        num_iter += 1
                                        out.close()
        return
parser = argparse.ArgumentParser()
parser.add_argument('filename',type=str)
parser.add_argument('length',type=int)
parser.add_argument('nsites',type=int)
parser.add_argument('numIndividuals',type=int)
parser.add_argument('outfilename',type=str)
parser.add_argument('--posn',type=int,default=2000000)
args = parser.parse_args()

filename = args.filename
length = args.length
nsites = args.nsites
numIndividuals = args.numIndividuals
outfilename = args.outfilename
posn = args.posn

convert(filename,length,numIndividuals,outfilename,posn)
