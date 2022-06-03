##########################################################################
# Load pdbs  with the chain and residue constraints.
##########################################################################

import numpy as np
import os
import sys
import pickle
import determine_ss
import argparse
import json

threetoone={
'ALA':'A',
'ARG':'R',
'ASN':'N',
'ASP':'D',
'CYS':'C',
'GLU':'E',
'GLN':'Q',
'GLY':'G',
'HIS':'H',
'ILE':'I',
'LEU':'L',
'LYS':'K',
'MET':'M',
'PHE':'F',
'PRO':'P',
'SER':'S',
'THR':'T',
'TRP':'W',
'TYR':'Y',
'VAL':'V',
'MSE':'M'
}

def selection(pdb_path, chain, start, end, ss):
    b=0
    jd=0
    jd1=0
    outs=[]
    start = str(start).replace(')','')
    start = str(start).replace('(','')
    seqs=''
    ca_coor=[]
    end = str(end).replace(')','')
    end = str(end).replace('(','')
    line_count = 0
    file_error = 0
    with open(pdb_path+".txt", "r") as f:
        data = f.readlines()
    for lines in data:
        line_count+=1
        if 1:
            if len(lines)<5 or lines[0:4]!='ATOM':
                if b==2:
                    break
                continue
            if lines[21]!=chain:
                if b==2:
                    break
                continue
            resi = lines[22:27].strip(' ')
            if b==2 and resi!=end:
                break
            if resi==start:
                b=1
                jd=1
            if resi==end:
                b=2
                jd1=1
            elif b==2:
                break
            if b==1 or b==2:
                        
                if lines[13:16]=='CA ':
                    resi={'name': lines[17:20]}
                    try:
                        resi['CA'] = [float(lines[30:37]), float(lines[37:45]), float(lines[46:54])]
                    except:
                        print(pdb_path+chain+" "+start+','+end+" "*5+"Line: "+str(line_count))
                        raise ValueError("could not convert string to float")
                    ca_coor.append(resi)
                    seqs+=threetoone[lines[17:20]]

    if jd*jd1!=1:
        open(pdb_path.split('/')[0]+'/'+pdb_path.split('/')[1]+'/inconsistent_PDBs', 'a').write(pdb_path+chain+" "+start+','+end+"\n")
        file_error = 1
        return ca_coor, ss['seq'], file_error
#         raise ValueError("encounter inconsistent pdb structure:"+pdb_path+chain+" "+start+','+end)



    #print(seqs)
    start,end,file_error2,no_alignment = dp(seqs, ss['seq'])
    if file_error2:
        open(pdb_path.split('/')[0]+'/'+pdb_path.split('/')[1]+'/inconsistent_PDBs', 'a').write(pdb_path+chain+"\n")
        file_error = 1
        return ca_coor, ss['seq'], file_error
    
    if no_alignment:    
        open(pdb_path.split('/')[0]+'/'+pdb_path.split('/')[1]+'/NoAlignmentFound_PDBs', 'a').write(pdb_path+chain+"\n")
        file_error = 1
        return ca_coor, ss['seq'], file_error

    j=start
    for i in range(len(ca_coor)):
        while ss['seq'][j]!=threetoone[ca_coor[i]['name']]:
            j+=1
#        ca_coor[i]['ss'] = ss['ss']
        ca_coor[i]['ss'] = ss['ss'][j]

    return ca_coor, ss['seq'][start:end], file_error


def dp(cst, cseq):

    k=0
    best_start=0
    best_end=0
    best_score=10000
    file_error2 = 0
    no_alignment = 0
    while k<len(cseq)-len(cst)+1:
        try:    
            if cseq[k] == cst[0]:
                i=0
                j=k

                while i<len(cst) and j<len(cseq):
                    if cst[i]==cseq[j]:
                        i+=1
                    j+=1

                if i==len(cst):
                    if j-k< best_score:
                        best_score=j-k
                        best_start=k
                        best_end=j
                        if best_score == len(cst):
                             break
            k+=1
        except:
            file_error2 = 1
            return best_start, best_end, file_error2

    if best_score==10000:
#        print(cst)
#        print(cseq)
#        raise ValueError("do not find alignment.")
        no_alignment = 1

#    print(best_score - len(cst))
#    print("# of mismatches in the best alignment: ", best_score - len(cst))
    
    return best_start, best_end, file_error2, no_alignment


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments for domain dictionary')
    parser.add_argument('--domain_list', default=None, type=str)
    parser.add_argument('--out', default=None, type=str)
    parser.add_argument('--ss', default='./ss.txt', type=str)
    args = parser.parse_args()
    
    P_dir = args.domain_list.split('/')[1].split('.')[0]+'_PDB/preprocessed/'

    if os.path.isfile(args.out):
        with open(args.out, "rb") as f:
            domain_seq = pickle.load(f)
            print("# of PDBs in the existing domain dictionary: ", len(domain_seq.keys()))
    else:
        domain_seq={}
#     domain_seq = {}
   
    seq_ss = determine_ss.read_ss(args.ss)
    num=0
    file_error_count = 0
    keys = []
    progress_check = 0
    line_count = 0

    error_PDBs = []

    with open(args.domain_list, "r") as f:
        PDB_ids = f.readlines()
    for lines in PDB_ids:
        line_count+=1
        line = lines.split('\n')[0].strip(' ')
        label = line[:4].upper()+':'+line[4]
	
        if line in domain_seq:
            continue
        elif not os.path.isfile(P_dir+line+".txt"):
            continue          
	
        if label in seq_ss:
            if 1:
                x1,x2,file_error = selection(P_dir+line ,  line[4], line[8:].split('-')[0],\
                                             line[8:].split('-')[1], seq_ss[label])
            else:
                error_PDBs.append(line+" "*10+"Line: "+str(line_count-1))
            
        if file_error==0:
            domain_seq[line]={}
            domain_seq[line]['seq'] =x2
            domain_seq[line]['3d'] = x1
        
        if line_count*100/len(PDB_ids)>=progress_check:
            print(progress_check,"% files processed")
            progress_check+=10
            
        
    with open(args.out, "wb") as f:
        pickle.dump(domain_seq, f)

    print("# of PDBs found: ", len(domain_seq.keys()), " out of ",len(PDB_ids), " PDBs in the domain list")
    with open(args.domain_list.split('/')[1].split('.')[0]+"_PDBerrors.jsonl", "w") as f:
        f.write(json.dumps({"PDB_ids":error_PDBs}))



