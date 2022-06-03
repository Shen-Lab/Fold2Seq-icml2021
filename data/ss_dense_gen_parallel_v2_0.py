import numpy as np
import pickle
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import multivariate_normal
import os

from joblib import Parallel, delayed
import multiprocessing

amino_acid = ['A', 'R', 'N', 'D',  'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
IUPAC_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29)])

def to_intger_padding(seq, maxlen):
    vec = np.zeros((maxlen), dtype=np.int32)
    for i in range(len(seq)):
        vec[i] = IUPAC_VOCAB[seq[i]]
    
    return vec



def cal_mean(coor1):
	coor=[]
	for i in coor1:
		coor.append(i['CA'])
	return np.mean(coor, axis=0)

def rot_matrix(theta, ux, uy, uz):
	
	sint = np.sin(theta)
	cost = np.cos(theta)

	rot =[ [ cost + ux**2*(1-cost), ux*uy*(1-cost) - uz*sint, ux*uz*(1-cost)+uy*sint ] ,
	       [ uy*ux*(1-cost) + uz*sint,  cost + uy**2*(1-cost), uy*uz*(1-cost) - ux*sint],
	       [ uz*ux*(1-cost) - uy*sint,  uz*uy*(1-cost) + ux*sint, cost + uz**2*(1-cost)]
	        ]
	return np.array(rot)

def feat_gen(domain_dict):

	print ("start translating proteins to make the center of mass to be at the orign of the coordinate system.")
	print ("start rotating proteins to make the first CA atom on the z-axis......")
	print ("total number of proteins: ", len(domain_dict))

	maxd =[]
	for i in domain_dict:
		camean = cal_mean(domain_dict[i]['3d'])

		for j in range(len(domain_dict[i]['3d'])):
			domain_dict[i]['3d'][j]['CA'] -= camean


		first_residue_ca_coor = domain_dict[i]['3d'][0]['CA']
		
		theta = np.arccos(-first_residue_ca_coor[2] / np.sqrt(np.sum(first_residue_ca_coor**2)))
		x1 = -first_residue_ca_coor[1] / np.sqrt(first_residue_ca_coor[0]**2 + first_residue_ca_coor[1]**2)
		y1 = first_residue_ca_coor[0] / np.sqrt(first_residue_ca_coor[0]**2 + first_residue_ca_coor[1]**2)

		rot = rot_matrix(theta, x1, y1, 0.)

		c1 = np.matmul(rot, first_residue_ca_coor.T)
		assert c1[0]**2 + c1[1]**2 < 1E-20

		
		for j in range(len(domain_dict[i]['3d'])):
			domain_dict[i]['3d'][j]['CA'] = np.matmul(rot, domain_dict[i]['3d'][j]['CA'].T).T

	return domain_dict



def featurization(domain_dict, keys, args):

	print ('start generating features......')
	box_size = 40.
	voxel_size= 2.
	std_gassuian = 2.0
	var_gassuian = std_gassuian**2 
	threshold = (std_gassuian*3)**2
	numbox = int(box_size/voxel_size)

	ss_map = {'H': 0, 'G': 0, 'I': 0, 'B': 1, 'E':1, ' ':2, 'T':3, 'S':3}
	ratio_all=[]
 
	count = 0
	progress_check = 0

	def par_for_m(i):
# 	for i in keys:
# 		count+=1 
# 		if count*100/len(keys)>=progress_check:
# 			print(progress_check,"% files processed; next file's index: ", keys.index(i))
# 			progress_check+=10
		if not os.path.isfile(args.out+i+".npy"): 
			new_coor=[]
			features = np.zeros((numbox, numbox, numbox, 4), dtype=float)

			for j in range(len(domain_dict[i]['3d'])):
				new_coor.append(domain_dict[i]['3d'][j]['CA'])

			minc = np.min(new_coor, axis=0)
			maxc = np.max(new_coor, axis=0)

		
		
			ratio = box_size/(maxc-minc)

#		print ('ratio: ', ratio)
			ratio_all.extend(ratio)
			centerx = np.linspace(minc[0]*ratio[0]+voxel_size/2, minc[0]*ratio[0]+box_size-voxel_size/2, numbox)
			centery = np.linspace(minc[1]*ratio[1]+voxel_size/2, minc[1]*ratio[1]+box_size-voxel_size/2, numbox)
			centerz = np.linspace(minc[2]*ratio[2]+voxel_size/2, minc[2]*ratio[2]+box_size-voxel_size/2, numbox)

		
			for j in range(len(domain_dict[i]['3d'])):
				atom_pos = domain_dict[i]['3d'][j]['CA']*ratio
				type_ss = ss_map[domain_dict[i]['3d'][j]['ss']]
				for i1 in range(numbox):
					for i2 in range(numbox):
						for i3 in range(numbox):
							d2square = (atom_pos[0]-centerx[i1])**2+(atom_pos[1]-centerx[i2])**2+(atom_pos[2]-centerx[i3])**2 


							density = np.exp(-d2square/var_gassuian)

							features[i1][i2][i3][type_ss] +=  density

			np.save(args.out+i.replace('/','-'), features)

	num_cores = args.num_workers #multiprocessing.cpu_count()
	print("# of CPU cores found: ", num_cores)
	results = Parallel(n_jobs=num_cores)(delayed(par_for_m)(i) for i in keys)


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Arguments for pretrain_seq.py')
	parser.add_argument('--num_workers', type=int, default=5, help='number of parallel porcesses to spped up data generation')
	parser.add_argument('--domain_data', type=str, default=None, help='File for input domain dictionary')
	parser.add_argument('--out', type=str, default=None, help='Directory to store input features')
	args = parser.parse_args()

	with open(args.domain_data, "rb") as f:
		domain_dict = pickle.load(f)
	domain_dict = feat_gen(domain_dict)
	keys=[]
	for i in domain_dict:
		keys.append(i)
	featurization(domain_dict, keys, args)