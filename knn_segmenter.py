import torch
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
import math
from scipy.signal import find_peaks
import eval_segmentation
import data_loader
import faiss

parser = argparse.ArgumentParser(description='kNN word segmentation')
parser.add_argument('--win', type=int, default=10, help='window length')
parser.add_argument('--train_n', type=int, default=200, help='number of files from training data to use')
parser.add_argument('--eval_n', type=int, default=200, help='number of files from evaluation data to use')
parser.add_argument('--layer', type=int, default=-1, help='layer index (output)')
parser.add_argument('--offset', type=int, default=4, help='offset to window center')
parser.add_argument('--knn', type=int, default=-10, help='number of nns')
parser.add_argument('--avg', action='store_true', help='concatenate or average the embeddings')
parser.add_argument('--arc', type=str, default='BASE', help='model architecture options: BASE, LARGE, LARGE_LV60K, XLSR53, HUBERT_BASE, HUBERT_LARGE, HUBERT_XLARGE')


parser.add_argument('--width', type=int, default=1, help='width param for find_peaks')
parser.add_argument('--distance', type=int, default=5, help='distance param for find_peaks')
parser.add_argument('--prominence', type=int, default=1, help='prominence param for find_peaks')

parser.add_argument('--train_dir', type=str, default='train/', help='dir of .wav, .wrd files for training data')
parser.add_argument('--val_dir', type=str, default='val/', help='dir of .wav, .wrd files for val data')



args = parser.parse_args()
print(args)


frames_per_embedding = 160

is_cuda = True
#init seeds from SO
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# np.random.seed(1)
# random.seed(0)
# Model init
model, dim = data_loader.get_model(args.arc)
model = model.cuda()


train_wavs, train_bounds = data_loader.get_data_buckeye(args.train_dir, args.train_n)
val_wavs, val_bounds = data_loader.get_data_buckeye(args.val_dir, args.eval_n)


train_e = data_loader.get_emb(train_wavs, model, args.layer)
val_e = data_loader.get_emb(val_wavs, model, args.layer)
train_e = data_loader.make_gallery(train_e, args.win, args.avg)


index = faiss.IndexFlatL2(train_e.shape[1])
index.add(train_e.astype('float32'))

print("frame duration (s): %f" % (frames_per_embedding/16000))

ref_bounds = []
seg_bounds = []
for idx in range(len(val_e)):
	e = data_loader.make_gallery(val_e[idx:idx+1], args.win, args.avg)
	D, I = index.search(e.astype('float32'), args.knn)
	d = D.sum(1)
	
	p, properties = find_peaks(d, width=args.width, distance=args.distance, prominence=args.prominence)

	p = p + args.win // 2
	p = p * 2 + args.offset 
	p = np.minimum(p, 2*(len(e)-1))
	p = p.astype('int')


	boundaries = np.array(data_loader.get_bounds(val_bounds[idx])) // frames_per_embedding
	boundaries = np.minimum(boundaries[1:-1], (len(e)-1)*2)

	ref_bound = np.zeros(len(e)*2)
	seg_bound = np.zeros(len(e)*2)
	ref_bound[boundaries] = 1
	seg_bound[p] = 1
	ref_bound[-1] = 1
	seg_bound[-1] = 1
	ref_bounds.append(ref_bound)
	seg_bounds.append(seg_bound)


	# if idx % 10 == 0:
	# 	precision, recall, f = eval_segmentation.score_boundaries(ref_bounds, seg_bounds, 2)
	# 	print(idx, precision, recall, f)

precision, recall, f = eval_segmentation.score_boundaries(ref_bounds, seg_bounds, 2)
os = eval_segmentation.get_os(precision, recall)*100
r_val = eval_segmentation.get_rvalue(precision, recall)*100
print("Final result:", precision*100, recall*100, f*100, os, r_val)
