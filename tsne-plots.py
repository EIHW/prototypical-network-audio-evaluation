import pandas as pd
import numpy as np 
from os.path import basename, splitext
import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
import seaborn as sns

sns.set_style("whitegrid")

def update(handle, orig):
    handle.update_from(orig)
    handle.set_alpha(1)

figure(figsize=(8, 6), dpi=80)

colorname= ['green','gray','purple','red']
markers = ['x', '+', 'o', '*', 'x', '+']

emotions = ["joi", "col","pla","tri"]
emotion_english =['Elation','Anger','Pleasure','Sad']

colors = ['#695167','#B85B4F', '#9E8F8D', '#697848',  'black']

home = 'embeddings/train8k256/train-embeddings/'
proto_files = glob(home + '*.txt')

augs = ['wavegan','source']
augs_true = ['WaveGAN','Source']

def point_distance(point1, point2, metric="EUCLIDEAN"):
    if metric == "EUCLIDEAN":
        return np.sqrt(np.sum((point1 - point2)**2))
        
def remvove_doubles(points):
    new_points = []
    for point in points:
        min_dist = np.inf
        for i in range(len(new_points)):
            dist = point_distance(point, new_points[i])
            if dist < min_dist:
                min_dist = dist
        if min_dist > 1e-10:
            new_points.append(point)
    new_points = np.array(new_points)
    return new_points
lr = 20
perp = 90
interp = 1
for interp in range(0,10):
	for num_aug in range(len(augs)): 
		print(augs[num_aug])
		aug_type = augs[num_aug] 
		home = 'embeddings/train8k256/test-train-embeddings/' + aug_type +'/'
		embedding_files = glob(home + '*.txt')
		augmentation_techniques = []
		result_dir = "results/"
		prototypes = []
		all_embeddings = []
		all_num_embeddings = []
		for num_emotion in range(len(emotions)):
			emotion = emotions[num_emotion]
			for embedding_file in embedding_files:
				if not emotion in basename(embedding_file):
					continue
				augmentation_technique = "_".join(basename(embedding_file).split("_")[1:3])
				augmentation_techniques.append(augmentation_technique)
				proto = np.genfromtxt(proto_files[num_emotion])
				proto = remvove_doubles(proto)
				embeddings = np.genfromtxt(embedding_file)
				proto = remvove_doubles(embeddings)

				all_embeddings.append(embeddings)
				all_num_embeddings.append(len(embeddings))
				prototype = embeddings.mean(axis=0)[np.newaxis,...]

				prototypes.append(prototype)
				all_embeddings.append(prototype)

		distance_matrix = np.empty((len(prototypes), len(prototypes)))
		for i in range(len(prototypes)):
			for j in range(len(prototypes)):
				distance_matrix[i,j] = np.sqrt(np.sum((prototypes[i] - prototypes[j])**2))
		print("----------------------------------------------------------------------------")
		print(emotion)
		print(augmentation_techniques)
		
		all_embeddings = np.vstack(all_embeddings)

		tsne_embeddings = TSNE(2,learning_rate=lr,verbose=0, perplexity=perp, n_iter=450).fit_transform(all_embeddings)
		all_prototypes = np.vstack(prototypes)
		tsne_prototypes = TSNE(2,learning_rate=lr,verbose=0, perplexity=perp, n_iter=450).fit_transform(all_prototypes)


		current_count = 0
		for num_emotion in range(len(emotions)):
			emotion = emotions[num_emotion]
			print(emotion_english[num_emotion])
			
			for step in range(int(len(all_num_embeddings)/len(emotions))):
				if step == 0:
					pass
				print("Step: {},".format(step) + emotion + ", " + augmentation_techniques[step])
				num_embeddings = all_num_embeddings[step + num_emotion*int(len(all_num_embeddings)/len(emotions))]
				embeddings = tsne_embeddings[current_count:current_count + num_embeddings]
				prototype = tsne_embeddings[current_count + num_embeddings]
				if aug_type == 'source': 
					plt.scatter(prototype[0], prototype[1], marker='o', s=500, color=colors[num_emotion], edgecolors='w', alpha=1)
				else:
					plt.scatter(prototype[0], prototype[1], marker=markers[num_aug], s=100, color=colors[num_emotion], edgecolors='w', alpha=0.1)
				# ~ if num_emotion == 0:
					# ~ plt.scatter(embeddings[:, 0], embeddings[:, 1],marker=markers[num_aug], s=80, color=colors[num_emotion], alpha=0.6)
				# ~ else:
				plt.scatter(embeddings[:, 0], embeddings[:, 1], marker=markers[num_aug],s=40, color=colors[num_emotion], alpha=0.6,label = emotion_english[num_emotion] + ' ' + augs_true[num_aug])

			
				current_count += num_embeddings + 1
		
		
		plt.legend(handler_map={PathCollection : HandlerPathCollection(update_func= update),
                        plt.Line2D : HandlerLine2D(update_func = update)})
	plt.xlim([-30, 30])
	plt.ylim([-30, 30])
	plt.savefig(f'{result_dir}{interp}_{augs[num_aug]}_{emotion_english[0]}lr-{lr}-p-{perp}.pdf')
	plt.clf()
