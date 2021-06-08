import numpy as np
from os.path import basename
from glob import glob
import pandas as pd
from utils import remvove_doubles, get_distance_matrix, find_minimal_average_distance_betwen_points
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

def calculate_average_pairwise_distance(embedding_dir, train_embedding_dir, result_file):



    emotions = ["col", "joi", "pla", "tri"]


    embedding_files = glob(embedding_dir + "*/*.txt")
    embedding_files.sort()

    distance_matrix_list = []

    # calculate pairwise-distances per emotion
    for num_emotion in range(len(emotions)):
        emotion = emotions[num_emotion]
        print("----------------------------------------------------------------------------")
        print("Emotion (french): " + emotion)

        augmentation_techniques = []
        all_embeddings_emotion = []
        train_embedding_wildcard = train_embedding_dir + "*" + emotion + "*"
        train_embedding_file = glob(train_embedding_wildcard)[0]
        # Embeddings of the source train data is treated separately as all augmentation techniques are compared against these.
        print("Loading embeddings for source train data ...")
        train_embeddings = np.genfromtxt(train_embedding_file)
        # as the prototypical network selects samples at random, the embedding files contain multiple instances of the same embedding, which we want to exlclude here
        train_embeddings = remvove_doubles(train_embeddings)
        all_embeddings_emotion.append(train_embeddings)
        augmentation_techniques.append("source_train")
        # load embeddings for all augmentation techniques and the given emotion
        for embedding_file in embedding_files:
            if not emotion in basename(embedding_file):
                continue
            augmentation_technique = embedding_file.split("/")[-2]
            augmentation_techniques.append(augmentation_technique)
            print("Loading embeddings for " + augmentation_technique + "...")
            embeddings = np.genfromtxt(embedding_file)
            # remove multiple instances of the same embeddings again.
            embeddings = remvove_doubles(embeddings)
            all_embeddings_emotion.append(embeddings)

        # calculate pairwise distance of embeddings of any augmentation technique and the source embeddings
        result_matrix = np.empty((len(augmentation_techniques), 1))

        source_idx = augmentation_techniques.index("source")
        source_embeddings = all_embeddings_emotion[source_idx]
        for i in range(len(augmentation_techniques)):
            embeddings = all_embeddings_emotion[i]
            # calculates a pair-wise distance of all embeddings in 'embeddings' and 'source_embeddings'.
            distance_matrix = get_distance_matrix(embeddings, source_embeddings)
            # Calculate the minimal average pair-wise distance greedily by iteratively removing points with the smallest distance
            # from the matrix and adding their distance to the total distance.
            min_distance = find_minimal_average_distance_betwen_points(distance_matrix)
            # Save min_distance for each augmentation technique.
            result_matrix[i,0] = min_distance
            if i==0:
                print()
        distance_matrix_list.append(result_matrix)

    # pu together results from all emotions to one dataframe which is then saved to a csv file.
    cols = []
    cols.append("Data")
    for i in range(len(emotions)):
        cols.append(emotions[i])
    distance_result_matrix = np.hstack(distance_matrix_list)
    distance_result_matrix = distance_result_matrix
    aug_techs = np.array(augmentation_techniques).reshape((distance_result_matrix.shape[0], 1))
    distance_result_matrix = np.hstack([aug_techs, distance_result_matrix])
    df = pd.DataFrame(distance_result_matrix, columns=cols)
    df.to_csv(result_file, index=False)
    print("Saved results to " + result_file)



def create_heatmap(result_file, image_file, fold):
    print("Creating heatmap...")
    pointwise = np.round(
        pd.read_csv(result_file, sep=',')[['col', 'joi', 'pla', 'tri']], 3)

    labels_proto = pd.read_csv(result_file, sep=',')[['Data']]

    pointwise = pointwise.rename(columns={'col': 'Anger',
                                          'joi': 'Elation',
                                          'pla': 'Pleasure',
                                          'tri': 'Sad'})
    pointwise = pointwise[['Pleasure', 'Anger', 'Elation', 'Sad']]
    pointwise = pointwise.reindex([0, 2, 1, 3, 4, 5])
    labels_proto = labels_proto.reindex([0, 2, 1, 3, 4, 5])

    #print(labels_proto)

    labels_proto = labels_proto.replace('source_train', 'Support\nSource-' + fold)
    labels_proto = labels_proto.replace('noise', 'Noise-F1')
    labels_proto = labels_proto.replace('source', 'Query\nSource-F1')
    labels_proto = labels_proto.replace('specaug', 'SpecAug-F1')
    labels_proto = labels_proto.replace('time', 'Time-\nShift-F1')
    labels_proto = labels_proto.replace('wavegan', 'WaveGAN-F1')
    #print(labels_proto)

    labels = labels_proto['Data']
    sns.heatmap(pointwise, yticklabels=labels, annot=True, cbar=True, cmap="Greys")
    plt.tight_layout()

    plt.savefig(image_file)
    plt.clf()
    print("done")




# In case of fold 1 embeddings are trained on the partition F1, i.e., the 'train partition' ('train8k256')
# The Embeddings to be investigated are the augmented techniques under 'test-train-embeddings'
embedding_dir_fold_1 = "embeddings/train8k256/test-train-embeddings/"
# Reference embeddings are the train source embeddings under 'train-embeddings'
train_embedding_dir_fold_1 = "embeddings/train8k256/train-embeddings/"
result_file_fold_1 = "result_pairwise_distance/point_wise_distance_results_Fold_1.csv"
image_file_fold_1 = "result_pairwise_distance/point_wise_distance_heatmap_Fold_1.pdf"

# In case of the fold 2+3 embeddings are trained on folds F2 and F3, i.e., the 'devel' and 'test' partitions (develtest8k256)
embedding_dir_fold_2_3 = "embeddings/develtest8k256/test-train-embeddings/"
train_embedding_dir_fold_2_3 = "embeddings/develtest8k256/train-embeddings/"
result_file_fold_2_3 = "result_pairwise_distance/point_wise_distance_results_Fold_2_3.csv"
image_file_fold_2_3 = "result_pairwise_distance/point_wise_distance_heatmap_Fold_2_3.pdf"

print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")
print("Fold 1...")
print()
calculate_average_pairwise_distance(embedding_dir_fold_1, train_embedding_dir_fold_1, result_file_fold_1)
create_heatmap(result_file_fold_1, image_file_fold_1, "F1")
print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")
print("Fold 2...")
print()
calculate_average_pairwise_distance(embedding_dir_fold_2_3, train_embedding_dir_fold_2_3, result_file_fold_2_3)
create_heatmap(result_file_fold_2_3, image_file_fold_2_3, "F2+F3")