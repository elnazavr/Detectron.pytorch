import faiss
import numpy as np


OUTPUT_PATH = "data/outputs/coco/valonly/faster-rcnn/r-50-fpn-2x/"


def find_threhold_for_each_class(index, db, classes,  k_neighbours=10):
    print("Doing search")
    distance, indecies = index.search(db, k_neighbours)
    print("Finishing search")
    classes_idx = classes[indecies]
    average_distance_sample = []
    counts = {}
    for idx, neighbours in enumerate(classes_idx):
        myself = int(classes[idx])
        not_class_neighbours = np.where(neighbours!=myself)[0]
        if len(not_class_neighbours) == 0:
            not_class_neighbours = [k_neighbours-1]
        first_not_class_neighbours = not_class_neighbours[0]
        if first_not_class_neighbours==0:
            import ipdb; ipdb.set_trace()
        if myself not in counts.keys():
            counts[myself] = []
        counts[myself].append(first_not_class_neighbours)
        average_distance_sample.append(distance[idx, first_not_class_neighbours])
    average_distance_sample = np.array(average_distance_sample)
    average_distance_class = {}
    for class_idx in set(classes):
        average_distance_class[class_idx] = np.median(average_distance_sample[np.where(classes==class_idx)])
    return average_distance_class, counts



def create_db_exact_NN(db):
    dimension = 1024
    db = db.astype('float32')
    faiss_db = faiss.IndexFlatL2(dimension)
    faiss_db.add(db)
    return faiss_db