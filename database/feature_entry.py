


class Database():
    def __init__(self):
        self.database = {}
        self.features, self.rois = [], []
        self.roi_to_index, self.index_to_roi = {}, {}
        for idx, entry in self.database.items():
            self.features.append(entry.feature)
            self.rois.append(entry.roi)
            #self.index_to_roi[idx] = entry.roi
            #self.roi_to_index[entry.roi] = idx


    def add_entry(self, entry):
        idx = len(self.database)
        self.database[idx] = entry
        self.features.append(entry.feature)
        self.rois.append(entry.roi)
        #self.index_to_roi[idx] = entry.roi
        #self.roi_to_index[entry.roi] = idx

    def update_entry_feature(self, id, new_feature):
        self.database[id].feature = new_feature
        self.features[id] = new_feature




class FeatureEntry():
    def __init__(self, feature, class_id=None, dataset_id=None, ground_truth=False, image_id=None, bbox = None, roi= None, annotation_id=None):
        self.feature = feature
        self.class_id = class_id
        self.dataset_id = dataset_id
        self.ground_truth = ground_truth
        self.image_id = image_id
        self.annotation_id = annotation_id
        self.bbox = bbox
        self.roi = roi