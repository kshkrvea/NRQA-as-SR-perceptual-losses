

def select_dataset(dataset_opt):
    
    if dataset_opt['name'] == 'vimeo':
        from data.dataset_video_train import VideoRecurrentTrainVimeoDataset as D
    
    elif dataset_opt['name'] == 'realvsr':
        from data.dataset_video_train import VideoRecurrentTrainRealVSRDataset as D
    
    elif dataset_opt['name'] == 'test_realvsr':
        from data.dataset_video_test import VideoTrain_FR_Dataset as D

    elif dataset_opt['name'] == 'test_realhb':
        from data.dataset_video_test import VideoTrain_NR_Dataset as D

    else:
        raise NotImplementedError("Dataset name [%s] is not recognized." % dataset_opt['name'])
    
    return D(dataset_opt)