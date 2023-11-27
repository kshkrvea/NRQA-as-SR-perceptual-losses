

def select_dataset(dataset_opt):
    
    if dataset_opt['name'] == 'vimeo':
        from data.dataset_video_train import VideoRecurrentTrainVimeoDataset as D
    
    elif dataset_opt['name'] == 'realvsr':
        from data.dataset_video_train import VideoRecurrentTrainRealVSRDataset as D
    
    elif dataset_opt['name'] == 'fr_dataset':
        from data.dataset_video_test import Video_FR_Dataset as D

    elif dataset_opt['name'] == 'nr_dataset':
        from data.dataset_video_test import VideoTest_NR_Dataset as D

    else:
        raise NotImplementedError("Dataset name [%s] is not recognized." % dataset_opt['name'])
    
    return D(dataset_opt)