

def select_dataset(dataset_opt):
    
    if dataset_opt['name'] == 'vimeo':
        from data.dataset_video_train import VideoRecurrentTrainVimeoDataset as D
    
    elif dataset_opt['name'] == 'realvsr':
        from data.dataset_video_train import VideoRecurrentTrainRealVSRDataset as D
    
    else:
        raise NotImplementedError("Dataset name [%s] is not recognized." % dataset_opt['name'])
    
    return D(dataset_opt)