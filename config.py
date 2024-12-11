input_channel = 4
data_root = './dataset_2019'
channels = [128, 256, 512, 1024]
config = dict(
    model = dict(
        network = dict(
            type = 'Model',
            branch_number = 1,
            channels = channels,
            input_channel = input_channel,
            num_classes = 4,
            hidden_dim = channels[0],
            layer = [2, 2, 6, 2],
            heads = [3, 6, 12, 24],
            window_size = 5,
            down_scaling_factor = [2, 2, 2, 2]
        ),
        loss_ce = dict(use = True, weight = 0.5),
        loss_dice = dict(use = True, weight = 1)
    ),
    dataset = dict(
        from_txt = ['SegmentationClass2019/train_list_2019.txt', 'SegmentationClass2019/valid_list_2019.txt'],
        format = 'nii',
        num_modals = 'full',
        random_crop = 224,
        random_flip = 0.5,
        data_folder = data_root,
        train_list = data_root + '/dataset/train',
        val_list = data_root + '/dataset/val',
        year = 2019
    ),
    train_dataloader = dict(
        batch_size = 16,
    ),
    val_dataloader = dict(
        batch_size = 1,
    ),
    param_schedule = dict(
        optimizer_type = 'Adam',
        optimizer = dict(lr=2e-4, weight_decay=1e-5, betas=(0.9, 0.99), eps=1e-8),
        epoch = 150,
        scheduler_type = 'StepLR',
        scheduler = dict(step_size = 10, gamma = 0.9)
    ),
    resume = None,
    pretrain = None,
    load_weight = None,
    work_dir = './logs_2019'
)