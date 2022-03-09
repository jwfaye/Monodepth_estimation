from imports import (virtual_memory, os, json, torch, nn, F, exp, resize, warp, random_split,
                     crop, vutils, image_histogram2d, DataLoader, StepLR, SummaryWriter,
                     AffineTransform, SamplesLoss)
from datasets import *
from models import DGModel


def set_prefix(dico):
    prefix = '{}_{}_{}_{}'.format(dico['model_name'], dico['dataset_name'],
                                  dico['batch_size'], dico['loss_type'])
    return prefix


def get_params(config_file):
    with open(config_file) as f:
        args = json.load(f)
    return args


def prepare_dataset(parameters, training, transform):
    if (parameters['dataset_name'] == 'balser_v1'
            or parameters['dataset_name'] == 'balser_v2'):
        dataset = BALSER(data_root=parameters['data_root'],
                         training=training, transform=transform)
    elif parameters['dataset_name'] == 'nyu_v2':
        dataset = NYUV2(data_root=parameters['data_root'], training=training, transform=transform)
    else:
        dataset = DIODE(data_root=parameters['data_root'], training=training, transform=transform)
    if training:
        t_length = int(0.95 * len(dataset))
        v_length = len(dataset) - t_length
        # Assign train/val datasets for use in dataloaders
        dataset_train, dataset_val = random_split(dataset, [t_length, v_length])
        print("Length of balser train : {}, length val")
        dataset_train = DataLoader(dataset_train,
                                   batch_size=parameters['batch_size'],
                                   num_workers=parameters['num_workers'],
                                   pin_memory=True,
                                   drop_last=True,
                                   worker_init_fn=worker_init_fn)

        dataset_val = DataLoader(dataset_val,
                                 batch_size=1,
                                 num_workers=parameters['num_workers'],
                                 pin_memory=True,
                                 drop_last=True,
                                 worker_init_fn=worker_init_fn)

        print("Dataloader :\nLength of train sample {},\nLength of val sample {}"
              .format(len(dataset_train), len(dataset_val)))
        ###
        example_sample = next(iter(dataset_val))
        print("Shape of first element :{}/{}".format(example_sample['image'].shape,
                                                     example_sample['depth'].shape))
        return dataset_train, dataset_val, example_sample
    else:
        dataset_test = DataLoader(dataset, batch_size=1, num_workers=parameters['num_workers'],
                                  pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
        print("Dataloader :\nLength of test sample {}".format(len(dataset_test)))
        return dataset_test


def prepare_training(parameters, prefix, start_epoch, device,
                     load_checkpoint=True, load_optimizer=True, training=True):
    if parameters['model_name'] == 'dgunet':
        model = DGModel().to(device)
    else:
        print("Not supported yet XoXo")
    model_ckpt_file = os.path.join(parameters['output_root'],
                                   '{}_checkpoint.pth.tar'.format(prefix))
    optimizer = torch.optim.AdamW(model.parameters(), parameters['lr'])

    if training:
        writer = SummaryWriter(logdir=osp.join(parameters['data_root'], 'logs'),
                               comment='{}-lr{}-e{}-bs{}'.format(prefix, parameters['lr'],
                                                                 parameters['epochs'],
                                                                 parameters['batch_size']),
                               flush_secs=30)
        if parameters['loss_type'] == 'l1':
            loss_func = nn.SmoothL1Loss(20, reduction='sum')
        elif parameters['loss_type'] == 'l2':
            loss_func = nn.MSELoss(reduction='sum')
        else:
            loss_func = SamplesLoss(loss='sinkhorn', p=2, blur=.05, scaling=0.5, diameter=1)
        if load_checkpoint:
            model, optimizer, scheduler = load_model(model_ckpt_file, model, optimizer, load_optimizer, device)

        return start_epoch, writer, loss_func, optimizer, scheduler, model_ckpt_file, model
    else:
        model, _, _ = load_model(model_ckpt_file, model, optimizer, load_optimizer, device)
        return model


def load_model(model_ckpt_file, model, optimizer, load_optimizer, device):
    if osp.isfile(model_ckpt_file):
        print("=> loading checkpoint '{}'".format(model_ckpt_file))
        checkpoint = torch.load(model_ckpt_file, map_location=torch.device(device))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}, optimizer loaded :{})"
              .format(model_ckpt_file, checkpoint['epoch'], load_optimizer))
    scheduler = StepLR(optimizer, step_size=250, gamma=0.1)
    return model, optimizer, scheduler


def fetch_params(config_dict, supported_dataset, supported_models):
    # Batch size
    if not config_dict['batch_size']:
        batch_size = 4
    else:
        batch_size = config_dict['batch_size']
    assert type(batch_size) is int
    # Num workers
    if not config_dict['num_workers']:
        num_workers = 2
    else:
        num_workers = config_dict['num_workers']
    assert type(num_workers) is int
    # Loss type
    l_type = config_dict['loss_type']
    assert l_type is not None
    assert type(l_type) is str
    # Epochs
    n_epochs = config_dict['epochs']
    assert n_epochs is not None
    assert type(n_epochs) is int
    # Lr
    lr = config_dict['lr']
    assert type(lr) is float
    # dataset_name
    dataset_name = config_dict['dataset_name']
    assert dataset_name in supported_dataset
    # model_name
    model_name = config_dict['model_name']
    assert model_name in supported_models
    # data_root
    if not config_dict['data_root']:
        data_root = os.path.join(os.getcwd(), 'datasets', dataset_name)
        print("Created data_root", data_root)
    else:
        data_root = config_dict['data_root']
        print("Provided data_root", data_root)
    # Output root
    if not config_dict['output_root']:
        output_root = os.path.join(os.getcwd(), 'outputs',  model_name, dataset_name)
        if not os.path.isdir(output_root):
            os.makedirs(output_root)
            print("Created output directory", output_root)
    else:
        output_root = config_dict['output_root']
        print("Provided output directory", output_root)

    return {"batch_size": batch_size,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "output_root": output_root,
            "data_root": data_root,
            "epochs": n_epochs,
            "lr": lr,
            "loss_type": l_type,
            "num_workers": num_workers,
            }


def available_ram():
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))


def set_seed(seed=123456789):
    # print(seed)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    np.random.seed(seed)
    return device


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def check_and_tuplize_tokens(tokens, valid_tokens):
    if not isinstance(tokens, (tuple, list)):
        tokens = (tokens, )
    for split in tokens:
        assert split in valid_tokens
    return tokens


def normalize(value):
    return (value - value.min())/(value.max() - value.min())


def depthnorm(depth, maxdepth):
    depth /= maxdepth
    return depth


def depthdenorm(depth, maxdepth):
    depth *= maxdepth
    return depth


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    length = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    c1 = (0.01 * length) ** 2
    c2 = (0.03 * length) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret


def logprogress(model, writer, track_sample, epoch, maxdepth):
    model.eval()
    sample_batched = track_sample.copy()
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    if epoch == 1:
        writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 1:
        writer.add_image('Train.2.Depth', normalize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    output = depthdenorm(model(image), maxdepth)
    writer.add_image('Train.3.Ours', normalize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.4.Diff', normalize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)),
                     epoch)
    del image, depth, output, model


def errors(gt, pred, maxdepth):
    valid_mask = gt > 0
    pred_eval, gt_eval = pred[valid_mask], gt[valid_mask]

    pred_eval, gt_eval = depthdenorm(pred_eval, maxdepth), depthdenorm(gt_eval, maxdepth)
    threshold = np.maximum((gt_eval / pred_eval), (pred_eval / gt_eval))
    delta1 = (threshold < 1.25).mean()
    delta2 = (threshold < 1.25 ** 2).mean()
    delta3 = (threshold < 1.25 ** 3).mean()
    abs_diff = np.abs(pred_eval - gt_eval)
    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(np.power(abs_diff, 2)))
    abs_rel = np.mean(abs_diff / gt_eval)
    log_abs_diff = np.abs(np.log10(pred_eval) - np.log10(gt_eval))
    log_mae = np.mean(log_abs_diff)
    log_rmse = np.sqrt(np.mean(np.power(log_abs_diff, 2)))
    return mae, rmse, abs_rel, log_mae, log_rmse, delta1, delta2, delta3


class Metric(object):
    def __init__(self, max_depth):
        self.results = {}
        self.eval_keys = ['mae', 'rmse', 'abs_rel', 'log_mae', 'log_rmse', 'delta1', 'delta2', 'delta3']
        self.max_depth = max_depth
        for item in self.eval_keys:
            self.results[item] = []

    def update(self, gt, pred):
        assert (gt.shape == pred.shape)
        mae, rmse, abs_rel, log_mae, log_rmse, delta1, delta2, delta3 = errors(gt, pred, self.max_depth)
        for item in self.eval_keys:
            self.results[item].append(eval(item))

    def display_avg(self):
        print("Evaluation Complete:")
        for item in self.eval_keys:
            print("{}: {:.4f}".format(item, np.mean(self.results[item])))


class SoftHistogram(nn.Module):
    def __init__(self, bins=2, mini=0.0, maxi=10.0):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.mini = mini
        self.maxi = maxi

    def update_bins(self, factor=2):
        self.bins *= factor

    def forward(self, x):
        # Return density of probability
        _, x = image_histogram2d(x, min=self.mini, max=self.maxi, n_bins=self.bins, return_pdf=True)
        return x


class ToTensor(object):
    """
    The use of a custom ToTensor class is motivated by the fact that
    the Pytorch ToTensor class converts a PIL or a numpy.ndarray in the range [0,255]
    To  a torch.FLoatTensor in the range [0.0, 1.0].

    This class converts the raw values to tensor in respect of the format (CxHxW)
    """
    def __call__(self, sample):
        assert (isinstance(sample, dict))
        # numpy image : HxWxC
        # torch imae : CxHxW0
        image, depth = sample['image'].copy(), sample['depth'].copy()
        mask = sample['mask'].copy()
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).type(torch.float32)
        #
        try:
            depth = depth.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)
        except:
            depth = np.expand_dims(depth, axis=2).transpose(2, 0, 1)
            mask = np.expand_dims(mask, axis=2).transpose(2, 0, 1)
        #
        depth = torch.from_numpy(depth).type(torch.float32)
        mask = torch.from_numpy(mask).type(torch.float32)
        return {'image': image, 'depth': depth, 'mask': mask}


class RandomHorizontalFlip(object):
    """Flip horizontally the image and the depth"""
    def __call__(self, sample):
        image, depth, mask = sample['image'].copy(), sample['depth'].copy(), sample['mask'].copy()
        #
        if np.random.random() > 0.5:
            image, depth = np.fliplr(image).copy(), np.fliplr(depth).copy()
            mask = np.fliplr(mask).copy()
        return {'image': image, 'depth': depth, 'mask': mask}


class Resize(object):
    def __init__(self, size):
        self.height = size[0]
        self.width = size[1]

    def __call__(self, sample):
        image, depth, mask = sample['image'].copy(), sample['depth'].copy(), sample['mask'].copy()
        if self.height != image.shape[0]:
            image = resize(image, (self.height, self.width), preserve_range=True)
        depth = resize(depth, (image.shape[0] // 2, image.shape[1] // 2),
                       preserve_range=True)
        # print("Shape de mask {}".format(mask.shape))
        mask = resize(mask, (image.shape[0] // 2, image.shape[1] // 2),
                      preserve_range=True)
        # print("Shape de mask {}".format(mask.shape))
        mask_ = np.zeros_like(mask)
        mask_[mask > 0.5] = 1.0
        return {'image': image, 'depth': depth, 'mask': mask_}


class RandomRotate(object):
    """Rotate randomly"""
    def __init__(self):
        self.angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    def __call__(self, sample):
        image, depth, mask = sample['image'].copy(), sample['depth'].copy(), sample['mask'].copy()
        arg = np.random.randint(0, 4)
        image = warp(image, AffineTransform(rotation=self.angles[arg]), mode='reflect')
        depth = warp(depth, AffineTransform(rotation=self.angles[arg]), mode='reflect')
        mask = warp(mask, AffineTransform(rotation=self.angles[arg]), mode='reflect')
        return {'image': image, 'depth': depth, 'mask': mask}


class RandomHorizontalTranslate(object):
    """Translate Horizontally randomly"""
    def __call__(self, sample):
        image, depth, mask = sample['image'].copy(), sample['depth'].copy(), sample['mask'].copy()
        steps = [image.shape[0] // 8, image.shape[0] // 4, image.shape[0] // 2, 0]
        arg = np.random.randint(0, 4)
        image = warp(image, AffineTransform(translation=(steps[arg], 0)), mode='reflect')
        depth = warp(depth, AffineTransform(translation=(steps[arg], 0)), mode='reflect')
        mask = warp(mask, AffineTransform(translation=(steps[arg], 0)), mode='reflect')
        return {'image': image, 'depth': depth, 'mask': mask}


class RandomVerticalTranslate(object):
    """ Translate Vertically randomly"""
    def __call__(self, sample):
        image, depth, mask = sample['image'].copy(), sample['depth'].copy(), sample['mask'].copy()
        steps = [image.shape[1] // 8, image.shape[1] // 4, image.shape[1] // 2, 0]
        arg = np.random.randint(0, 4)
        image = warp(image, AffineTransform(translation=(0, steps[arg])), mode='reflect')
        depth = warp(depth, AffineTransform(translation=(0, steps[arg])), mode='reflect')
        mask = warp(mask, AffineTransform(translation=(0, steps[arg])), mode='reflect')
        return {'image': image, 'depth': depth, 'mask': mask}


class RandomCrop(object):
    """ Crop randomly"""
    def __call__(self, sample):
        image, depth, mask = sample['image'].copy(), sample['depth'].copy(), sample['mask'].copy()
        # print("Image width : {}\nDepth width : {}".format(image.shape, depth.shape))

        image_crop_widths = [((0, image.shape[0]//2), (0, image.shape[1] // 2), (0, 0)),
                             ((0, image.shape[0] // 2), (image.shape[1] // 2, 0), (0, 0)),
                             ((image.shape[0] // 2, 0), (0, image.shape[1] // 2), (0, 0)),
                             ((image.shape[0] // 2, 0), (image.shape[1] // 2, 0), (0, 0)),
                             ((image.shape[0] // 4, image.shape[0] // 4), (image.shape[1] // 4, image.shape[1] // 4),
                              (0, 0)),
                             ((0, 0), (0, 0), (0, 0))]
        depth_crop_widths = [((0, image.shape[0] // 2), (0, image.shape[1] // 2)),
                             ((0, image.shape[0] // 2), (image.shape[1] // 2, 0)),
                             ((image.shape[0] // 2, 0), (0, image.shape[1] // 2)),
                             ((image.shape[0] // 2, 0), (image.shape[1] // 2, 0)),
                             ((image.shape[0] // 4, image.shape[0] // 4), (image.shape[1] // 4, image.shape[1] // 4)),
                             ((0, 0), (0, 0))]
        arg = np.random.randint(0, 5)

        image = crop(image, image_crop_widths[arg], copy=False)
        depth = crop(depth, depth_crop_widths[arg], copy=False)
        mask = crop(mask, depth_crop_widths[arg], copy=False)
        return {'image': image, 'depth': depth, 'mask': mask}


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
