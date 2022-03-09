from imports import (os, np, tqdm, transforms, plt, binned_statistic)
from utils import (set_seed, ToTensor, depthdenorm, fetch_params, normalize,
                   set_prefix, get_params, prepare_dataset, prepare_training,
                   Resize, Metric)

_SUPPORTED_DATASETS = ['balser_v1', 'nyu_v2', 'diode', 'balser_v2']
_SUPPORTED_MODELS = ['dgunet']
_TRAIN_TRANSFORM = transforms.Compose([
            Resize((480, 640)),
            ToTensor()])
_LOAD_CHECKPOINT = True
_LOAD_OPTIMIZER = False
_SEED_VALUE = 123456789
_DEVICE = set_seed(_SEED_VALUE)
_CONFIG_FILE = os.path.join(os.getcwd(), 'config.json')
parameters = fetch_params(get_params(_CONFIG_FILE), supported_dataset=_SUPPORTED_DATASETS,
                          supported_models=_SUPPORTED_MODELS)
if parameters['dataset_name'] == 'balser_v1':
    _MAX_DEPTH = 1300
elif parameters['dataset_name'] == 'nyu_v2':
    _MAX_DEPTH = 10
elif parameters['dataset_name'] == 'diode':
    _MAX_DEPTh = 10
else:
    _MAX_DEPTH = 1500

_PREFIX = set_prefix(parameters)
_IMAGE_DIR = os.path.join(parameters['output_root'], 'Images_sample')
_CHARTS_DIR = os.path.join(parameters['output_root'], 'Charts')
#
if not os.path.isdir(_IMAGE_DIR):
    os.mkdir(_IMAGE_DIR)
if not os.path.isdir(_CHARTS_DIR):
    os.mkdir(_CHARTS_DIR)
#
data_test = prepare_dataset(parameters, False, _TRAIN_TRANSFORM)
model = prepare_training(parameters, _PREFIX, 0, _DEVICE, load_checkpoint=_LOAD_CHECKPOINT,
                         load_optimizer=_LOAD_OPTIMIZER, training=False)

if __name__ == '__main__':
    eval_metrics = Metric(max_depth=_MAX_DEPTH)
    preds = np.array([])
    targets = np.array([])
    n_bins = 1000
    delta = _MAX_DEPTH / n_bins
    bins = np.arange(0, _MAX_DEPTH, delta)
    centers = np.arange((delta / 2), (_MAX_DEPTH - delta / 2), delta)
    vrange = tqdm(data_test)
    for i, sample_batched in enumerate(vrange):
        intensity_file = os.path.join(_IMAGE_DIR, 'intensity_{}.png'.format(i))
        depth_file = os.path.join(_IMAGE_DIR, 'depth_{}.png'.format(i))
        prediction_file = os.path.join(_IMAGE_DIR, 'prediction_{}.png'.format(i))
        #
        image = sample_batched['image']
        depth = sample_batched['depth']
        # print("Shape of input image : {}".format(image.shape))
        # print("Shape of target depth : {}".format(depth.shape))
        mask = sample_batched['mask']

        prediction = depthdenorm(model(image.to(_DEVICE)), _MAX_DEPTH)
        #
        image = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
        depth = depth.detach().cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0]
        mask = mask.detach().cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0]
        #
        prediction = prediction.detach().cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0]
        prediction *= mask
        depth = depth * mask
        # print("Shape of prediction : {}".format(prediction.shape))
        # print("Shape of ground truth : {}".format(depth.shape))
        eval_metrics.update(depth, prediction)
        targets = np.append(targets, (depth[depth > 0]).flatten()).flatten()
        preds = np.append(preds, (prediction[depth > 0]).flatten()).flatten()

        if i % 10 == 0:
            fig = plt.figure(1)
            plt.imshow(image)
            plt.savefig(intensity_file)
            #
            fig2 = plt.figure(2)
            plt.imshow(normalize(depth), cmap='gray')
            plt.savefig(depth_file)
            #
            fig3 = plt.figure(3)
            plt.imshow(normalize(prediction), cmap='gray')
            plt.savefig(prediction_file)
    eval_metrics.display_avg()

    # # # Prediction against targets. (Nuage de points.)
    plt.figure(num=11, figsize=(10,10))
    plt.scatter(targets, preds, color='b', label='Pred(Targets)', alpha=0.5)
    plt.xlabel("Ground truth")
    plt.ylabel("Predictions")
    plt.show()

    # # # Binning
    bin_means, bin_edges, binnumber = binned_statistic(targets, [preds, targets],
                                                       'mean', bins=bins)
    bin_std, _, _ = binned_statistic(targets, [preds, targets], 'std', bins=bins)
    bin_min, _, _ = binned_statistic(targets, [preds, targets], 'min', bins=bins)
    bin_max, _, _ = binned_statistic(targets, [preds, targets], 'max', bins=bins)

    print("Bin means shape:{}\nBin edges shape: {}\nBin number shape: {}"
          .format(bin_means.shape, bin_edges.shape, binnumber.shape))
    print("Binned mean predictions {}\nBinned mean ground thruth {}".format(bin_means[0],
                                                                            bin_means[1]))
    print("Bins", bins)
    print("Centers", centers)

    # # Difference between mean value and target value.
    plt.figure(num=4, figsize=(10,10))
    plt.scatter(centers, np.abs(np.subtract(bin_means[0], centers)), marker='^')
    plt.plot(centers, np.abs(np.subtract(bin_means[0], centers)), color='b',
             label='Predictions-Centers')
    #
    plt.scatter(centers, np.abs(np.subtract(bin_means[1], centers)), marker='o')
    plt.plot(centers, np.abs(np.subtract(bin_means[1], centers)), color='r',
             label='Targets-Centers')
    plt.legend()
    plt.xlabel('Center values (mm)')
    plt.ylabel('Differences (mm)')
    plt.grid(True)
    plt.show()
    plt.gcf().savefig(os.path.join(_CHARTS_DIR,'Diff2Center.png'))
    plt.clf()

    # # Mean 2 Mean difference
    plt.figure(num=5, figsize=(10, 10))
    plt.scatter(centers, np.abs(np.subtract(bin_means[0], bin_means[1])), marker='^')
    plt.plot(centers, np.abs(np.subtract(bin_means[0], bin_means[1])), color='b',
             label='Targets-Predictions')
    plt.xlabel('Center values(mm)')
    plt.ylabel('Differences (mm)')
    plt.grid(True)
    plt.show()
    plt.gcf().savefig(os.path.join(_CHARTS_DIR, 'Mean2Mean.png'))
    plt.clf()

    # # Prediction function of target
    plt.figure(num=6, figsize=(10, 10))
    plt.scatter(centers, bin_means[0], marker='o', color='b')
    plt.scatter(centers, bin_means[1], marker='o', color='r')

    plt.vlines(centers, bin_means[0]-bin_std[0], bin_means[0]-bin_std[0], color='b',
               label='Std Predictions')
    plt.vlines(centers, bin_means[1]-bin_std[1], bin_means[1]-bin_std[1], color='r',
               label='Std Targets')

    plt.scatter(centers, bin_max[0], marker='1', color='b', label='Max Predictions')
    plt.scatter(centers, bin_max[1], marker='1', color='r', label='Max Targets')

    # plt.scatter(centers, bin_min[0], marker='2', color='b', label='Min Predictions')
    # plt.scatter(centers, bin_min[1], marker='2', color='r', label='Min Targets')

    plt.plot(centers, bin_means[0], color='b', label='Pred(Tar')
    plt.plot(centers, bin_means[1], color='r', label='Ground Thruth')

    plt.xlabel('Target mean values (mm')
    plt.ylabel('Prediction mean values (mm)')

    plt.grid(True)
    plt.show()

    plt.gcf().savefig(os.path.join(_CHARTS_DIR, 'Pred2Tar.png'))
    plt.clf()