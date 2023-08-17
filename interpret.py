import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import argparse
from copy import deepcopy
from parse_config import ConfigParser
from explainer.explainer import RRP
from evaluator.evaluator import evaluate, getextendeddelays, evaluatedelay
from utils import prepare_device
from sklearn.cluster import KMeans

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def load_model(path, args, name='Causality Detecting', run_id=None):
    config_path = path + '/config.json'
    checkpoint_path = path + '/model_best.pth'
    args_dict = {'name': name,
                 'config': config_path,
                 'resume': None,
                 'device': args.device}
    config = ConfigParser.from_args(args=args_dict, run_id=run_id)

    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    config['data_loader']['args']['series_num']=data_loader.series_num
    config['data_loader']['args']['time_step']=data_loader.time_step
    config['data_loader']['args']['output_window']=data_loader.output_window
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, config)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model, config, data_loader
    
def analyze(relA, relK, m, n, time_step):
    """
    This function performs causal analysis using relevance scores and constructs the temporal causal graph.

    Args:
        relA (List[torch.Tensor]): List of relevance scores of attention matrix for each time series.
        relK (List[torch.Tensor]): List of relevance scores of causal convolution kernels for each time series.
        m (int): Number of top clusters of causal scores to consider.
        n (int): Number of total clusters for k-means clustering.
        time_step (int): Number of time steps in the input.

    Returns:
        ans (List[Tuple[int, int, int]]): List of tuples representing causal graph edges (cause, effect, lag).
    """
    estimator = KMeans(n_clusters=n)
    ans = []
    # find causes of series i
    for i,relAi in enumerate(relA):
        if relAi.sum()==0.0: # all the weights to series i are zero
            continue
        data=np.array(relAi)
        estimator.fit(data.reshape(-1,1))
        cluster_labels = estimator.labels_
        cluster_centers = estimator.cluster_centers_
        cluster_centers = cluster_centers.reshape(-1)
        largest_m_clusters = np.argsort(cluster_centers)[-m:]
        for j in range(len(relAi)):
            if cluster_labels[j] in largest_m_clusters:
                relKij = relK[i][j]
                indices = np.argsort(-1 * relKij)
                ans.append((j,i,time_step-1-indices[0]))
    return ans

def eval(logger, gt, allcauses, alldelays, columns):
    FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = evaluate(logger, gt, allcauses, columns)
    # evaluate delay discovery
    extendeddelays, readgt, extendedreadgt = getextendeddelays(gt, columns)
    percentagecorrect = evaluatedelay(extendeddelays, alldelays, TPs, 1)*100
    logger.info(f"Percentage of delays that are correctly discovered: {percentagecorrect}%")

def main(model, config, data_loader, gt, bigdata=False):
    logger = config.get_logger('train')
    logger.info("===================Running===================")
    attribution_generator = RRP(model)
    logger.info("ground_truth:"+ (gt if gt else "None"))
    
    device, device_ids = prepare_device(config['n_gpu'])
    columns = list(data_loader.df_data.columns)
    series_num = data_loader.series_num
    data = [timeslice[0] for timeslice in data_loader.dataset]
    label = [timeslice[1] for timeslice in data_loader.dataset]
    data = torch.tensor(np.array(data), dtype=torch.float).to(device)
    label = torch.tensor(np.array(label), dtype=torch.float).to(device)
    if bigdata:
        data = data.mean(0).unsqueeze(0)
        label = label.mean(0).unsqueeze(0)
    relA=[]
    relK=[]
    # interpret each time series
    for interpreted_series in range(series_num):
        rel_a, rel_k = attribution_generator.generate_RRP(data_loader.batch_size, data, interpreted_series)
        relA.append(rel_a.detach().cpu().numpy()[interpreted_series])
        relk_align = deepcopy(rel_k.detach().cpu().numpy()[:,interpreted_series,-1,:])
        # The relK[i][i][-1] is zero vector due to the time_step th data can not be used to predict the time_step th future itself.
        relk_align[interpreted_series,:] = rel_k.detach().cpu().numpy()[interpreted_series,interpreted_series,-2,:]
        relK.append(relk_align)

    m = config['explainer']['m']
    n = config['explainer']['n']
    assert m<n, "the number of selected top m clusters must be smaller than the total number of n clusters"
    ans = analyze(relA, relK, m, n, config['data_loader']['args']['time_step'])

    # causal realtions (causal graph edge)
    logger.info("===================Results===================")
    for e in ans:
        logger.info(f"{columns[e[0]]} causes {columns[e[1]]} with a delay of {e[2]} time steps.")

    # evaluate
    allcauses={i:[] for i in range(len(columns))}
    alldelays={}
    for causal in ans:
        allcauses[causal[1]].append(causal[0])
        alldelays[(causal[1],causal[0])]=causal[2]
    if gt:
        logger.info("===================Evaluation===================")
        eval(logger, gt, allcauses, alldelays, columns)

'''
Directly run Causality Detector
'''
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='CausalityInterpret')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    args = args.parse_args()
    def render(args):
        return load_model('saved/models/Causality Discovery/0714_134931', args), 'data/fMRI/sim1_gt_processed.csv', False
    (model, config, data_loader), gt, bigdata = render(args)
    main(model, config, data_loader, gt, bigdata)
