import torch

# Regression Relevance Propogation
class RRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    # generate_RRP for batches
    def generate_RRP(self, batch_size, input, interpreted_series):
        inputs = torch.split(input, batch_size) # split inputs into batches
        relAs, relKs = [], []
        for data in inputs:
            relA, relK = self._generate_RRP(data, interpreted_series) # generate RRP for each batch
            relAs.append(relA)
            relKs.append(relK)
        relA = torch.stack(relAs).mean(0) # mean for batch
        relK = torch.stack(relKs).mean(0) # mean for batch
        return relA, relK # return causal scores for attetnion matrix and convolution kernel

    def _generate_RRP(self, input, interpreted_series):
        """
        This method generates the Causal Scores.

        Args:
            input (torch.Tensor): Input data tensor [total_batch, input_window, series_num, feature_dim].
            interpreted_series (int): Index of the interpreted time series.

        Returns:
            relevance_scores (torch.Tensor): Relevance scores for causal graph construction.
        """
        # Forward pass through the model
        output = self.model(input)
        # Create a one-hot tensor for the interpreted series
        one_hot = torch.zeros_like(output, dtype=torch.float).to(output.device)
        one_hot[:,:,interpreted_series,:] = 1
        # Clone the one-hot tensor and set requires_grad to True for gradient computation
        one_hot_vector = one_hot.clone()
        one_hot.requires_grad_(True)
        # Compute the dot product of one-hot tensor and model output
        one_hot = torch.sum(one_hot * output)
        # Reset gradients and perform backward pass
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # Apply regression relevance propagation to calculate relevance scores
        self.model.relprop(one_hot_vector)
        relAs=[]
        relKs=[]
        # collect causal scores from each encoder layers (in practice, there is only one encoder layer) 
        for layer in self.model.encoder.layers:
            # gradient modulation
            relA = layer.attention.attention.get_rel() * torch.abs(layer.attention.attention.get_grad())
            relK = layer.attention.Wv.get_rel() * torch.abs(layer.attention.Wv.get_grad())

            # w/o interpretation
            # relA = layer.attention.attention.get_wgt()
            # relK = layer.attention.Wv.get_wgt()

            relA = relA.clamp(min=0)        # only the positive causal scores are taken into consideration
            relK = relK.clamp(min=0)        # only the positive causal scores are taken into consideration
            relAs.append(relA.mean((0,1)))  # mean for sample and head
            relKs.append(relK.mean(0))      # mean for head
        relA = torch.stack(relAs).prod(0)   # multiply each sub-tensor along the `encoder layer` dimension
        relK = torch.stack(relKs).prod(0)   # multiply each sub-tensor along the `encoder layer` dimension
        return relA, relK