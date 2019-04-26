from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


######################################################################################################################
class imageCaptionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        "imageCaptionModel" is the main module class for the image captioning network
        
        Args:
            config: Dictionary holding neural network configuration

        Returns:
            self.Embedding  : An instance of nn.Embedding, shape[vocabulary_size, embedding_size]
            self.inputLayer : An instance of nn.Linear, shape[VggFc7Size, hidden_state_sizes]
            self.rnn        : An instance of RNN
            self.outputLayer: An instance of nn.Linear, shape[hidden_state_sizes, vocabulary_size]
        """
        self.config = config
        self.vocabulary_size    = config['vocabulary_size']
        self.embedding_size     = config['embedding_size']
        self.VggFc7Size         = config['VggFc7Size']
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers     = config['num_rnn_layers']
        self.cell_type          = config['cellType']

        # ToDo
        self.Embedding = nn.Embedding(self.vocabulary_size, \
                                      self.embedding_size)

        self.inputLayer = nn.Linear(self.VggFc7Size, \
                                    self.hidden_state_sizes)

        self.rnn = RNN(self.embedding_size, \
                       self.hidden_state_sizes, \
                       self.num_rnn_layers, \
                       cell_type=self.cell_type)

        self.outputLayer = nn.Linear(self.hidden_state_sizes, \
                                     self.vocabulary_size)
        return

    def forward(self, vgg_fc7_features, xTokens, is_train, current_hidden_state=None):
        """
        Args:
            vgg_fc7_features    : Features from the VGG16 network, shape[batch_size, VggFc7Size]
            xTokens             : Shape[batch_size, truncated_backprop_length]
            is_train            : "is_train" is a flag used to select whether or not to use estimated token as input
            current_hidden_state: If not None, "current_hidden_state" should be passed into the rnn module
                                  shape[num_rnn_layers, batch_size, hidden_state_sizes]

        Returns:
            logits              : Shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_hidden_state: shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # ToDO
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.
        if current_hidden_state is None:
            initial_hidden_state = \
            torch.tanh(self.inputLayer.forward(vgg_fc7_features)).repeat(self.num_rnn_layers, 1, 1)
        else:
            initial_hidden_state = current_hidden_state

        # use self.rnn to calculate "logits" and "current_hidden_state"
        logits, current_hidden_state_out = self.rnn.forward(xTokens, \
                                                        initial_hidden_state, \
                                                        self.outputLayer, \
                                                        self.Embedding, \
                                                        is_train=is_train)
        
        return logits, current_hidden_state_out

######################################################################################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='RNN'):
        super().__init__()
        """
        Args:
            input_size (Int)        : embedding_size
            hidden_state_size (Int) : Number of features in the rnn cells (will be equal for all rnn layers)
            num_rnn_layers (Int)    : Number of stacked rnns
            cell_type               : Whether to use vanilla or GRU cells
            
        Returns:
            self.cells              : A nn.ModuleList with entities of "RNNCell" or "GRUCell"
        """
        self.input_size        = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers    = num_rnn_layers
        self.cell_type         = cell_type

        # ToDo
        # Your task is to create a list (self.cells) of type "nn.ModuleList" and populated it with cells of type "self.cell_type".
        self.cells = nn.ModuleList([{self.cell_type == "RNN" : RNNCell(self.hidden_state_size, self.input_size),
                                     self.cell_type == "GRU" : GRUCell(self.hidden_state_size, self.input_size)}.get(True)\
                                    if i == 0 else 
                                    {self.cell_type == "RNN" : RNNCell(self.hidden_state_size, self.hidden_state_size),
                                     self.cell_type == "GRU" : GRUCell(self.hidden_state_size, self.hidden_state_size)}.get(True)\
                                    for i in range(self.num_rnn_layers)])
        return


    def forward(self, xTokens, initial_hidden_state, outputLayer, Embedding, is_train=True):
        """
        Args:
            xTokens:        shape [batch_size, truncated_backprop_length]
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            outputLayer:    handle to the last fully connected layer (an instance of nn.Linear)
            Embedding:      An instance of nn.Embedding. This is the embedding matrix.
            is_train:       flag: whether or not to feed in the predicated token vector as input for next step

        Returns:
            logits        : The predicted logits. shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_state : The hidden state from the last iteration (in time/words).
                            Shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train==True:
            seqLen = xTokens.shape[1] #truncated_backprop_length
        else:
            seqLen = 40 #Max sequence length to be generated

        # ToDo
        # While iterate through the (stacked) rnn, it may be easier to use lists instead of indexing the tensors.
        # You can use "list(torch.unbind())" and "torch.stack()" to convert from pytorch tensor to lists and back again.

        # get input embedding vectors
        input_embedding_vectors = Embedding(xTokens)
        
        # Need batch_size and vocabulary_size
        # to initialize empty logits and
        # current_state tensors with correct shape.
        batch_size, _ = xTokens.shape
        vocabulary_size = outputLayer.out_features
        
        # Emtpy tensor with correct shape for using torch.cat((logits, x), dim=1)
        # containing recurrent outputs of the final dense layer. 
        #logits = torch.empty(batch_size, 0, vocabulary_size)
        logits = torch.empty(batch_size, 0, vocabulary_size, device="cuda:0")
        
        # Empty tensor to store the hidden states for all layers in a recurrence
        #current_state = torch.empty(0, batch_size, self.hidden_state_size)
        current_state = torch.empty(0, batch_size, self.hidden_state_size, device="cuda:0")
        
        # Empty tensor to store all hidden states for all layers for all recurrences
        #hidden_states = torch.empty(0, self.num_rnn_layers, batch_size, self.hidden_state_size)
        hidden_states = torch.empty(0, self.num_rnn_layers, batch_size, self.hidden_state_size, device="cuda:0")

        # Use for loops to run over "seqLen" and "self.num_rnn_layers" to calculate logits        
        for recurrence_number in range(seqLen):
            
            if recurrence_number > 0:
            
                # Store the (previous) current_state in hidden_states
                hidden_states = \
                torch.cat((hidden_states, current_state.view(1, self.num_rnn_layers, batch_size, self.hidden_state_size)), dim=0)
                # Empty the current_state (new recurrence)
                #current_state = torch.empty(0, batch_size, self.hidden_state_size)
                current_state = torch.empty(0, batch_size, self.hidden_state_size, device="cuda:0")
            
            for layer_number, state_old in enumerate(initial_hidden_state):
                
                if layer_number == 0:
                    
                    if is_train:
                        # Get the embedding vector
                        # for the correct input word.
                        x = input_embedding_vectors[:, recurrence_number, :]
                    
                    else:
                        if recurrence_number == 0:
                            # First recurrence input
                            # is the (word) embedding for 
                            # the first word.
                            x = input_embedding_vectors[:, recurrence_number, :]
                        else:
                            # Each later input is
                            # an embedding of a
                            # predicted word.
                            x = Embedding(pred)
                
                if recurrence_number > 0:
                    # No longer using initial 
                    # hidden states.
                    # Get the previous hidden state
                    # for the correct cell.
                    state_old = hidden_states[-1][layer_number]
                    
                # Forward the cell.
                state_new = self.cells[layer_number].forward(x, state_old)
                
                # Append the current state for the layer to current_state
                current_state = \
                torch.cat((current_state, state_new.view(1, batch_size, self.hidden_state_size)), dim=0)
                
                if layer_number == self.num_rnn_layers - 1:
                    # Last layer.
                    
                    # The output of the layer is forwarded into
                    # the final dense layer.
                    
                    # Forward the final state of the
                    # recurrent layer time instance
                    # into a fully connected (dense)
                    # layer to get values that each can be 
                    # interpreted as a "logistic unit"
                    # (logit) for prediction.
                    logit = outputLayer.forward(state_new)
                    
                    if not is_train:
                        # Predict the next word
                        # by locating the index of the 
                        # word with the largest
                        # logit.
                        # Done for each example in the batch.
                        pred = torch.argmax(logit, dim=1)
                    
                    # Append the logit values (reshaped)
                    # to the logits tensor containing the logit
                    # behind the predicted word for the entire
                    # recurrentce (sequence/sentence length).
                    logits = torch.cat((logits, logit.view(batch_size, 1, vocabulary_size)), dim=1)
                    
                else:
                    # Intermediate layer.
                    
                    # The output of the layer (cell) is: 
                    # y = 1*state_new = x in next layer.
                    x = state_new
        
        return logits, current_state

########################################################################################################################
class GRUCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super().__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight_u: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight_r: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean. 

            self.bias_u: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias_r: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_sizes = hidden_state_size
        
        # TODO:
        self.weight_u = nn.Parameter(\
                                     nn.init.normal_(\
                                                     torch.empty(\
                                                                 self.hidden_state_sizes+input_size, \
                                                                 self.hidden_state_sizes), \
                                                     mean=0, \
                                                     std=np.sqrt(1/(self.hidden_state_sizes+input_size))
                                                    )\
                                    )
        
        self.bias_u = nn.Parameter(\
                                   torch.zeros(\
                                               1, \
                                               self.hidden_state_sizes)\
                                  )

        self.weight_r = nn.Parameter(\
                                     nn.init.normal_(\
                                                     torch.empty(\
                                                                 self.hidden_state_sizes+input_size, \
                                                                 self.hidden_state_sizes), \
                                                     mean=0, \
                                                     std=np.sqrt(1/(self.hidden_state_sizes+input_size))
                                                    )\
                                    )
        
        self.bias_r = nn.Parameter(\
                                   torch.zeros(\
                                               1, \
                                               self.hidden_state_sizes)\
                                  )

        self.weight = nn.Parameter(\
                                   nn.init.normal_(\
                                                   torch.empty(\
                                                               self.hidden_state_sizes+input_size, \
                                                               self.hidden_state_sizes), \
                                                   mean=0, \
                                                   std=np.sqrt(1/(self.hidden_state_sizes+input_size))
                                                  )\
                                  )
        
        self.bias = nn.Parameter(\
                                 torch.zeros(\
                                             1, \
                                             self.hidden_state_sizes)\
                                )
        
        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO:        
        update_gate = torch.sigmoid( torch.mm(torch.cat((x, state_old), dim=1), self.weight_u) + self.bias_u )
        
        reset_gate = torch.sigmoid( torch.mm(torch.cat((x, state_old), dim=1), self.weight_r) + self.bias_r )
        
        cand_cell = torch.tanh( torch.mm(torch.cat((x, reset_gate*state_old), dim=1), self.weight) + self.bias )
        
        fin_cell = update_gate*state_old + (1 - update_gate)*cand_cell
        
        state_new = fin_cell
        return state_new

######################################################################################################################
class RNNCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super().__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight: A nn.Parameter with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size

        # TODO:
        self.weight = nn.Parameter(\
                                   nn.init.normal_(\
                                                   torch.empty(\
                                                               self.hidden_state_size + \
                                                               input_size, \
                                                               self.hidden_state_size), \
                                                   mean=0, \
                                                   std=np.sqrt(1/(self.hidden_state_size+input_size))\
                                                  )\
                                  )
                                   
        self.bias   = nn.Parameter(torch.zeros(1, self.hidden_state_size))
        
        return


    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO:
        state_new = torch.tanh( torch.mm(torch.cat((x, state_old), dim=1), self.weight) + self.bias )
        
        return state_new

######################################################################################################################
def loss_fn(logits, yTokens, yWeights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits          : shape[batch_size, truncated_backprop_length, vocabulary_size]
        yTokens (labels): Shape[batch_size, truncated_backprop_length]
        yWeights        : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only from words exsisting 
                          (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sumLoss: The total cross entropy loss for all words
        meanLoss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 0.0000000001 #used to not divide on zero
    
    # TODO:
    losses = F.cross_entropy(logits.permute(0, 2, 1), yTokens, reduction="none")
    
    losses_masked = torch.masked_select(losses, yWeights.eq(1))
    
    sumLoss = torch.sum(losses_masked)

    meanLoss = torch.mean(losses_masked)

    return sumLoss, meanLoss
