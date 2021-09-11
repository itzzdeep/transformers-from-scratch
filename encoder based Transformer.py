
import torch
import torch.nn as nn

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Utility function
    
    
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval
    
    

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,no_heads,mask=False):
        super(MultiHeadAttention,self).__init__()
        self.emb = embed_dim
        self.h = no_heads
        self.s = embed_dim // no_heads
        assert (self.s*self.h == self.emb) , "Embedding dimension must be divisible by no of heads"
        self.values = nn.Linear(embed_dim,embed_dim,bias=False)
        self.keys = nn.Linear(embed_dim,embed_dim,bias=False)
        self.querys = nn.Linear(embed_dim,embed_dim,bias=False)
        self.ff = nn.Linear(embed_dim,embed_dim)
        self.mask = mask
        
    def forward(self,x):
        b,t,e = x.size()
        s = e//self.h
        h = self.h
        assert (e == self.emb) , "Embed dimension should match input dimension"
        
        #linear
        query = self.querys(x)
        value = self.values(x)
        key = self.keys(x)
        
        query = query.view(b,t,h,s)
        value = value.view(b,t,h,s)
        key = key.view(b,t,h,s)
        
        query = query.transpose(1,2).contiguous().view(b*h,t,s)
        value = value.transpose(1,2).contiguous().view(b*h,t,s)
        key = key.transpose(1,2).contiguous().view(b*h,t,s)
        
        out = torch.bmm(query, key.transpose(1,2)).view(b*h,t,t)
        
        if self.mask:
            out = mask_(out,maskval='-inf',mask_diagonal=False)
        
        weights = nn.functional.softmax(out//(e**0.5),dim=2)
        
        attention = torch.bmm(weights,value).view(b,h,t,s)
        #concat
        attention = attention.transpose(1,2).contiguous().view(b,t,h*s)
        #linear
        out = self.ff(attention)
        
        return out  
 

class TransformerBlock(nn.Module):
    
    """
    TransformerBlock is one single block of the encoder layer 
    We will stack this layer sequentially for the number of depths
    to get the actual Encoder layer
    """

    def __init__(self, emb=512, heads=8, mask=False, seq_length=100,forward_expansion=4, dropout=0.0):
        
        """
        Args:
        
        emb(:obj:`int`): embedding dimensions 
                          default to 512
        heads(:obj:`int`): number of heads 
                            default to 8
        mask(:obj:`boolean`): Wheather mask will be applied or not
                              default is False
        seq_length(:obj:`int`): max length of the sequence
                                default is 100
        forward_expansion(:obj:`int`): expansion of dimension in feed forward block
                                       default to 4
        dropout(:obj:`float`): dropout
                               default to 0
                               
        Return
        
        
        """
        
        super().__init__()
        
        self.mask = mask
        
        self.attention = MultiHeadAttention(embed_dim=512,no_heads=8,mask=mask)

        # layer normalization 
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        
        # feed forward block
        # map into 4*dimensions and after applying relu maps to original dimension
        self.ff = nn.Sequential(

            nn.Linear(emb, forward_expansion * emb),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb, emb)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # straingt forward implementation as per diagram
        
        attended = self.attention(x)
        
        # 1st residual connection with layer normalization and dropout
        x = self.norm1(attended + x)

        x = self.dropout(x)

        fedforward = self.ff(x)
        
        # 2nd residual connection with layer normalization and dropout
        x = self.norm2(fedforward + x)

        x = self.dropout(x)

        return x
    
    

class Encoder(nn.Module):
    
    """
    Enoder part of Transformers architechture 
    Stack of #(depth) layers of TransformerBlock 
    Add token embedding and position embedding
    
    """
    def __init__(self,src_vocab=10,depth=6, emb=512, heads=8,
                 mask=False, seq_length=100,forward_expansion=4,
                 dropout=0.0,device="cpu"):
        """
        Args:
        
        emb(:obj:`int`): embedding dimensions 
                          default to 512
        heads(:obj:`int`): number of heads 
                            default to 8
        mask(:obj:`boolean`): Wheather mask will be applied or not
                              default is False
        seq_length(:obj:`int`): max length of the sequence
                                default is 100
        forward_expansion(:obj:`int`): expansion of dimension in feed forward block
                                       default to 4
        dropout(:obj:`float`): dropout
                               default to 0
        
        device(:ojj) : cpu or gpu
                       default is cpu
                               
        Return
        
        
        """
        super().__init__()
        
        # Here I am using a position embedding instead of positional encoding
        self.position = nn.Embedding(seq_length,emb)
        
        # Its basically the token embedding from the tokens. i have used the name of word embedding
        self.word = nn.Embedding(src_vocab,emb)
        
        self.dropout = nn.Dropout(dropout)
        
        # stack of 6 (default) layers of transformerblock
        self.layers = nn.ModuleList([TransformerBlock(emb,heads, mask=mask,
                                                       seq_length=seq_length,forward_expansion=forward_expansion,
                                                       dropout=dropout) for _ in range(depth)])
        
        # for gpu computing if available
        self.device = device

    def forward(self, x):
        
        word_embedding = self.word(x)
        
        b,t,e = word_embedding.size()
        
        # aranging a list of postion (0 to seq_length)
        # and then adding a dimension and reshaping to the dimension of word embedding
        # as we need to add both
        position = torch.arange(t).to(self.device)
        position_embedding = self.position(position)[None,:,:].expand(b,t,e)
        
        assert (position_embedding.size() == word_embedding.size()) , " position and token embedding dimension need to be same"
        
        # adding the 
        x = self.dropout(word_embedding+position_embedding)
        
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    
    
    
class ObjectiveTransformer(nn.Module):
    """
    ObjectiveTransformer is a basic implementation of Encoder only transformers architechture
    Any problem that has a Encoder only objective
    
    """
    def __init__(self,src_vocab=10,depth=6, emb=512, heads=8,
                 mask=False, seq_length=100,forward_expansion=4,
                 dropout=0.0,device="cpu"):
        super().__init__()
        
        self.encoder = Encoder(src_vocab,depth, emb, heads,
                 mask, seq_length,forward_expansion,
                 dropout,device)
        
        
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        
        encode = self.encoder(x)
        
        b,t,e = encode.size()
        
        flat = self.flatten(encode)
        
        ff = nn.Linear(t*e,1)(flat)
        
        out = self.sigmoid(ff)
        
        return out
        
        
        
        
##########################################################################

# Testing with some random value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([[7, 4, 5, 6, 2, 5, 6, 7, 6], [9,6, 2, 5, 6, 5, 6, 7, 2]])


x.size() # output ->  torch.Size([2, 9])

ObjectiveTransformer(device=device)(x).size()  # output ->   torch.Size([2, 1])


