import torch
import torch.nn as nn
import json
import DataLoad as DLoad


def Transpose(Tensor, Dimension1, Dimension2):
    return torch.transpose(Tensor, Dimension1, Dimension2)


def CreateAndInit(*Dimsions):
    W = torch.zeros(Dimsions, dtype=torch.float)
    if len(Dimsions) > 1:
        torch.nn.init.kaiming_uniform_(W, mode='fan_in', nonlinearity='relu')
    return W


def MatMul(TensorInput1, TensorInput2):
    return torch.matmul(TensorInput1, TensorInput2)


def Reshape(TensorInput, Shape):
    return TensorInput.reshape(Shape)


def VerticalMaskByMinusInfinity(TensorInput, BoolTensorMask):
    assert len(TensorInput.size()) == len(BoolTensorMask.size())
    assert BoolTensorMask.size()[-2] == 1
    return TensorInput.masked_fill(BoolTensorMask, -1e10)


def HorizontalMaskByZero(TensorInput, BoolTensorMask):
    assert len(TensorInput.size()) == len(BoolTensorMask.size())
    assert BoolTensorMask.size()[-1] == 1
    return TensorInput.masked_fill(BoolTensorMask, 0.0)


def Softmax(TensorInput, Dimension):
    return torch.nn.functional.softmax(TensorInput, Dimension)


def Sqrt(TensorInput):
    return torch.sqrt(TensorInput)


def VerticalMask(Mask):
    B, L = Mask.size()
    return Reshape(Mask, [B, 1, 1, L])


def HorizontalMask(Mask):
    B, L = Mask.size()
    return Reshape(Mask, [B, 1, L, 1])


def Attention(Query, Key, Value, QMask, KMask, Dropout):
    # Q=[B,H,L,E]
    # K=[B,H,L,E]
    # V=[B,H,L,E]
    LInd, EInd = 2, 3
    EmbeddingLength = torch.Tensor([Query.size()[-1]])
    if torch.cuda.is_available():
        EmbeddingLength = EmbeddingLength.cuda()
    QMask = ~HorizontalMask(QMask)
    KMask = ~VerticalMask(KMask)
    QKTransposeScale = MatMul(Query, Transpose(
        Key, LInd, EInd))/Sqrt(EmbeddingLength)
    InfinityMasked = VerticalMaskByMinusInfinity(QKTransposeScale, KMask)
    Softmaxed = Softmax(InfinityMasked, -1)
    HorizontalMaksed = HorizontalMaskByZero(Softmaxed, QMask)
    return MatMul(Dropout(HorizontalMaksed), Value), HorizontalMaksed


class MultiHeadAttention(nn.Module):
    def __init__(self, HeadNum, EmbeddingSize, MaxLength, Dropout):
        super().__init__()
        assert EmbeddingSize % HeadNum == 0
        self.HeadNum = HeadNum
        self.HeadEmbeddingLength = int(EmbeddingSize/HeadNum)
        self.EmbeddingSize = EmbeddingSize
        self.MaxLength = MaxLength
        self.HeadLinear = torch.nn.Parameter(CreateAndInit(
            HeadNum, self.HeadEmbeddingLength, self.HeadEmbeddingLength))
        self.OutputLiner = torch.nn.Parameter(
            CreateAndInit(EmbeddingSize, EmbeddingSize))
        self.Dropout = torch.nn.Dropout(Dropout)

    def forward(self, Q, K, V, QMask, KMask):
        # Q=[B,L,E]
        # K=[B,L,E]
        # V=[B,L,E]
        Q, K, V = self.SplitAndPreprocess(Q, K, V)
        # print(Q.size())
        # print(K.size())
        # print(V.size())
        AttResult, Atte = Attention(Q, K, V, QMask, KMask, self.Dropout)
        # print(AttResult.size())
        # print(Atte.size())
        # print(Atte)
        L, E = self.MaxLength, self.EmbeddingSize
        HInd, LInd = 1, 2
        #print(Transpose(AttResult, HInd, LInd).size())
        # print(E)
        #print(Reshape(Transpose(AttResult, HInd, LInd), [-1, L, E]).size())
        return MatMul(Reshape(Transpose(AttResult, HInd, LInd), [-1, L, E]), self.OutputLiner)

    def SplitAndPreprocess(self, Q, K, V):
        L, H, EDivH = self.MaxLength, self.HeadNum, self.HeadEmbeddingLength
        LInd, HInd = 1, 2
        Q = MatMul(
            Transpose(Reshape(Q, [-1, L, H, EDivH]), LInd, HInd), self.HeadLinear)
        K = MatMul(
            Transpose(Reshape(K, [-1, L, H, EDivH]), LInd, HInd), self.HeadLinear)
        V = MatMul(
            Transpose(Reshape(V, [-1, L, H, EDivH]), LInd, HInd), self.HeadLinear)
        return Q, K, V


class PostionwiseFeedForward(nn.Module):
    def __init__(self, EmbeddingLength):
        super().__init__()
        self.MParameterList = torch.nn.ParameterList([torch.nn.Parameter(
            CreateAndInit(EmbeddingLength, EmbeddingLength))for i in range(2)])
        self.bParameterList = torch.nn.ParameterList(
            [torch.nn.Parameter(CreateAndInit(EmbeddingLength)) for i in range(2)])
        self.Relu = torch.nn.ReLU()

    def forward(self, Input):
        return MatMul(self.Relu(MatMul(Input, self.MParameterList[0])+self.bParameterList[0]), self.MParameterList[1])+self.bParameterList[1]


class EncoderLayer(nn.Module):
    def __init__(self, HeadNum, EmbeddingLength, MaxLength, Dropout):
        super().__init__()
        self.HeadNum = HeadNum
        self.EmbeddingLength = EmbeddingLength
        self.MaxLength = MaxLength
        self.MultiHeadAttention = MultiHeadAttention(
            HeadNum, EmbeddingLength, MaxLength, Dropout)
        self.PostionwiseFeedForward = PostionwiseFeedForward(EmbeddingLength)
        self.LayerNorm = torch.nn.LayerNorm([EmbeddingLength])
        self.AttentionDropout = torch.nn.Dropout(Dropout)
        self.FeedForwardDropout = torch.nn.Dropout(Dropout)

    def forward(self, Input, SrcMask):
        SelfAttentionResult = self.MultiHeadAttention(
            Input, Input, Input, SrcMask, SrcMask)
        LayerNormResultAfterAttention = self.LayerNorm(
            self.AttentionDropout(SelfAttentionResult)+Input)

        PostionwiseFeedForwardResult = self.PostionwiseFeedForward(
            LayerNormResultAfterAttention)
        LayerNormResultAfterFeedForward = self.LayerNorm(
            self.FeedForwardDropout(PostionwiseFeedForwardResult)+LayerNormResultAfterAttention)
        return LayerNormResultAfterFeedForward


class Encoder(nn.Module):
    def __init__(self, HeadNum, EmbeddingLength, MaxLength, LayerNum, Dropout):
        super().__init__()
        self.LayerList = torch.nn.ModuleList(
            [EncoderLayer(HeadNum, EmbeddingLength, MaxLength, Dropout) for i in range(LayerNum)])

    def forward(self, Input, SrcMask):
        Output = Input
        for Layer in self.LayerList:
            Output = Layer(Output, SrcMask)
        return Output


class PositionalEmbedding(nn.Module):
    def __init__(self, EmbeddingLength, MaxLength):
        super().__init__()
        self.EmbeddingLength = EmbeddingLength
        self.MaxLength = MaxLength
        self.CacheEmbedding = self.BuildPostionalEmbedding()
        if torch.cuda.is_available():
            self.CacheEmbedding = self.CacheEmbedding.cuda()

    def BuildPostionalEmbedding(self):
        def VerticalMaskByZero(TensorInput, BoolTensorMask):
            assert len(TensorInput.size()) == len(BoolTensorMask.size())
            assert BoolTensorMask.size()[-2] == 1
            Zero = 0.0
            # print(TensorInput.device))
            return TensorInput.masked_fill(BoolTensorMask, Zero)

        PositionSequence = Reshape(torch.Tensor(
            [i for i in range(self.MaxLength)]), [self.MaxLength, 1])
        EmbeddingIndSequenct = Reshape(torch.Tensor(
            [i for i in range(self.EmbeddingLength)]), [1, self.EmbeddingLength])
        PrimaryEmbedding = PositionSequence / \
            (torch.pow(torch.Tensor([10000]), 2.0 *
                       EmbeddingIndSequenct/self.EmbeddingLength))
        CosMask = Reshape(torch.BoolTensor(
            [i % 2 for i in range(self.EmbeddingLength)]), [1, self.EmbeddingLength])
        # print(CosMask.size())
        SinMask = ~CosMask
        # print(SinMask)
        #print(VerticalMaskByZero(PrimaryEmbedding, CosMask))
        #print(VerticalMaskByZero(PrimaryEmbedding, SinMask))
        return torch.cos(VerticalMaskByZero(PrimaryEmbedding, CosMask))+torch.sin(VerticalMaskByZero(PrimaryEmbedding, SinMask))

    def forward(self, Mask):
        # Mask=[B,L]
        Mask = HorizontalMask(~Mask)
        #print(Reshape(Mask, [-1, self.MaxLength, 1]).size())
        # print(Reshape(self.CacheEmbedding, [
        #     1, self.MaxLength, self.EmbeddingLength]).size())
        return HorizontalMaskByZero(Reshape(self.CacheEmbedding, [1, self.MaxLength, self.EmbeddingLength]), Reshape(Mask, [-1, self.MaxLength, 1]))


class DecoderLayer(nn.Module):
    def __init__(self, HeadNum, EmbeddingLength, MaxLength, Dropout):
        super().__init__()
        self.HeadNum = HeadNum
        self.EmbeddingLength = EmbeddingLength
        self.MaxLength = MaxLength
        self.SelfMultiHeadAttention = MultiHeadAttention(
            HeadNum, EmbeddingLength, MaxLength, Dropout)
        self.EnDeMultiHeadAttention = MultiHeadAttention(
            HeadNum, EmbeddingLength, MaxLength, Dropout)
        self.PostionwiseFeedForward = PostionwiseFeedForward(EmbeddingLength)
        self.LayerNorm = torch.nn.LayerNorm([EmbeddingLength])
        self.SelfAttentionDropout = torch.nn.Dropout(Dropout)
        self.EnDeAttentionDropout = torch.nn.Dropout(Dropout)
        self.FeedForwardDropout = torch.nn.Dropout(Dropout)

    def forward(self, Input, Memory, SrcMask, TgtMask):
        # selfAttention
        SelfAttentionResult = self.SelfMultiHeadAttention(
            Input, Input, Input, TgtMask, TgtMask)
        # Residual And LayerNorm
        LayerNormResultAfterSelfAttention = self.LayerNorm(
            self.SelfAttentionDropout(SelfAttentionResult)+Input)
        # EnDeAttention
        EnDeAttentionResult = self.EnDeMultiHeadAttention(
            LayerNormResultAfterSelfAttention, Memory, Memory, TgtMask, SrcMask)
        # Residual And LayerNorm
        LayerNormResultAfterEnDeAttention = self.LayerNorm(
            self.EnDeAttentionDropout(EnDeAttentionResult)+LayerNormResultAfterSelfAttention)
        # FFN
        PostionwiseFeedForwardResult = self.PostionwiseFeedForward(
            LayerNormResultAfterEnDeAttention)
        # Residual And LayerNorm
        LayerNormResultAfterFeedForward = self.LayerNorm(
            self.FeedForwardDropout(PostionwiseFeedForwardResult)+LayerNormResultAfterEnDeAttention)
        return LayerNormResultAfterFeedForward


class Decoder(nn.Module):
    def __init__(self, HeadNum, EmbeddingLength, MaxLength, LayerNum, Dropout):
        super().__init__()
        self.LayerList = torch.nn.ModuleList(
            [DecoderLayer(HeadNum, EmbeddingLength, MaxLength, Dropout) for i in range(LayerNum)])

    def forward(self, Input, Memory, SrcMask, TgtMask):
        Output = Input
        for Layer in self.LayerList:
            Output = Layer(Output, Memory, SrcMask, TgtMask)
        return Output


class Generator(nn.Module):
    def __init__(self, VocabularySize, EmbeddingSize):
        super().__init__()
        self.MParameter = torch.nn.Parameter(
            CreateAndInit(EmbeddingSize, VocabularySize))
        self.Softmax = torch.nn.Softmax()

    def forward(self, Input):
        return Softmax(MatMul(Input, self.MParameter), -1)


class TransformerNMTModel(nn.Module):
    def __init__(self, HeadNum, EmbeddingSize, SrcVocabularySize, TgtVocabularySize, MaxLength, EncoderLayerNum, DecoderLayerNum, Dropout=0.1):
        super().__init__()
        self.PositionalEmbedding = PositionalEmbedding(
            EmbeddingSize, MaxLength)
        self.PositionalEmbeddingDropout = torch.nn.Dropout(Dropout)

        self.SrcEmbedding = torch.nn.Embedding(
            SrcVocabularySize, EmbeddingSize, padding_idx=0)
        self.Encoder = Encoder(HeadNum, EmbeddingSize,
                               MaxLength, EncoderLayerNum, Dropout)

        self.TgtEmbedding = torch.nn.Embedding(
            TgtVocabularySize, EmbeddingSize, padding_idx=0)
        self.Decoder = Decoder(HeadNum, EmbeddingSize,
                               MaxLength, DecoderLayerNum, Dropout)

        self.Generator = Generator(TgtVocabularySize, EmbeddingSize)

    def forward(self, SrcIndexIntTensorInput, TgtIndexIntTensorInput, SrcMask, TgtMask):
        assert SrcIndexIntTensorInput.size()[0:2] == TgtIndexIntTensorInput.size()[
            0:2] == SrcMask.size()[0:2] == TgtMask.size()[0:2]
        SrcTensorInput = self.SrcEmbedding(SrcIndexIntTensorInput)
        SrcTensorInput = self.PositionalEmbeddingDropout(
            SrcTensorInput+self.PositionalEmbedding(SrcMask))
        Memory = self.Encoder(SrcTensorInput, SrcMask)

        TgtTensorInput = self.TgtEmbedding(TgtIndexIntTensorInput)
        TgtTensorInput = self.PositionalEmbeddingDropout(
            TgtTensorInput)+self.PositionalEmbedding(TgtMask)
        DecoderOuput = self.Decoder(TgtTensorInput, Memory, SrcMask, TgtMask)
        return self.Generator(DecoderOuput)

    def Save(self, Path):
        torch.save(self.state_dict(), Path)

    def Load(self, Path):
        self.load_state_dict(torch.load(Path))

    def Train(self):
        self.train()

    def Eval(self):
        self.eval()


def BatchLengthToBoolTensorMask(NumLength, MaxLength):
    def NumToNumMask(Num, MaxLength):
        NumMask = [0 for i in range(MaxLength)]
        for Ind in range(Num):
            NumMask[Ind] = 1
        return NumMask
    return torch.BoolTensor([NumToNumMask(Num, MaxLength) for Num in NumLength])
class LabelSmooth(nn.Module):
    def __init__(self,LabelNum,SmoothRate=0.1):
        super().__init__()
        self.LabelNum=LabelNum
        self.SmoothRate=SmoothRate
    def forward(self,Input):
        InputSize=Input.size()
        B=InputSize[0]
        L=InputSize[1]
        Output=torch.ones(B,L,self.LabelNum)*(self.SmoothRate/self.LabelNum).cuda()
        if torch.cuda.is_available():
            Output=Output.cuda()
        Input=Reshape(Input,[B,L,1])
        return torch.scatter(Output,-1,Input,0.9+(self.SmoothRate/self.LabelNum))


        

def TrainBuildTransformer():
    with open("Model/Config.json") as Fd:
        ConfigDict = json.load(Fd)
        MaxLength = ConfigDict["MaxLength"]
        BatchSize = ConfigDict["BatchSize"]
        EmbeddingSize = ConfigDict["EmbeddingSize"]
        HeadNum = ConfigDict["HeadNum"]
        EnLayer = ConfigDict["EnLayer"]
        DeLayer = ConfigDict["DeLayer"]
        Dropout = ConfigDict["Dropout"]
        print("Loading Tgt vocab")
        TgtDict = DLoad.LoadVocabulary("Data/tgt.vocab")
        print("Tgt vocab Loading finished")
        print("Loading Src vocab")
        SrcDict = DLoad.LoadVocabulary("Data/src.vocab")
        print("Src vocab Loadinf finished")
        # SrcIndSentences, SrcLength, SrcDict = DLoad.LoadData(
        #    "Data/src.sents", "Data/src.vocab", MaxLength)
        # TgtIndSentences, TgtLength, TgtDict = DLoad.LoadData(
        #    "Data/tgt.sents", "Data/tgt.vocab", MaxLength)
        # TrainDataset = DLoad.TrainCorpusDataset(
        #    SrcIndSentences, SrcLength, TgtIndSentences, TgtLength)
        #BatchDatas = DLoad.TrainDataLoaderCreator(TrainDataset, BatchSize)
        SrcVocabularySize = SrcDict.VocabularySize()
        TgtVocabularySize = TgtDict.VocabularySize()
        print("Building Model")
        Trans = TransformerNMTModel(HeadNum, EmbeddingSize, SrcVocabularySize,
                                    TgtVocabularySize, MaxLength, EnLayer, DeLayer, Dropout)
        print("Model building finished")
        # return Trans, BatchDatas, SrcDict, TgtDict, MaxLength, EmbeddingSize
        return Trans, BatchSize, SrcDict, TgtDict, MaxLength, EmbeddingSize


def TestBuildTransformer():
    with open("Model/Config.json") as Fd:
        ConfigDict = json.load(Fd)
        MaxLength = ConfigDict["MaxLength"]
        BatchSize = ConfigDict["BatchSize"]
        EmbeddingSize = ConfigDict["EmbeddingSize"]
        HeadNum = ConfigDict["HeadNum"]
        EnLayer = ConfigDict["EnLayer"]
        DeLayer = ConfigDict["DeLayer"]
        SrcIndSentences, SrcLength, SrcDict = DLoad.LoadData(
            "Data/test.sents", "Data/src.vocab", MaxLength)
        TgtDict = DLoad.LoadVocabulary("Data/tgt.vocab")
        TestDataset = DLoad.TestCorpusDataset(SrcIndSentences, SrcLength)
        BatchDatas = DLoad.TestDataLoaderCreator(TestDataset, BatchSize)
        SrcVocabularySize = SrcDict.VocabularySize()
        TgtVocabularySize = TgtDict.VocabularySize()
        print("Building Model")
        Trans = TransformerNMTModel(
            HeadNum, EmbeddingSize, SrcVocabularySize, TgtVocabularySize, MaxLength, EnLayer, DeLayer)
        print("Model building finished")
        return Trans, BatchDatas, SrcDict, TgtDict, MaxLength
