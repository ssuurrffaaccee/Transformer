import Transformer as T
import torch as t
import numpy as np
def Test1():
    B,H,L,E=1,1,5,4
    #SrcMask=t.BoolTensor(np.random.binomial(1,0.5,[1,5]))
    #TgtMask=t.BoolTensor(np.random.binomial(1,0.5,[1,5]))\
    SrcMask=t.BoolTensor(np.array([[1,1,1,1,0]]))
    TgtMask=t.BoolTensor(np.array([[1,1,0,0,0]]))
    Src=t.Tensor(np.random.randn(B,H,L,E))
    Tgt=t.Tensor(np.random.randn(B,H,L,E))
    x,a=T.Attention(Tgt,Src,Src,TgtMask,SrcMask)
    print(x.size())
    print(SrcMask)
    print(TgtMask)
    print(x)
    print(a)
def Test2():
    x=T.VerticalMask(t.BoolTensor(np.random.binomial(1,0.5,[32,500])))
    y=T.HorizontalMask(t.BoolTensor(np.random.binomial(1,0.5,[32,500])))
    print(x.size())
    print(y.size())
    #print(x)
def Test3():
    B,H,L=2,4,6
    E=H*2
    MH=T.MultiHeadAttention(HeadNum=H,EmbeddingSize=E,MaxLength=L)
    #SrcMask=t.BoolTensor(np.random.binomial(1,0.5,[1,5]))
    #TgtMask=t.BoolTensor(np.random.binomial(1,0.5,[1,5]))\
    SrcMask=t.BoolTensor(np.array([[1,1,1,1,0,0],[1,1,1,0,0,0]]))
    TgtMask=t.BoolTensor(np.array([[1,1,0,0,0,0],[1,1,1,0,0,0]]))
    Src=t.Tensor(np.random.randn(B,L,E))
    Tgt=t.Tensor(np.random.randn(B,L,E))
    #x,a=T.Attention(Tgt,Src,Src,TgtMask,SrcMask)
    x=MH(Tgt,Src,Src,TgtMask,SrcMask)
    print(x.size())
    print(SrcMask)
    print(TgtMask)
    print(x)
def Test4():
    B,L,E=2,6,5
    PFF=T.PostionwiseFeedForward(5)
    Src=t.Tensor(np.random.randn(B,L,E))
    x=PFF(Src)
    print(x.size())
def Test5():
    B,H,L,E=2,2,6,4
    EL=T.EncoderLayer(H,E,L)
    SrcMask=t.BoolTensor(np.array([[1,1,1,1,0,0],[1,1,1,0,0,0]]))
    Src=t.Tensor(np.random.randn(B,L,E))
    x=EL(Src,SrcMask)
    print(x.size())
def Test6():
    B,H,L,E=2,2,6,4
    En=T.Encoder(H,E,L,3)
    SrcMask=t.BoolTensor(np.array([[1,1,1,1,0,0],[1,1,1,0,0,0]]))
    Src=t.Tensor(np.random.randn(B,L,E))
    x=En(Src,SrcMask)
    print(x.size())
def Test7():
    B,H,L,E=2,2,6,50
    PE=T.PositionalEmbedding(E,L)
    SrcMask=t.BoolTensor(np.array([[1,1,1,1,0,0],[1,1,1,0,0,0]]))
    x=PE(SrcMask)
    print(x.size())
    print(x)
def Test8():
    B,H,L,E=2,2,6,4
    De=T.DecoderLayer(H,E,L)
    SrcMask=t.BoolTensor(np.array([[1,1,1,1,0,0],[1,1,1,0,0,0]]))
    TgtMask=t.BoolTensor(np.array([[1,1,0,0,0,0],[1,1,1,0,0,0]]))
    Memory=t.Tensor(np.random.randn(B,L,E))
    Tgt=t.Tensor(np.random.randn(B,L,E))
    De(Tgt,Memory,SrcMask,TgtMask)
def Test9():
    B,H,L,E=2,2,6,4
    De=T.Decoder(H,E,L,2)
    SrcMask=t.BoolTensor(np.array([[1,1,1,1,0,0],[1,1,1,0,0,0]]))
    TgtMask=t.BoolTensor(np.array([[1,1,0,0,0,0],[1,1,1,0,0,0]]))
    Memory=t.Tensor(np.random.randn(B,L,E))
    Tgt=t.Tensor(np.random.randn(B,L,E))
    x=De(Tgt,Memory,SrcMask,TgtMask)
    print(x.size())
def Test10():
    VacabularySize=4
    B,L,E=1,2,4
    Gen=T.Generator(VacabularySize,E)
    Tgt=t.Tensor(np.random.randn(B,L,E))
    x=Gen(Tgt)
    print(x.size())
    print(x)
def Test11():
    B,H,L,E=2,4,6,8
    SrcVocab=10
    TgtVocab=12
    EnLayer=2
    Delayer=3
    Trans=T.TransformerNMTModel(H,E,SrcVocab,TgtVocab,L,EnLayer,Delayer)
    SrcMask=t.BoolTensor(np.array([[1,1,1,1,0,0],[1,1,1,0,0,0]]))
    TgtMask=t.BoolTensor(np.array([[1,1,0,0,0,0],[1,1,1,0,0,0]]))
    Src=t.LongTensor(np.random.binomial(SrcVocab,0.5,[B,L]))
    Tgt=t.LongTensor(np.random.binomial(SrcVocab,0.5,[B,L]))
    x=Trans(Src,Tgt,SrcMask,TgtMask)
    print(x.size())
    print(x[0][0])
#def Test12():
#    Infi=T.LoopIterator(100,range,10)
#    for x in Infi:
#        print(x)
import DataLoad as DL
def Test13():
    Dict=DL.LoadVocabulary("src.vocab")
    for i in range(80):
        print(Dict.GetWord(i))
def Test14():
    Sentences=DL.LoadSentences("src.sents")
    for Sent in Sentences:
        print(Sent)
def Test15():
    Sentences=DL.LoadSentences("src.sents")
    PaddedSentences,Length=DL.PaddingSentences(Sentences,30)
    for Sent in PaddedSentences:
        print(Sent)
    for L in Length:
        print(L)
def Test16():
    Sentences=DL.LoadSentences("src.sents")
    PaddedSentences,Length=DL.PaddingSentences(Sentences,30)
    Dict=DL.LoadVocabulary("src.vocab")
    IndSentences=DL.ChangePaddedSentencesToInd(PaddedSentences,Dict)
    print(Dict.VocabularySize())
    for Sent in IndSentences:
        print(Sent)
def Test17():
    IndSentences,Length,Dict=DL.LoadData("src.sents","src.vocab",30)
    for Sent in IndSentences:
        print(Sent)
    for L in Length:
        print(L)
def Test18():
    B,H,L,E=2,4,6,8
    SrcVocab=10
    TgtVocab=12
    EnLayer=2
    Delayer=3
    Trans=T.TransformerNMTModel(H,E,SrcVocab,TgtVocab,L,EnLayer,Delayer)
    print(Trans)
    for key in Trans.state_dict().keys():
        print(key)
    #SrcMask=t.BoolTensor(np.array([[1,1,1,1,0,0],[1,1,1,0,0,0]]))
    #TgtMask=t.BoolTensor(np.array([[1,1,0,0,0,0],[1,1,1,0,0,0]]))
    #Src=t.LongTensor(np.random.binomial(SrcVocab,0.5,[B,L]))
    #Tgt=t.LongTensor(np.random.binomial(SrcVocab,0.5,[B,L]))
    #x=Trans(Src,Tgt,SrcMask,TgtMask)
    #print(x.size())
    #print(x[0][0])
def Test19():
    B,H,L,E=2,4,6,8
    SrcVocab=10
    TgtVocab=12
    EnLayer=2
    Delayer=3
    Trans=T.TransformerNMTModel(H,E,SrcVocab,TgtVocab,L,EnLayer,Delayer)
    Trans.Save("Model/Trans.0")
def Test20():
    B,H,L,E=2,4,6,8
    SrcVocab=10
    TgtVocab=12
    EnLayer=2
    Delayer=3
    Trans=T.TransformerNMTModel(H,E,SrcVocab,TgtVocab,L,EnLayer,Delayer)
    Trans.Load("Model/Trans.0")
    SrcMask=t.BoolTensor(np.array([[1,1,1,1,0,0],[1,1,1,0,0,0]]))
    TgtMask=t.BoolTensor(np.array([[1,1,0,0,0,0],[1,1,1,0,0,0]]))
    Src=t.LongTensor(np.random.binomial(SrcVocab,0.5,[B,L]))
    Tgt=t.LongTensor(np.random.binomial(SrcVocab,0.5,[B,L]))
    x=Trans(Src,Tgt,SrcMask,TgtMask)
    print(x.size())
    print(x[0][0])
def Test21():
    MaxLength=30
    def CollateFunction(Batch):
        #print(len(Batch))
        OutputBatch={"SrcSent":[],"SrcLength":[],"TgtSent":[],"TgtLength":[]}
        for Elem in Batch:
            #print(Elem[0][0])
            OutputBatch["SrcSent"].append(Elem[0][0])
            OutputBatch["SrcLength"].append(Elem[0][1])
            OutputBatch["TgtSent"].append(Elem[1][0])
            OutputBatch["TgtLength"].append(Elem[1][1])
        #print(OutputBatch["SrcSent"])
        OutputBatch["SrcSent"]=t.LongTensor(OutputBatch["SrcSent"])
        OutputBatch["TgtSent"]=t.LongTensor(OutputBatch["TgtSent"])
        return OutputBatch
    SrcIndSentences,SrcLength,SrcDict=DL.LoadData("src.sents","src.vocab",MaxLength)
    TgtIndSentences,TgtLength,TgtDict=DL.LoadData("tgt.sents","tgt.vocab",MaxLength)
    TrainDataset=DL.TrainCorpusDataset(SrcIndSentences,SrcLength,TgtIndSentences,TgtLength)
    z=DL.TrainDataLoaderCreator(TrainDataset,4)
    Count=0
    while True:
        if Count==100:
            break
        Count=Count+1
        for x in z:
            print("Batch")
            print(x["SrcSent"].size())
    #print(DL.DataLoaderCreator(TrainDataset,4,CollateFunction).__next__())
def Test22():
    x=DL.Looper([1,2,3],4)
    for i in x:
        print(i)

def Test23():
    x=DL.Looper([1,2,3])
    for i in x:
        print(i)

def Test24():
    print(T.BatchLengthToBoolTensorMask([2,3,4],6))
def Test25():
    MaxLength=30
    BatchSize=2
    EmbeddingSize=4
    HeadNum=2
    SrcIndSentences,SrcLength,SrcDict=DL.LoadData("src.sents","src.vocab",MaxLength)
    TgtIndSentences,TgtLength,TgtDict=DL.LoadData("tgt.sents","tgt.vocab",MaxLength)
    TrainDataset=DL.TrainCorpusDataset(SrcIndSentences,SrcLength,TgtIndSentences,TgtLength)
    BatchDatas=DL.TrainDataLoaderCreator(TrainDataset,BatchSize)
    for Batch in BatchDatas:
        SrcSent=Batch["SrcSent"]
        print(SrcSent)
        SrcLength=Batch["SrcLength"]
        print(SrcLength)
        TgtSent=Batch["TgtSent"]
        print(TgtSent)
        TgtLength=Batch["TgtLength"]
        print(TgtLength)
def Test26():
    MaxLength=30
    BatchSize=2
    EmbeddingSize=4
    HeadNum=2
    EnLayer=2
    DeLayer=2
    SrcIndSentences,SrcLength,SrcDict=DL.LoadData("src.sents","src.vocab",MaxLength)
    TgtIndSentences,TgtLength,TgtDict=DL.LoadData("tgt.sents","tgt.vocab",MaxLength)
    TrainDataset=DL.TrainCorpusDataset(SrcIndSentences,SrcLength,TgtIndSentences,TgtLength)
    BatchDatas=DL.TrainDataLoaderCreator(TrainDataset,BatchSize)
    SrcVocabularySize=SrcDict.VocabularySize()
    TgtVocabularySize=TgtDict.VocabularySize()
    Trans=T.TransformerNMTModel(HeadNum,EmbeddingSize,SrcVocabularySize,TgtVocabularySize,MaxLength,EnLayer,DeLayer)
    for BatchInd,Batch in enumerate(BatchDatas):
        print("BegingBatch")
        SrcSent=Batch["SrcSent"]
        print(SrcSent.size())
        SrcLength=Batch["SrcLength"]
        #print(SrcLength.size())
        TgtSent=Batch["TgtSent"]
        print(TgtSent.size())
        TgtLength=Batch["TgtLength"]
        #print(TgtLength.size())
        SrcMask=T.BatchLengthToBoolTensorMask(SrcLength,MaxLength)
        TgtMask=T.BatchLengthToBoolTensorMask(TgtLength,MaxLength)
        Output=Trans(SrcSent,TgtSent,SrcMask,TgtMask)
        print("Step")
        print(BatchInd+1)
        print(Output.size())
        print(Output[0][2])
import json
def Test27():
    with open("Model/Config.json") as Fd:
        ConfigDict=json.load(Fd)
        print(ConfigDict.keys())
        print(ConfigDict.values())
def Test28():
    Trans,BatchDatas,SrcDict,TgtDict,MaxLength=T.TrainBuildTransformer()
    for BatchInd,Batch in enumerate(BatchDatas):
        SrcSent=Batch["SrcSent"]
        SrcLength=Batch["SrcLength"]
        TgtSent=Batch["TgtSent"]
        TgtLength=Batch["TgtLength"]
        SrcMask=T.BatchLengthToBoolTensorMask(SrcLength,MaxLength)
        TgtMask=T.BatchLengthToBoolTensorMask(TgtLength,MaxLength)
        Output=Trans(SrcSent,TgtSent,SrcMask,TgtMask)
        print("Step")
        print(BatchInd+1)
        print(Output.size())
        print(Output[0][2])
def Test29():
    Trans,BatchDatas,SrcDict,TgtDict,MaxLength=T.TestBuildTransformer()
    for BatchInd,Batch in enumerate(BatchDatas):
        SrcSent=Batch["SrcSent"]
        SrcLength=Batch["SrcLength"]
        SrcMask=T.BatchLengthToBoolTensorMask(SrcLength,MaxLength)
        #Output=Trans(SrcSent,TgtSent,SrcMask,TgtMask)
        print("Step")
        print(BatchInd+1)
        #print(Output.size())
        #print(Output[0][2])
import Translator as TT
def Test30():
    x=t.Tensor(3,10)
    y,z=TT.Max(x)
    print(x)
    print(y)
    print(z)
def Test31():
    TgtDict=DL.LoadVocabulary("Model/tgt.vocab")
    Out=TT.TranslateOutput(TgtDict,5).Init(4)
    print(Out.IndexSent)
    Out.Add([1,2,3,4])
    print(Out.IndexSent)
    Out.Add([2,3,4,5])
    print(Out.IndexSent)
    Out.Add([2,3,4,5])
    print(Out.AllFinish())
    print(Out.IndexSent)
    Out.Add([2,3,4,5])
    print(Out.AllFinish())
    Out.Add([2,3,4,5])
    Out.Add([2,3,4,5])
    print(Out.AllFinish())
    print(Out.GetCurrentIndexTensor())
    print(Out.IndexSent)
    print(Out.GetWordSent())
    print(Out.ToFile("Output/predict"))
def Test32():
    Trs=TT.GreedyTranslator("Model/Trans.0","Output/predict")
    x=Trs.GetTgtMask(2,4)
    print(x)
    y=t.Tensor(5,4,3)
    p,i=Trs.PickWord(y,2)
    print(p)
    print(i)
    Trs.Do()
import Trainer
def Test33():
    Trainer.Train(10)
    #print(Trainer.Model)
def Test34():
    Trs=TT.GreedyTranslator("Model/Trans.0","Output/predict")
    print(Trs.Model)
    Trs.Do()
#Test1()
#Test2()
#Test3()
#Test4()
#Test5()
#Test6()
#Test7()
#Test8()
#Test9()
#Test10()
#Test11()
#Test12()
#Test13()
#Test14()
#Test15()
#Test16()
#Test17()
#Test18()
#Test19()
#Test20()
#Test21()
#Test22()
#Test23()
#Test24()
#Test25()
#Test26()
#Test27()
#Test28()
#Test29()
#Test30()
#Test31()
Test33()
#Test34()