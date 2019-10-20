import torch
import copy
class VocabularyDict:
    #0=<PAD>
    #1=<UNK>
    #2=<BOS>
    #3=<EOS>
    def __init__(self):
        self.IndexToWord={}
        self.WordToIndex={}
        self.WordNum=0
    def AddWord(self,Word):
        try:
            self.WordToIndex[Word]
            return 
        except:
            self.WordToIndex[Word]=self.WordNum
            self.IndexToWord[self.WordNum]=Word
            self.WordNum=self.WordNum+1
    def GetIndex(self,Word):
        try:
            return self.WordToIndex[Word]
        except:
            return self.WordToIndex["<UNK>"]
    def GetWord(self,Index):
        try:
            return self.IndexToWord[Index]
        except:
            return self.IndexToWord[self.WordToIndex["<UNK>"]]
    def VocabularySize(self):
        return self.WordNum

def LoadSentences(Path):
    Sentences=[]
    with open(Path,"r",encoding="utf-8") as Fd:
        for Line in Fd:
            Line=Line.strip().split(" ")
            Sentences.append(Line)
    return Sentences
def Split(StringSents):
    StringSents=StringSents.strip().split(" ")
    return [StringSents]
def LoadVocabulary(Path):
    Dict=VocabularyDict()
    with open(Path,"r",encoding="utf-8") as Fd:
        for Line in Fd:
            Word=Line.strip()
            Dict.AddWord(Word)
    return Dict
def PaddingSentences(Sentences,MaxLength):
    def AddBOSAndEOS(Sent):
        Sent.append("<EOS>")
        Sent.reverse()
        Sent.append("<BOS>")
        Sent.reverse()
        return Sent
    PaddedSentences=[]
    Length=[]
    for Sent in Sentences:
        if len(Sent)>MaxLength-2:
            Sent=Sent[:MaxLength-2]
            PaddedSentences.append(AddBOSAndEOS(Sent))
            Length.append(MaxLength)
            continue
        Sent=AddBOSAndEOS(Sent)
        Length.append(len(Sent))
        Padding=["<PAD>" for i in range(MaxLength-len(Sent))]
        Sent.extend(Padding)
        PaddedSentences.append(Sent)
    return PaddedSentences,Length

def ChangePaddedSentencesToInd(PaddedSentences,Dict):
    print("Begin transform  word to ind")
    IndexSentences=[]
    for Line in PaddedSentences:
        IndexSentences.append([Dict.GetIndex(Word) for Word in Line])
    print("Transform finished")
    return IndexSentences
def LoadData(SentencePath,VocabPath,MaxLength):
    print("Begin load Dict")
    Dict=LoadVocabulary(VocabPath)
    print("Dict load finished")
    print("Begin load sentence")
    PaddedSentences,Length=PaddingSentences(LoadSentences(SentencePath),MaxLength)
    print("Sentence load finished")
    return ChangePaddedSentencesToInd(PaddedSentences,Dict),Length,Dict

class TrainCorpusDataset(torch.utils.data.IterableDataset):
     def __init__(self,SrcSentenceInd,SrcLength,TgtSentenceInd,TgtLength):
         super().__init__()
         self.Data=list(zip(list(zip(SrcSentenceInd,SrcLength)),list(zip(TgtSentenceInd,TgtLength))))
     def __iter__(self):
        for Elem in self.Data:
            yield Elem
class TestCorpusDataset(torch.utils.data.IterableDataset):
     def __init__(self,SrcSentenceInd,SrcLength):
         super().__init__()
         self.Data=list(zip(SrcSentenceInd,SrcLength))
     def __iter__(self):
        for Elem in self.Data:
            yield Elem
class OnlineTrainCorpusDataset(torch.utils.data.IterableDataset):
     def __init__(self,SrcSentsPath,TgtSentsPath,MaxLength,SrcDict,TgtDict):
         super().__init__()
         self.SrcSentsPath=SrcSentsPath
         self.TgtSentsPath=TgtSentsPath
         self.MaxLength=MaxLength
         self.SrcDict=SrcDict
         self.TgtDict=TgtDict
     def __iter__(self):
        with open(self.SrcSentsPath,"r",encoding="utf-8") as SrcFd:
            with open(self.TgtSentsPath,"r",encoding="utf-8") as TgtFd:
                for SrcSent,TgtSent in zip(SrcFd,TgtFd):
                    SrcPaddedSentence,SrcLength=PaddingSentences(Split(SrcSent),self.MaxLength)
                    TgtPaddedSentence,TgtLength=PaddingSentences(Split(TgtSent),self.MaxLength)
                    SrcIndSentence=ChangePaddedSentencesToInd(SrcPaddedSentence,self.SrcDict)
                    TgtIndSnetence=ChangePaddedSentencesToInd(TgtPaddedSentence,self.TgtDict)
                    yield ((SrcIndSentence[0],SrcLength[0]),(TgtIndSnetence[0],TgtLength[0]))
def TrainDataLoaderCreator(Dataset,BatchSize):
    def CollateFunction(Batch):
        OutputBatch={"SrcSent":[],"SrcLength":[],"TgtSent":[],"TgtLength":[]}
        for Elem in Batch:
            OutputBatch["SrcSent"].append(Elem[0][0])
            OutputBatch["SrcLength"].append(Elem[0][1])
            OutputBatch["TgtSent"].append(Elem[1][0])
            OutputBatch["TgtLength"].append(Elem[1][1])
        OutputBatch["SrcSent"]=torch.LongTensor(OutputBatch["SrcSent"])
        OutputBatch["TgtSent"]=torch.LongTensor(OutputBatch["TgtSent"])
        return OutputBatch
    return torch.utils.data.DataLoader(Dataset,batch_size=BatchSize,collate_fn=CollateFunction,num_workers=0)
def TestDataLoaderCreator(Dataset,BatchSize):
    def CollateFunction(Batch):
        OutputBatch={"SrcSent":[],"SrcLength":[]}
        for Elem in Batch:
            OutputBatch["SrcSent"].append(Elem[0])
            OutputBatch["SrcLength"].append(Elem[1])
        OutputBatch["SrcSent"]=torch.LongTensor(OutputBatch["SrcSent"])
        return OutputBatch
    return torch.utils.data.DataLoader(Dataset,batch_size=BatchSize,collate_fn=CollateFunction,num_workers=0)
def SimpleInfinityLooper(Iter):
    while True:
        for Elem in Iter:
            yield Elem     
def ShardLooper(BatchSize,EpochNum,SrcDict,TgtDict,MaxLength):
    def Load(Path,Dict,MaxLength):
        Sentences=LoadSentences(Path)
        CurrentSentenceNum=len(Sentences)
        PaddedSentence,Length=PaddingSentences(Sentences,MaxLength)
        PaddedIndSentence=ChangePaddedSentencesToInd(PaddedSentence,Dict)
        return CurrentSentenceNum,PaddedIndSentence,Length
    def GeneratePath():
        Total=100
        return [("Data/src.sents-"+str(i)+"-of-"+str(Total),"Data/tgt.sents-"+str(i)+"-of-"+str(Total)) for i in range(Total)]
    Count=0
    EpochFinished=False
    AcumulatedEopch=0
    for SrcPath,TgtPath in SimpleInfinityLooper(GeneratePath()):
        if EpochFinished:
            break
        print("Loading Shard :"+str(Count))
        CurrentSrcSentenceNum,SrcPaddedIndSentence,SrcLength=Load(SrcPath,SrcDict,MaxLength)
        CurrentTgtSentenceNum,TgtPaddedIndSentence,TgtLength=Load(TgtPath,TgtDict,MaxLength)
        assert CurrentSrcSentenceNum==CurrentTgtSentenceNum
        print("Shard loading finished")
        Count=Count+1
        print("Building Dataset")
        CurrentDataSet=TrainCorpusDataset(SrcPaddedIndSentence,SrcLength,TgtPaddedIndSentence,TgtLength)
        print("Dataset Building finished")
        print("Building Dataloader")
        CurrentDataLoader=TrainDataLoaderCreator(CurrentDataSet,BatchSize)
        print("DataLoader building finished")
        for Batch in CurrentDataLoader:
            AcumulatedEopch=AcumulatedEopch+1
            if AcumulatedEopch==EpochNum:
                EpochFinished=True
                break
            yield Batch
def TrainLoop(SrcSentsPath,TgtSentsPath,MaxLength,SrcDict,TgtDict,BatchSize):
    Dataset=OnlineTrainCorpusDataset(SrcSentsPath,TgtSentsPath,MaxLength,SrcDict,TgtDict)
    return TrainDataLoaderCreator(Dataset,BatchSize)