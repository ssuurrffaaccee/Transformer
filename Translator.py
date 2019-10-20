import Transformer as Tr
import torch
import copy


def Max(StepProOutput):
    # StepProOutput=[B,VocabLen]
    MaxInd = torch.argmax(StepProOutput, dim=-1)
    Index = MaxInd.tolist()
    ProValue = []
    for Ind, Pro in zip(MaxInd, StepProOutput):
        ProValue.append(torch.Tensor(Pro.index_select(0, Ind)))
    return torch.cat(ProValue), Index


class TranslateOutput:
    def __init__(self, TgtDict, MaxLength):
        self.Dict = TgtDict
        self.MaxLength = MaxLength
        self.IndexSent = None
        self.ReachMaxLength = None
        self.HadEnd = None

    def Init(self, BatchSampleNum):
        self.IndexSent = [[1] for i in range(BatchSampleNum)]
        self.ReachMaxLength = [False for i in range(BatchSampleNum)]
        self.HadEnd = [False for i in range(BatchSampleNum)]
        return self

    def Add(self, BatchedIndex):
        EOS = 2
        assert len(BatchedIndex) == len(self.IndexSent)
        for Ind, Index in enumerate(BatchedIndex):
            if self.ReachMaxLength[Ind] != True and self.HadEnd[Ind] != True:
                self.IndexSent[Ind].append(Index)
                if Index == EOS:
                    self.HadEnd[Ind] = True
                if len(self.IndexSent[Ind]) == self.MaxLength:
                    self.ReachMaxLength[Ind] = True

    def AllFinish(self):
        Finished = True
        for i in range(len(self.HadEnd)):
            Finished = Finished and (
                self.HadEnd[i] == True or self.ReachMaxLength[i] == True)
        return Finished

    def GetCurrentIndexTensor(self):
        CopyedIndexSent = copy.deepcopy(self.IndexSent)
        for Sent in CopyedIndexSent:
            for i in range(self.MaxLength-len(Sent)):
                Sent.append(0)
        return torch.LongTensor(CopyedIndexSent)

    def GetWordSent(self):
        WordSents = []
        for IndexSent in self.IndexSent:
            WordSents.append([self.Dict.GetWord(Index) for Index in IndexSent])
        return [" ".join(Sent) for Sent in WordSents]

    def ToFile(self, Path):
        with open(Path, "a", encoding="utf-8") as Fd:
            for Sent in self.GetWordSent():
                Fd.write(Sent+"\n")


class GreedyTranslator:
    def __init__(self, ModelPath, ResultPath):
        self.UseGPU=torch.cuda.is_available()
        Trans, BatchDatas, SrcDict, TgtDict, MaxLength = Tr.TestBuildTransformer()
        self.Model = Trans
        self.BatchDatas = BatchDatas
        self.SrcDict = SrcDict
        self.TgtDict = TgtDict
        self.MaxLength = MaxLength
        self.ResultPath = ResultPath
        self.Model.Load(ModelPath)
        self.Model.Eval()
        if self.UseGPU:
            self.Model=self.Model.cuda()

    def Do(self):
        for BatchInd, Batch in enumerate(self.BatchDatas):
            SrcSent = Batch["SrcSent"]
            SrcLength = Batch["SrcLength"]
            BatchSampleNum = SrcSent.size()[0]
            #print(type(BatchSampleNum))
            CurrentBatchOuput = TranslateOutput(
                self.TgtDict, self.MaxLength).Init(BatchSampleNum)
            TgtIndexSent = CurrentBatchOuput.GetCurrentIndexTensor()
            SrcMask = Tr.BatchLengthToBoolTensorMask(SrcLength, self.MaxLength)
            for Step in range(self.MaxLength):
                TgtMask = self.GetTgtMask(Step, BatchSampleNum)
                if self.UseGPU:
                    SrcSent=SrcSent.cuda()
                    TgtMask=TgtMask.cuda()
                    SrcMask=SrcMask.cuda()
                    TgtMask=TgtMask.cuda()
                ProOutput = self.Model(SrcSent, TgtIndexSent, SrcMask, TgtMask)
                ProOutput=ProOutput.cpu()
                LocalMaxPro, Idx = self.PickWord(ProOutput, Step)
                CurrentBatchOuput.Add(Idx)
                if CurrentBatchOuput.AllFinish():
                    print("Appending Ouput of Batch "+str(BatchInd))
                    CurrentBatchOuput.ToFile(self.ResultPath)
                    break
                TgtIndexSent = CurrentBatchOuput.GetCurrentIndexTensor()

    def GetTgtMask(self, Step, BatchSampleNum):
        #print(self.MaxLength)
        #print(type(BatchSampleNum))
        StepTgtMask = [[1 if i <= Step else 0 for i in range(
            self.MaxLength)] for i in range(BatchSampleNum)]
        return torch.BoolTensor(StepTgtMask)

    def PickWord(self, ProOuput, Step):
        LocalProValue, Index = Max(ProOuput[:, Step, :])
        return LocalProValue, Index
