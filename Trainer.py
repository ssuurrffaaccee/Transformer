import torch
from Transformer import TrainBuildTransformer,BatchLengthToBoolTensorMask,LabelSmooth
from DataLoad import TrainLoop
import math
class Adam:
    def __init__(self,Model):
        self.Optim=torch.optim.Adam(Model.parameters(), lr=0,eps=10e-9, betas=(0.9, 0.98))
    def Step(self,BatchInd,EmbeddingSize):
        LearningRate=self.Rate(BatchInd,EmbeddingSize)
        for Para in self.Optim.param_groups:
            Para["lr"]=LearningRate
        self.Optim.step()
    def Rate(self,BatchInd,EmbeddingSize):
        return pow(EmbeddingSize,0.5)*min(pow(BatchInd+1,-0.5),(BatchInd+1)*pow(4000,-1.5))
    def ZeroGrad(self):
        self.Optim.zero_grad()
class Trainer:
    def __init__(self):
        self.UseGPU=torch.cuda.is_available()
        Trans,BatchSize,SrcDict,TgtDict,MaxLength,EmbeddingSize=TrainBuildTransformer()
        self.Model=Trans
        self.Model.Train()
        if self.UseGPU:
            print("Sending Model to GPU")
            self.Model=self.Model.cuda()
            print("Sending finished")
        self.BatchSize=BatchSize
        self.TgtDict=TgtDict
        self.SrcDict=SrcDict
        self.MaxLength=MaxLength
        self.EmbeddingSize=EmbeddingSize
        self.LossFunc=torch.nn.KLDivLoss(reduction="batchmean")
        self.LabelSmooth=LabelSmooth(self.TgtDict.VocabularySize())
        if self.UseGPU:
            self.LossFunc=self.LossFunc.cuda()
            self.LabelSmooth=self.LabelSmooth.cuda()
        print("Building Optim")
        self.Optim=Adam(self.Model)
    def Train(self,Epoch):
        for BatchInd,Batch in enumerate(TrainLoop("Data/src.sents","Data/tgt.sents",self.MaxLength,self.SrcDict,self.TgtDict,self.BatchSize)):
            print("Batch:"+str(BatchInd))
            self.Optim.ZeroGrad()
            SrcSent=Batch["SrcSent"]
            SrcLength=Batch["SrcLength"]
            TgtSent=Batch["TgtSent"]
            TgtLength=Batch["TgtLength"]
            SrcMask=BatchLengthToBoolTensorMask(SrcLength,self.MaxLength)
            TgtMask=BatchLengthToBoolTensorMask(TgtLength,self.MaxLength)
            if self.UseGPU:
                print("Sending Input to GPU")
                SrcSent=SrcSent.cuda()
                SrcMask=SrcMask.cuda()
                TgtSent=TgtSent.cuda()
                TgtMask=TgtMask.cuda()
            TgtSentSoomthLabel=self.LabelSmooth(TgtSent)
            print("Model Computing ......")
            ProOutput=self.Model(SrcSent,TgtSent,SrcMask,TgtMask)
            Loss=self.LossFunc(torch.log(ProOutput),TgtSentSoomthLabel)
            Loss.backward()
            Loss=Loss.cpu()
            self.Optim.Step(BatchInd,self.EmbeddingSize)
            print("Loss: "+str(Loss))
            if BatchInd%1000==0:
                print("Saveing Model."+str(BatchInd))
                self.Model.Save("Model/Trans."+str(BatchInd))
    #def GetSmoothLabel(self,TgtSent):
    #    def NumToSmoothLabel(Num,TgtVocabLength):
    #        SmootLabel=[0.1/TgtVocabLength for i in range(TgtVocabLength)]
    #        SmootLabel[Num]=SmootLabel[Num]+0.9
    #        return SmootLabel
    #    def SentToSoomthLabel(Sent,TgtVocabLength):
    #        return [NumToSmoothLabel(Num,self.TgtDict.VocabularySize()) for Num in Sent]
    #    TgtIndexSent=TgtSent.tolist()
    #    TgtSentSmoothLabel=torch.Tensor([SentToSoomthLabel(Sent,self.MaxLength) for Sent in TgtIndexSent])
    #    return TgtSentSmoothLabel
    def CheckExistOldModel(self):
        pass
        
def Train(Epoch=None):
    TrainProcess=Trainer()
    print(TrainProcess.Model)
    TrainProcess.Train(Epoch)