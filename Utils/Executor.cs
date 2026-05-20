// Equivalent Python file: cosyvoice/utils/executor.py
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace CosyVoiceNet.Utils
{
    public class Executor
    {
        public bool Gan { get; }
        public dynamic RefModel { get; }
        public dynamic DpoLoss { get; }
        public int Step { get; private set; } = 0;
        public int Epoch { get; private set; } = 0;
        public int Rank { get; }
        public Device Device { get; }

        public Executor(bool gan = false, dynamic refModel = null, dynamic dpoLoss = null)
        {
            Gan = gan;
            RefModel = refModel;
            DpoLoss = dpoLoss;
            Rank = int.TryParse(Environment.GetEnvironmentVariable("RANK"), out var r) ? r : 0;
            Device = torch.device($"cuda:{Rank}");
        }

        public void SetStep(int step) => Step = step;
        public void SetEpoch(int epoch) => Epoch = epoch;

        public void TrainOneEpoch(dynamic model, dynamic optimizer, dynamic scheduler, dynamic trainDataLoader, dynamic cvDataLoader, dynamic writer, Dictionary<string, object> infoDict, dynamic scaler, dynamic groupJoin, dynamic refModel = null)
        {
            model.train();
            if (RefModel != null)
            {
                RefModel.eval();
            }

            foreach (var batch in trainDataLoader)
            {
                infoDict["tag"] = "TRAIN";
                infoDict["step"] = Step;
                infoDict["epoch"] = Epoch;

                var inputs = batch.inputs;
                var targets = batch.targets;
                optimizer.zero_grad();
                var outputs = model.forward(inputs);
                var loss = torch.nn.functional.cross_entropy(outputs, targets);
                loss.backward();
                optimizer.step();
                Step++;
            }
        }

        public void TrainOneEpochGan(dynamic model, dynamic optimizer, dynamic scheduler, dynamic optimizerD, dynamic schedulerD, dynamic trainDataLoader, dynamic cvDataLoader, dynamic writer, Dictionary<string, object> infoDict, dynamic scaler, dynamic groupJoin)
        {
            model.train();

            foreach (var batch in trainDataLoader)
            {
                infoDict["tag"] = "TRAIN";
                infoDict["step"] = Step;
                infoDict["epoch"] = Epoch;

                var inputs = batch.inputs;
                var targets = batch.targets;
                optimizer.zero_grad();
                var outputs = model.forward(inputs);
                var loss = torch.nn.functional.cross_entropy(outputs, targets);
                loss.backward();
                optimizer.step();
                Step++;
            }
        }

        public void Cv(dynamic model, dynamic cvDataLoader, dynamic writer, Dictionary<string, object> infoDict, bool onBatchEnd = true)
        {
            Console.WriteLine($"Epoch {Epoch} Step {Step + 1} onBatchEnd {onBatchEnd} CV rank {Rank}");
            model.eval();
            var totalNumUtts = 0;
            var totalLossDict = new Dictionary<string, double>();

            foreach (var batch in cvDataLoader)
            {
                infoDict["tag"] = "CV";
                infoDict["step"] = Step;
                infoDict["epoch"] = Epoch;

                var numUtts = batch["utts"].Length;
                totalNumUtts += numUtts;

                var lossDict = model.forward(batch);
                foreach (var kv in lossDict)
                {
                    if (!totalLossDict.ContainsKey(kv.Key))
                    {
                        totalLossDict[kv.Key] = 0.0;
                    }
                    totalLossDict[kv.Key] += kv.Value * numUtts;
                }
            }

            foreach (var kv in totalLossDict)
            {
                totalLossDict[kv.Key] /= totalNumUtts;
            }

            Console.WriteLine("Validation complete. Losses:");
            foreach (var kv in totalLossDict)
            {
                Console.WriteLine($"{kv.Key}: {kv.Value}");
            }
        }
    }
}
