using System;
using System.Collections.Generic;
using System.Linq;

namespace CosyVoiceNet.Utils
{
    // Equivalent Python file: cosyvoice/utils/scheduler.py
    public static class Scheduler
    {
        public static double WarmupLr(double baseLr, int stepNum, double warmupSteps)
        {
            if (warmupSteps == 0) return baseLr * Math.Pow(stepNum, -0.5);
            return baseLr * Math.Sqrt(warmupSteps) * Math.Min(Math.Pow(stepNum, -0.5), stepNum * Math.Pow(warmupSteps, -1.5));
        }

        public static double[] WarmupLR_GetLrs(double[] baseLrs, int lastEpoch, double warmupSteps)
        {
            var stepNum = lastEpoch + 1;
            if (warmupSteps == 0)
            {
                return baseLrs.Select(lr => lr * Math.Pow(stepNum, -0.5)).ToArray();
            }
            return baseLrs.Select(lr => lr * Math.Sqrt(warmupSteps) * Math.Min(Math.Pow(stepNum, -0.5), stepNum * Math.Pow(warmupSteps, -1.5))).ToArray();
        }

        public static double[] WarmupPolicy_GetLrs(double[] baseLrs, int lastEpoch, int warmupSteps, int maxSteps, double minLr)
        {
            int step = lastEpoch;
            if (step <= warmupSteps && warmupSteps > 0)
            {
                var lrVal = (step + 1) / (double)(warmupSteps + 1);
                return baseLrs.Select(initial => initial * lrVal).ToArray();
            }
            if (step > maxSteps)
            {
                return baseLrs.Select(_ => minLr).ToArray();
            }
            return baseLrs;
        }

        public static double[] SquareRootConstantPolicy_GetLrs(double[] baseLrs, int lastEpoch, int constantSteps, int maxSteps, double minLr)
        {
            int step = lastEpoch;
            if (step <= constantSteps) return baseLrs.Select(_ => 1.0 / Math.Sqrt(constantSteps)).ToArray();
            if (step > maxSteps) return baseLrs.Select(_ => minLr).ToArray();
            return baseLrs;
        }

        public static double[] WarmupHoldPolicy_GetLrs(double[] baseLrs, int lastEpoch, int warmupSteps, int holdSteps, int maxSteps, double minLr)
        {
            int step = lastEpoch;

            if (step <= warmupSteps && warmupSteps > 0)
            {
                var lrVal = (step + 1) / (double)(warmupSteps + 1);
                return baseLrs.Select(initial => initial * lrVal).ToArray();
            }

            if (step >= warmupSteps && step < holdSteps)
            {
                return baseLrs;
            }

            if (step > maxSteps)
            {
                return baseLrs.Select(_ => minLr).ToArray();
            }

            return baseLrs;
        }

        public static double[] WarmupAnnealHoldPolicy_GetLrs(double[] baseLrs, int lastEpoch, int warmupSteps, int constantSteps, int maxSteps, double minLr)
        {
            int step = lastEpoch;

            if (warmupSteps > 0 && step <= warmupSteps)
            {
                var lrVal = (step + 1) / (double)(warmupSteps + 1);
                return baseLrs.Select(initial => initial * lrVal).ToArray();
            }

            if (constantSteps > 0 && (warmupSteps + constantSteps) < step && step <= maxSteps)
            {
                return baseLrs.Select(_ => minLr).ToArray();
            }

            if (step > maxSteps)
            {
                return baseLrs.Select(_ => minLr).ToArray();
            }

            return baseLrs;
        }

        public static double SquareAnnealing(double initialLr, int step, int maxSteps, double minLr)
        {
            var mult = Math.Pow((maxSteps - step) / (double)maxSteps, 2);
            var outLr = initialLr * mult;
            return Math.Max(outLr, minLr);
        }

        public static double SquareRootAnnealing(double initialLr, int step, int maxSteps, double minLr)
        {
            var mult = Math.Sqrt((maxSteps - step) / (double)maxSteps);
            var outLr = initialLr * mult;
            return Math.Max(outLr, minLr);
        }

        public static double CosineAnnealing(double initialLr, int step, int maxSteps, double minLr)
        {
            var mult = 0.5 * (1 + Math.Cos(Math.PI * step / maxSteps));
            var outLr = (initialLr - minLr) * mult + minLr;
            return outLr;
        }

        public static double LinearWarmupWithCosineAnnealing(double maxLr, int warmupSteps, int step, int decaySteps, double minLr)
        {
            if (warmupSteps > 0 && step <= warmupSteps)
            {
                return maxLr * step / (double)warmupSteps;
            }

            if (step > warmupSteps + decaySteps)
            {
                return minLr;
            }

            var numSteps = step - warmupSteps;
            var decayRatio = numSteps / (double)decaySteps;
            var deltaLr = maxLr - minLr;
            var coeff = 0.5 * (Math.Cos(Math.PI * decayRatio) + 1.0);

            return minLr + coeff * deltaLr;
        }
    }
}
