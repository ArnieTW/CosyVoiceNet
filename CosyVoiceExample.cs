// Exported from CosyVoice/cosyvoice/cli/cosyvoice.py
using System;
using System.IO;
using System.Collections.Generic;
using CosyVoiceNet;
using CosyVoiceNet.cli;

namespace CosyVoiceApp
{
    public static class CosyVoiceExample
    {
        public static void RunAllExamples()
        {

            // Existing example calls
            //RunCosyVoiceExample();
            //CosyVoice2Example();
            Console.WriteLine("Run");
            CosyVoice3Example();
        }

        // Realign to original state and move outputs to 'TestOutputs' folder
        public static void RunCosyVoiceExample()
        {
            var assets = CosyVoiceApp.AppHost.Assets;

            // Retrieve the zero_shot_prompt asset dynamically
            var zs = assets.ContainsKey("zero_shot_prompt.wav") 
                ? assets["zero_shot_prompt.wav"] 
                : throw new FileNotFoundException("The zero_shot_prompt.wav asset was not found.");

            var cl = assets.ContainsKey("cross_lingual_prompt.wav") 
                ? assets["cross_lingual_prompt.wav"] 
                : throw new FileNotFoundException("The cross_lingual_prompt.wav asset was not found.");

            var cosyvoice = new CosyVoiceNet.cli.CosyVoice("cosyvoice/CosyVoice-300M-SFT");
            Console.WriteLine($"Created model type: {cosyvoice.GetType()}");
            Console.WriteLine(string.Join(", ", cosyvoice.ListAvailableSpks()));
            int outerIndex = 0;
            foreach (var chunk in cosyvoice.InferenceSft("你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？", "中文女", stream: false))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"sft_{outerIndex}.wav"), chunk.TtsSpeech);
                outerIndex++;
            }

            var cosyvoice2 = new CosyVoiceNet.cli.CosyVoice("cosyvoice/CosyVoice-300M");
            Console.WriteLine($"Created model type: {cosyvoice2.GetType()}");
            outerIndex = 0;
            foreach (var chunk in cosyvoice2.InferenceZeroShot("收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "希望你以后能够做的比我还好呦。", zs))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"zero_shot_{outerIndex}.wav"), chunk.TtsSpeech);
                outerIndex++;
            }

            outerIndex = 0;
            foreach (var chunk in cosyvoice2.InferenceCrossLingual("<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that's coming into the family is a reason why sometimes we don't buy the whole thing.", cl))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"cross_lingual_{outerIndex}.wav"), chunk.TtsSpeech);
                outerIndex++;
            }

            outerIndex = 0;
            foreach (var chunk in cosyvoice2.InferenceVc(cl, zs))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"vc_{outerIndex}.wav"), chunk.TtsSpeech);
                outerIndex++;
            }

            var cosyvoice3 = new CosyVoiceNet.cli.CosyVoice3("cosyvoice/CosyVoice-300M-Instruct");
            Console.WriteLine($"Created model type: {cosyvoice3.GetType()}");
            int i = 0;
            foreach (var chunk in cosyvoice3.InferenceInstruct("在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。", "中文男", "Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.<|endofprompt|>"))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"instruct_{i}.wav"), chunk.TtsSpeech);
                i++;
            }
        }

        // Explicitly cast to CosyVoice2 before calling InferenceInstruct2
        public static void CosyVoice2Example()
        {
            var modelPath = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512";
            var cosyvoice = new CosyVoiceNet.cli.CosyVoice2(modelPath);

            int i = 0;
            var assets = CosyVoiceApp.AppHost.Assets;
            var zs = assets.ContainsKey("zero_shot_prompt") ? assets["zero_shot_prompt"] : "./asset/zero_shot_prompt.wav";
            foreach (var chunk in cosyvoice.InferenceZeroShot("收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "希望你以后能够做的比我还好呦。", zs))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"zero_shot_{i}.wav"), ((CosyVoiceNet.cli.CosyVoice.InferenceResult)chunk).TtsSpeech);
                i++;
            }

            var added = cosyvoice.AddZeroShotSpk("希望你以后能够做的比我还好呦。", zs, "my_zero_shot_spk");
            if (!added) throw new Exception("Failed to add zero shot speaker");

            i = 0;
            foreach (var chunk in cosyvoice.InferenceZeroShot("收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "", "", stream: false))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"zero_shot_{i}.wav"), ((CosyVoiceNet.cli.CosyVoice.InferenceResult)chunk).TtsSpeech);
                i++;
            }

            cosyvoice.SaveSpkInfo();

            i = 0;
            foreach (var chunk in cosyvoice.InferenceCrossLingual("在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。", zs))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"fine_grained_control_{i}.wav"), ((CosyVoiceNet.cli.CosyVoice.InferenceResult)chunk).TtsSpeech);
                i++;
            }

            i = 0;
            foreach (var chunk in cosyvoice.InferenceInstruct2("收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "用四川话说这句话<|endofprompt|>", zs))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"instruct_{i}.wav"), ((CosyVoiceNet.cli.CosyVoice.InferenceResult)chunk).TtsSpeech);
                i++;
            }

            // bistream example
            IEnumerable<string> TextGenerator()
            {
                yield return "收到好友从远方寄来的生日礼物，";
                yield return "那份意外的惊喜与深深的祝福";
                yield return "让我心中充满了甜蜜的快乐，";
                yield return "笑容如花儿般绽放。";
            }

            i = 0;
            foreach (var chunk in cosyvoice.InferenceZeroShot(string.Join("", TextGenerator()), "希望你以后能够做的比我还好呦。", zs, stream: false))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"zero_shot_bistream_{i}.wav"), ((CosyVoiceNet.cli.CosyVoice.InferenceResult)chunk).TtsSpeech);
                i++;
            }
        }

        // Explicitly use CosyVoice3 in CosyVoice3Example
        public static void CosyVoice3Example()
        {
            var assets = CosyVoiceApp.AppHost.Assets;
            var zs = assets.ContainsKey("zero_shot_prompt.wav")
                ? assets["zero_shot_prompt.wav"]
                : throw new FileNotFoundException("The zero_shot_prompt.wav asset was not found.");

            var cl = assets.ContainsKey("cross_lingual_prompt.wav")
                ? assets["cross_lingual_prompt.wav"]
                : throw new FileNotFoundException("The cross_lingual_prompt.wav asset was not found.");

            var cosyvoice = new CosyVoiceNet.cli.CosyVoice3("FunAudioLLM/Fun-CosyVoice3-0.5B");
            int i = 0;
            foreach (var chunk in cosyvoice.InferenceZeroShot("八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。", "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。", zs, stream: false))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"zero_shot_{i}.wav"), chunk.TtsSpeech);
                i++;
            }

            i = 0;
            foreach (var chunk in cosyvoice.InferenceCrossLingual("You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]", zs, stream: false))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"fine_grained_control_{i}.wav"), chunk.TtsSpeech);
                i++;
            }

            i = 0;
            foreach (var chunk in cosyvoice.InferenceInstruct2("好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。", "You are a helpful assistant. 请用广东话表达。<|endofprompt|>", zs, stream: false))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"instruct_{i}.wav"), chunk.TtsSpeech);
                i++;
            }

            i = 0;
            foreach (var chunk in cosyvoice.InferenceInstruct2("收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>", zs, stream: false))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"instruct_{i}.wav"), chunk.TtsSpeech);
                i++;
            }

            i = 0;
            foreach (var chunk in cosyvoice.InferenceZeroShot("高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。", "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。", zs, stream: false))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"hotfix_{i}.wav"), chunk.TtsSpeech);
                i++;
            }

            i = 0;
            foreach (var chunk in cosyvoice.InferenceCrossLingual("You are a helpful assistant.<|endofprompt|>レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ モノ デ ワ ナイ、プラトン ノ イウ ゴトク ヒ ユー ガ ユー デ アル。", zs, stream: false))
            {
                File.WriteAllBytes(Path.Combine("outputs", $"japanese_{i}.wav"), chunk.TtsSpeech);
                i++;
            }
        }

        // Add a minimal test case to verify CosyVoice class instantiation and method calls
        public static void TestCosyVoice()
        {
            try
            {
                var cosyvoice = new CosyVoiceNet.cli.CosyVoice("cosyvoice/CosyVoice-300M");
                Console.WriteLine("CosyVoice instance created successfully.");
                Console.WriteLine(string.Join(", ", cosyvoice.ListAvailableSpks()));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}
